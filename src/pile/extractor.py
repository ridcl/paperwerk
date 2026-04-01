import base64
from collections.abc import Sequence

from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
from io import BytesIO
from PIL import Image, ImageDraw


def pil_to_base64_url(image: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode("utf-8")
    media_type = f"image/{format.lower()}"
    return f"data:{media_type};base64,{b64_data}"


class Grounded(BaseModel):
    key: str
    value: str
    box_2d: list[int]  # 0..1000, [x0, y0, x1, y1]

    @property
    def box_f(self):
        """Bounding box as a list of floating point numbers in interval 0..1"""
        return [u / 1000 for u in self.box_2d]

    @property
    def x0(self):
        return self.box_2d[0]

    @property
    def y0(self):
        return self.box_2d[1]

    @property
    def x1(self):
        return self.box_2d[2]

    @property
    def y1(self):
        return self.box_2d[3]


class _ExtractionResult(BaseModel):
    items: list[Grounded]


def visualize(image: Image.Image, items: list[Grounded]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    W, H = image.size
    for item in items:
        x0, y0, x1, y1 = item.box_2d
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 12)), item.key, fill="red")
    return image


class Extractor:

    def __init__(
        self,
        model: str = "/data/models/kvp10k-qwen3vl-4b/",
        base_url: str = "http://localhost:8000/v1",
    ):
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="")

    def __repr__(self):
        return "Extractor()"

    @staticmethod
    def _make_prompt(keys: list[str]) -> str:
        key_list = "\n".join(f"- {k}" for k in keys)
        return (
            "Extract the following key-value pairs from this document. "
            'Return a JSON list where each item has "key", "value", and "bbox" fields. '
            'The "box_2d" is the bounding box of the value text in the format '
            '{"box_2d": [x0, y0, x1, y1]} with coordinates in 0–1000 range.\n'
            f"Keys to extract:\n{key_list}"
        )

    @staticmethod
    def _make_conversation(image: Image.Image, keys: list[str]) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": Extractor._make_prompt(keys)},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_base64_url(image)},
                    },
                ],
            }
        ]

    def __call__(self, path: str, keys: Sequence[str]) -> list[Grounded]:
        image = Image.open(path)
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self._make_conversation(image, keys),
            response_format=_ExtractionResult,
        )
        return completion.choices[0].message.parsed.items


def main():
    extractor = Extractor()

    path = "/data/taxes.jpeg"
    keys = ["total_tax_box1", "payable_tax", "ssn", "reciavable_tax", "name"]
    items = extractor(path, keys)

    image = Image.open(path)
    visualize(image, items).save("output/out.jpeg")
