import base64
import json
from collections.abc import Sequence
import os
from typing import Optional

from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image
from io import BytesIO
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


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
    box_2d: list[int]  # [x0, y0, x1, y1], 0..1000
    meta: dict = Field(default_factory=dict, repr=False)

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


_GROUNDED_LIST_SCHEMA = {
    "type": "array",
    "items": Grounded.model_json_schema(),
}


class Extractor:

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
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

    def extract(self, image: Image.Image, keys: Sequence[str]) -> list[Grounded]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._make_conversation(image, keys),
            extra_body={"guided_json": _GROUNDED_LIST_SCHEMA},
        )
        data = json.loads(completion.choices[0].message.content)
        return [Grounded(**item) for item in data]

    def __call__(self, path: str, keys: Sequence[str]) -> list[Grounded]:
        _, ext = os.path.splitext(path)
        if ext in IMAGE_EXTENSIONS:
            items = self.extract(Image.open(path), keys)
            for item in items:
                item.meta["filename"] = path
            return items
        elif ext == ".pdf":
            images = convert_from_path(path, fmt="jpeg")
            items = []
            for page, image in enumerate(images):
                page_items = self.extract(image, keys)
                for item in page_items:
                    item.meta = {"filename": path, "page": page}
                items.extend(page_items)
            return items
        else:
            raise ValueError(f"Unsupported file type: {ext}")


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


def main():
    extractor = Extractor(
        base_url="http://localhost:8000/v1", model="/data/models/kvp10k-qwen3vl-4b/"
    )

    path = "/data/taxes.jpeg"
    # path = "/data/Documents/ruling/Zhabinski, A.V. - Yandex.pdf"
    keys = ["total_tax_box1", "payable_tax", "ssn", "reciavable_tax", "name"]
    # keys = ["date", "reference_number", "phone_number"]
    items = extractor(path, keys)

    image = Image.open(path)
    visualize(image, items).save("output/out.jpeg")
