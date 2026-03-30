import base64
import json

from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
from PIL import Image, ImageDraw


def _make_prompt(keys: list[str]) -> str:
    key_list = "\n".join(f"- {k}" for k in keys)
    return (
        "Extract the following key-value pairs from this document. "
        'Return a JSON list where each item has "key", "value", and "bbox" fields. '
        'The "box_2d" is the bounding box of the value text in the format '
        '{"box_2d": [x0, y0, x1, y1]} with coordinates in 0–1000 range.\n'
        f"Keys to extract:\n{key_list}"
    )


def pil_to_base64_url(image: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode("utf-8")
    media_type = f"image/{format.lower()}"
    return f"data:{media_type};base64,{b64_data}"


def make_conversation(image: Image.Image, keys: list[str]) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _make_prompt(keys)},
                {"type": "image_url", "image_url": {"url": pil_to_base64_url(image)}},
            ],
        }
    ]


def visualize(image: Image.Image, items: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    W, H = image.size
    for item in items:
        bbox = item.get("box_2d")
        x0, y0, x1, y1 = bbox
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 12)), item.get("key", ""), fill="red")
    return image


def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="",
    )

    image = Image.open("/data/taxes.jpeg")
    completion = client.chat.completions.create(
        model="/data/models/kvp10k-qwen3vl-4b/",
        # messages=[
        #     {"role": "user", "content": "Hello!"},
        # ],
        messages=make_conversation(
            image, ["total_tax_box1", "payable_tax", "ssn", "reciavable_tax", "name"]
        ),
    )

    items = json.loads(completion.choices[0].message.content)
    visualize(image, items).save("output/out.jpeg")
