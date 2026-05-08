import base64
from collections.abc import Sequence
from io import BytesIO
from PIL import Image

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def pil_to_base64_url(image: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=format)
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode("utf-8")
    media_type = f"image/{format.lower()}"
    return f"data:{media_type};base64,{b64_data}"
