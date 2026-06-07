"""VQA with visual grounding on top of the vqa-20260529 model.

The served model (trained by ``src/training/vqa_20260529.py``) answers a list of
queries about a document page and, for each query it can support, returns the
value together with the bounding box of the supporting evidence and the page it
came from. This module is the single home for two things:

* the message-formatting utilities shared by serving and every VQA training
  experiment — ``make_prompt``, ``make_target``, the per-image content parts,
  and the Qwen bbox conversions. Centralising them keeps the served prompt and
  the parsed output schema byte-for-byte identical to what the model was trained
  on (a divergent prompt silently degrades grounding); and
* the ``VQA`` serving class plus a ``visualize`` helper.
"""

import json
import os
from collections.abc import Sequence

from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from paperwerk.llm import LLM
from paperwerk.utils import IMAGE_EXTENSIONS, pil_to_base64_url

# ---------------------------------------------------------------------------
# Message formatting  (shared by serving and every VQA training experiment)
# ---------------------------------------------------------------------------


def make_prompt(queries: Sequence[str]) -> str:
    """Build the user prompt listing the queries to answer.

    Must stay byte-for-byte identical to the prompt the model was trained with
    (see ``src/training/vqa_20260529.py``) — serving with a different prompt
    silently degrades answer and grounding quality.
    """
    query_list = "\n".join(f"- {q}" for q in queries)
    return f"Extract values:\n{query_list}"


def pil_image_part(image: Image.Image) -> dict:
    """Per-image content entry for HF processors / chat templates (training)."""
    return {"type": "image", "image": image}


def url_image_part(image: Image.Image) -> dict:
    """Per-image content entry for OpenAI-compatible endpoints (serving)."""
    return {"type": "image_url", "image_url": {"url": pil_to_base64_url(image)}}


def make_user_message(
    images: Sequence[Image.Image],
    queries: Sequence[str],
    image_part=url_image_part,
) -> dict:
    """Assemble the user turn: one entry per page image, then the query prompt.

    ``image_part`` selects the backend's image encoding — ``pil_image_part`` for
    HF/Unsloth training collators, ``url_image_part`` for OpenAI-compatible
    serving. The model is trained without a system message, so callers should
    not prepend one.
    """
    content = [image_part(image) for image in images]
    content.append({"type": "text", "text": make_prompt(queries)})
    return {"role": "user", "content": content}


def gt_bbox_to_qwen(bbox: Sequence[float]) -> list[int]:
    """Ground-truth [x0,y0,x1,y1] (0–1 normalised) → Qwen [x0,y0,x1,y1] (0–1000)."""
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000), round(y0 * 1000), round(x1 * 1000), round(y1 * 1000)]


def qwen_bbox_to_normalized(bbox: Sequence[int]) -> list[float]:
    """Qwen [x0,y0,x1,y1] (0–1000) → [x0,y0,x1,y1] (0–1 normalised)."""
    x0, y0, x1, y1 = bbox
    return [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]


def make_target(answers: Sequence[dict]) -> str:
    """Serialize dataset answers into the assistant JSON target (training).

    Serving parses exactly this shape back into ``Answer`` objects, so the two
    sides stay in lock-step.
    """
    targets = [
        {
            "query": ans["query"],
            "value": ans["value"],
            "box_2d": gt_bbox_to_qwen(list(ans["bounding_box"])),
            "index": int(ans["index"]),
        }
        for ans in answers
    ]
    return json.dumps(targets)


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


class Answer(BaseModel):
    """One grounded answer the model produced for a query about a page."""

    query: str
    value: str
    box_2d: list[int]  # [x0, y0, x1, y1], 0..1000
    index: int = 0  # 0-based page index within the images passed to the model
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


# Guided-decoding schema matching the trained JSON target exactly (no `meta`,
# which is a serving-side annotation rather than part of the model's output).
_ANSWERS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "value": {"type": "string"},
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "index": {"type": "integer"},
        },
        "required": ["query", "value", "box_2d", "index"],
    },
}


class VQA:
    """Answer queries about a document with visual grounding."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def __repr__(self):
        return "VQA()"

    def ask(self, image: Image.Image, queries: Sequence[str]) -> list[Answer]:
        """Answer ``queries`` about a single page image.

        Returns one ``Answer`` per supported query; queries the model cannot
        answer from the page are omitted.
        """
        messages = [make_user_message([image], queries)]
        completion = self.llm.invoke(
            messages, extra_body={"guided_json": _ANSWERS_SCHEMA}
        )
        data = json.loads(completion.choices[0].message.content)
        return [Answer(**item) for item in data]

    def __call__(self, path: str, queries: Sequence[str]) -> list[Answer]:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in IMAGE_EXTENSIONS:
            items = self.ask(Image.open(path), queries)
            for item in items:
                item.meta = {"filename": path, "page": 0}
            return items
        elif ext == ".pdf":
            # The model was trained on single-page examples, so answer each page
            # on its own and stamp the real page number into `meta` (the model's
            # own `index` is always 0 for a single-image prompt).
            images = convert_from_path(path, fmt="jpeg")
            items: list[Answer] = []
            for page, image in enumerate(images):
                page_items = self.ask(image, queries)
                for item in page_items:
                    item.meta = {"filename": path, "page": page}
                items.extend(page_items)
            return items
        else:
            raise ValueError(f"Unsupported file type: {ext}")


def visualize(image: Image.Image, items: Sequence[Answer]) -> Image.Image:
    """Draw each answer's bounding box (red) and query label on the page."""
    draw = ImageDraw.Draw(image)
    W, H = image.size
    for item in items:
        x0, y0, x1, y1 = item.box_2d
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 12)), item.query, fill="red")
    return image


def main():
    llm = LLM(
        base_url="http://localhost:8000/v1",
        api_key="",
        model="/data/models/vqa-20260529-qwen3vl-4b/",
    )
    vqa = VQA(llm)

    path = "/data/taxes.jpeg"
    queries = ["What was the receivable tax?", "name", "ssn"]
    items = vqa(path, queries)
    for item in items:
        print(item)

    image = Image.open(path)
    visualize(image, items).save("output/out.jpeg")
