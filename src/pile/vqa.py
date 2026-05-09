import os

from pdf2image import convert_from_path
from PIL import Image

from pile.llm import LLM
from pile.utils import IMAGE_EXTENSIONS, pil_to_base64_url


SYSTEM_MESSAGE = """You are VQA bot.
Give a short a precise answer to the question provided by user.
"""


class VQA:

    def __init__(self, llm: LLM):
        self.llm = llm

    def __repr__(self):
        return "VQA()"

    @staticmethod
    def _make_conversation(image: Image.Image, question: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_base64_url(image)},
                    },
                ],
            },
        ]

    def ask(self, image: Image.Image, question: str) -> str:
        completion = self.llm.invoke(self._make_conversation(image, question))
        return completion.choices[0].message.content

    def __call__(self, path: str, question: str) -> str | list[str]:
        _, ext = os.path.splitext(path)
        if ext in IMAGE_EXTENSIONS:
            return self.ask(Image.open(path), question)
        elif ext == ".pdf":
            images = convert_from_path(path, fmt="jpeg")
            return [self.ask(image, question) for image in images]
        else:
            raise ValueError(f"Unsupported file type: {ext}")


def main():
    llm = LLM(
        base_url="http://localhost:8000/v1",
        api_key="",
        model="/data/models/kvp10k-qwen3vl-4b/",
    )
    vqa = VQA(llm)

    path = "/data/taxes.jpeg"
    question = "What was the receivable tax"
    answer = vqa(path, question)
    print(answer)
