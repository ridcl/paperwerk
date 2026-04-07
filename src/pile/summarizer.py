import os

from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

from pile.extractor import IMAGE_EXTENSIONS, pil_to_base64_url


class Summarizer:

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
        return "Summarizer()"

    @staticmethod
    def _make_prompt() -> str:
        return (
            "Give a short description of this document page. "
            "Focus on document type/purpose and key people. "
            "A few examples:\n"
            " * Passport of John Doe\n"
            " * List of transactions between Jan 1, 2021 and Aug 1, 2021\n"
            " * Sections of the contract: 4) Wages and holiday allowance; 5) Vacation days; ..."
        )

    # TODO: summarize document from page summaries

    @staticmethod
    def _make_short_prompt() -> str:
        return "Describe this document page in one sentence, focusing on its type and main subject."

    def summarize(self, image: Image.Image) -> str:
        """Summarize the content of the image for efficient retrieval later."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._make_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_base64_url(image)},
                        },
                    ],
                }
            ],
        )
        return completion.choices[0].message.content

    def _summarize_short(self, image: Image.Image) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._make_short_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_base64_url(image)},
                        },
                    ],
                }
            ],
        )
        return completion.choices[0].message.content

    def __call__(self, path: str) -> dict:
        """Summarize the document given by path (image or PDF)
        for efficient retrieval later. Returns the following dict:

        {
            "summary": <summary of the whole document>,
            "page_summary": [
                <one-sentence summary of the 0th page>,
                <one-sentence summary of the 1st page>,
                ...
          ]
        }

        Where "page_summary" is a list of page summaries (only for PDFs).
        """
        _, ext = os.path.splitext(path)
        if ext.lower() in IMAGE_EXTENSIONS:
            image = Image.open(path)
            return {"summary": self.summarize(image)}
        elif ext.lower() == ".pdf":
            images = convert_from_path(path, fmt="jpeg")
            page_summaries = [self._summarize_short(img) for img in images]
            combined_prompt = (
                "Below are one-sentence summaries of each page of a multi-page document. "
                "Write a single cohesive summary of the whole document.\n\n"
                + "\n".join(f"Page {i}: {s}" for i, s in enumerate(page_summaries))
            )
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": combined_prompt}],
            )
            return {
                "summary": completion.choices[0].message.content,
                "page_summary": page_summaries,
            }
        else:
            raise ValueError(f"Unsupported file type: {ext}")
