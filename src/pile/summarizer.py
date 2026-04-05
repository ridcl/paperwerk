from openai import OpenAI
from PIL import Image


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
        return "Summarizer()"

    @staticmethod
    def _make_prompt(keys: list[str]) -> str: ...

    def summarize(self, image: Image.Image) -> str:
        """Summarize the content of the image for efficient retrieval later"""
        ...

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
        ...
