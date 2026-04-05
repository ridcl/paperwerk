from typing import Any


class DocumentStorage:
    def __init__(self, root_dir: str, summarizer: Any):
        self.root_dir = root_dir
        self.summarizer = summarizer

    def index(self):
        """Index all files in the root directory.

        Index is stored in `{root_dir}/index.json` and contains list of
        records of the following format:

        {
          "filename": <full path to the file>,
          "hash": <md5 hash of the file>,
          "summary": <summary of the whole document>,
          "page_summary": [
            <one-sentence summary of the 0th page>,
            <one-sentence summary of the 1st page>,
            ...
          ]
        }


        """
        ...

    def list_files(self) -> list[str]:
        """List all files in the root_dir and their summaries"""
        ...
