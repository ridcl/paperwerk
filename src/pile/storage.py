from copy import deepcopy
import glob
import hashlib
import json
import os
import traceback
from typing import Any

from tqdm import tqdm

from pile.extractor import IMAGE_EXTENSIONS
from pile.summarizer import Summarizer

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + (".pdf",)
INDEX_FILE = "index.json"


class DocumentStorage:
    def __init__(self, root_dir: str, summarizer: Summarizer):
        self.root_dir = root_dir
        self.summarizer = summarizer

    def _index_path(self) -> str:
        return os.path.join(self.root_dir, INDEX_FILE)

    def _file_hash(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_index(self) -> list[dict]:
        path = self._index_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []

    def _save_index(self, records: list[dict]):
        with open(self._index_path(), "w") as f:
            json.dump(records, f, indent=2)

    def index(self):
        """Index all files in the root directory.

        Index is stored in `{root_dir}/index.json` and contains list of
        records of the following format:

        {
          "filename": <full path to the file>,
          "hash": <md5 hash of the file>,
          "summary": <summary of the whole document>,
          "page_summaries": [
            <one-sentence summary of the 0th page>,
            <one-sentence summary of the 1st page>,
            ...
          ]
        }
        """
        existing = {r["filename"]: r for r in self._load_index()}
        records = []

        all_files = glob.glob(os.path.join(self.root_dir, "**", "*"), recursive=True)
        pbar = tqdm(all_files)
        for full_path in pbar:
            if not os.path.isfile(full_path):
                continue
            _, ext = os.path.splitext(full_path)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_hash = self._file_hash(full_path)

            dlen = 30
            filler = "..." if len(full_path) > dlen else ""
            pbar.set_description(filler + full_path[-dlen:])

            if full_path in existing and existing[full_path]["hash"] == file_hash:
                records.append(existing[full_path])
                continue

            try:
                summary_data = self.summarizer(full_path)
                record = {
                    "filename": full_path,
                    "hash": file_hash,
                    "summary": summary_data["summary"],
                }
                if "page_summaries" in summary_data:
                    record["page_summaries"] = summary_data["page_summaries"]
                records.append(record)
            except KeyboardInterrupt:
                print("Interrupted, stopping...")
                break
            except:
                print(f"Failed to process file {full_path}")
                print(traceback.format_exc())

        self._save_index(records)

    def list_files(self):
        """List all available files"""
        records = self._load_index()
        fields = ("filename", "summary")
        relevant = [{k: v for k, v in rec.items() if k in fields} for rec in records]
        return relevant

    def summaries(self) -> list[dict]:
        """Show all files in the index and their summaries"""
        records = self._load_index()
        fields = ("filename", "summary")
        relevant = [{k: v for k, v in rec.items() if k in fields} for rec in records]
        return relevant

    def details(self, filename: str) -> dict:
        """Show details of the specified file"""
        records = self._load_index()
        records = [rec for rec in records if rec["filename"] == filename]
        if len(records) != 1:
            raise ValueError(
                f"Expected exactly one record for {filename}, but got {len(records)}"
            )
        record = deepcopy(records[0])
        pages = [
            {"page": i, "summary": s} for i, s in enumerate(record["page_summaries"])
        ]
        return {
            "filename": filename,
            "summary": record["summary"],
            "pages": pages,
        }


def main():
    summarizer = Summarizer(
        base_url="http://localhost:8000/v1", model="/data/models/kvp10k-qwen3vl-4b/"
    )
    self = DocumentStorage("/data/Documents", summarizer)
