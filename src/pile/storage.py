import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass
import glob
import hashlib
import json
import os
import traceback
from typing import Protocol

from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
from multimethod import multimethod

from pile.extractor import IMAGE_EXTENSIONS
from pile.llm import LLM
from pile.summarizer import Summarizer
from pile.utils import pil_to_base64_url

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


# --------------------------------------------------------
# Document dataclasses
# --------------------------------------------------------


@dataclass
class Page:
    index: int
    summary: int


@dataclass
class Document:
    path: str
    title: str
    summary: str
    pages: list[Page]


# --------------------------------------------------------
# Storage Backend
# --------------------------------------------------------


class StorageBackend(Protocol):
    """Storage backend protocol.

    Storage backend is responsible for managing document content.
    Implementations can store documents locally or remotely,
    retrieve eagerly or lazily, etc.
    """

    def add(self, filename: str, content: bytes) -> str:
        """Add a new document to the storage.

        Args:
          filename: Original filename for better readability by humans.
          content: Content of the document.

        Returns:
          Unique path of the document in the storage. The path includes
          file extension that helps to identify document type.
        """
        ...

    def get(self, path: str) -> bytes:
        """Get the document from the storage.

        Args:
          path: Unique path of the document in the storage.

        Returns:
          Document content as bytes.
        """
        ...

    def remove(self, path: str):
        """Remove the document from the storage.

        Args:
          path: Unique path of the document in the storage.
        """
        ...

    def write_meta(self, name: str, content: bytes):
        """Write a fixed-name metadata blob (e.g. the document index).

        Metadata blobs share the storage with documents but live in a
        reserved namespace, so a real document upload can never collide
        with them.
        """
        ...

    def read_meta(self, name: str) -> bytes:
        """Read a metadata blob previously written via ``write_meta``.

        Raises:
          FileNotFoundError: if no blob is stored under ``name``.
        """
        ...

    def delete_meta(self, name: str):
        """Delete a metadata blob."""
        ...


META_DIR = "_meta"


class LocalStorageBackend:
    """Storage backend that stores documents on the local filesystem.

    Documents are named as ``{hash_prefix}_{original_filename}`` where
    ``hash_prefix`` is the first 6 hex characters of the MD5 of the content.
    This keeps paths unique while still hinting at what's inside.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _full_path(self, path: str) -> str:
        return os.path.join(self.base_path, path)

    def add(self, filename: str, content: bytes) -> str:
        digest = hashlib.md5(content).hexdigest()[:6]
        path = f"{digest}_{os.path.basename(filename)}"
        full_path = self._full_path(path)
        if not os.path.exists(full_path):
            with open(full_path, "wb") as f:
                f.write(content)
        return path

    def get(self, path: str) -> bytes:
        with open(self._full_path(path), "rb") as f:
            return f.read()

    def remove(self, path: str):
        os.remove(self._full_path(path))

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.base_path, META_DIR, name)

    def write_meta(self, name: str, content: bytes):
        full_path = self._meta_path(name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)

    def read_meta(self, name: str) -> bytes:
        with open(self._meta_path(name), "rb") as f:
            return f.read()

    def delete_meta(self, name: str):
        os.remove(self._meta_path(name))


# --------------------------------------------------------
# Document Index
# --------------------------------------------------------

SUMMARY_PROMPT = """Extract the title of the document and summarize its content.
If title is not visible, make it up yourself (3-8 words).
Summary should be very short, preferably 1 sentence, and focus on the main subjects.

If the document refers to a specific person and/or legal entity,
mention it in the title. Example:

    "title": "Employment contract between John Doe and SuperCorp"

If it is about more people and/or entities, mention it in the summary. Example:

    "title": "Birth certificate of John Doe",
    "summary": "Birth certificate of John Doe. Born: January 1, 2022. Parents: Adam Doe and Mary Doe"

Format:
{
  "title": <title>,
  "summary": <summary>
}
"""


INDEX_NAME = "index.json"


class DocumentIndex:

    def __init__(self, backend: StorageBackend, llm: LLM):
        self.backend = backend
        self.llm = llm
        self.documents: dict[str, Document] = {}
        self._load()

    def _load(self):
        try:
            content = self.backend.read_meta(INDEX_NAME)
        except FileNotFoundError:
            return
        data = json.loads(content.decode("utf-8"))
        self.documents = {
            path: Document(
                path=d["path"],
                title=d["title"],
                summary=d["summary"],
                pages=[Page(**p) for p in d["pages"]],
            )
            for path, d in data.items()
        }

    def _save(self):
        data = {path: asdict(doc) for path, doc in self.documents.items()}
        content = json.dumps(data, indent=2).encode("utf-8")
        self.backend.write_meta(INDEX_NAME, content)

    @multimethod
    async def _summarize(self, image: Image.Image) -> dict:
        """Summarize the content of the image for efficient retrieval later."""
        completion = await self.llm.ainvoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SUMMARY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_base64_url(image)},
                        },
                    ],
                }
            ],
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        )
        return json.loads(completion.choices[0].message.content)

    @multimethod
    async def _summarize(self, local_path: str) -> dict:
        _, ext = os.path.splitext(local_path)
        if ext.lower() in IMAGE_EXTENSIONS:
            image = Image.open(local_path)
            return await self._summarize(image)
        elif ext.lower() == ".pdf":
            images = await asyncio.to_thread(convert_from_path, local_path, fmt="jpeg")
            page_summaries = await asyncio.gather(
                *(self._summarize(img) for img in images)
            )
            combined_prompt = (
                "Below are one-sentence summaries of each page of a multi-page document. "
                "Write a single cohesive summary of the whole document.\n\n"
                + "\n".join(
                    f"Page {i}: {s['summary']}" for i, s in enumerate(page_summaries)
                )
            )
            completion = await self.llm.ainvoke(
                [{"role": "user", "content": combined_prompt}],
            )
            return {
                "title": page_summaries[0]["title"],
                "summary": completion.choices[0].message.content,
                "pages": page_summaries,
            }

    async def add(self, local_path: str) -> Document:
        filename = os.path.basename(local_path)
        with open(local_path, "rb") as fp:
            content = fp.read()
        path = self.backend.add(filename, content)
        details = await self._summarize(local_path)
        page_details = details.get("pages", [])
        doc = Document(
            path=path,
            title=details["title"],
            summary=details["summary"],
            pages=[Page(i, ps["summary"]) for i, ps in enumerate(page_details)],
        )
        self.documents[path] = doc
        self._save()
        return path

    def get(self, path: str) -> Document:
        return self.documents[path]

    def list(self):
        return list(self.documents.keys())


async def main():
    llm = LLM(
        base_url="http://localhost:8000/v1",
        api_key="",
        model="/data/models/kvp10k-qwen3vl-4b-retrained/",
    )
    self = DocumentIndex(LocalStorageBackend("/data/pile/storage"), llm)
    await asyncio.gather(
        self.add("/data/Documents/PP/permit_andrei.jpg"),
        self.add(
            "/data/Documents/DataSnipper/Andrei Zhabinski + DataSnipper document.pdf"
        ),
    )
