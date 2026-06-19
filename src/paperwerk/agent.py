import json
import re
import sys
import os
import traceback
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from paperwerk.async_utils import run_async
from paperwerk.llm import LLM
from paperwerk.storage import DocumentIndex, StorageBackend
from paperwerk.tools import REGISTRY, tool
from paperwerk.vqa import VQA, Answer, visualize

VLLM_URL = "http://localhost:8000/v1"
VQA_MODEL_NAME = "ridcl/paperwerk-vqa"


@dataclass
class Context:
    vqa: VQA
    index: DocumentIndex


@tool
def ask_document(ctx: Context, document_path: str, queries: list[str]) -> list[dict]:
    """Answer one or more queries about a document (PDF or image) with grounding.

    Works both for free-form questions ("What was the receivable tax?") and for
    extracting specific field values (e.g. "name", "ssn"). Returns a list of
    answers; each has the `query` it answers, the extracted `value`, a `box_2d`
    bounding box ([x0, y0, x1, y1], 0–1000) locating the supporting evidence, and
    the page it was found on. Queries the document does not answer are omitted.

    Args:
        document_path: Path to the document file in storage.
        queries: Questions to answer or field names to extract from the document.
    """
    with ctx.index.as_local(document_path) as path:
        items = ctx.vqa(path, queries)
    for item in items:
        item.meta["filename"] = document_path
    return [item.model_dump() for item in items]


@tool
def visualize_answers(ctx: Context, items: list[dict], output_path: str) -> str:
    """Visualize answers by drawing their bounding boxes on the source document page.
    Returns the path to the saved visualization image.
    This tool can ONLY be used on the output from ask_document() function.

    Args:
        items: List of answer items as returned by ask_document.
        output_path: Path to save the visualization image.
            Can be absolute or relative.
            The path should ALWAYS be used exactly as provided by user.
    """
    grounded = [Answer(**item) for item in items]
    filename = grounded[0].meta["filename"]
    _, ext = filename.rsplit(".", 1)
    with ctx.index.as_local(filename) as path:
        if ext.lower() == "pdf":
            page = grounded[0].meta["page"]
            images = convert_from_path(path, fmt="jpeg")
            image = images[page]
        else:
            image = Image.open(path)
        # Answers may span several pages; only draw the ones on this page.
        page_items = [
            a
            for a in grounded
            if a.meta.get("page", 0) == grounded[0].meta.get("page", 0)
        ]
        visualize(image, page_items).save(output_path)
    return output_path


@tool
def find_documents(ctx: Context, query: str, n: int = 5) -> list[dict]:
    """Find up to n documents in the index that are relevant to the query.

    Returns a list of {"path", "title"} entries. The "path" can be passed
    to other tools (e.g. ask_document) to act on the
    document. Use a descriptive query, e.g. "tax return for 2023" or
    "John Doe's passport".

    Args:
        query: Free-form description of what to look for.
        n: Maximum number of documents to return.
    """
    docs = run_async([ctx.index.find(query, n)])[0]
    return [{"path": d.path, "title": d.title} for d in docs]


# -------------
# Tool dispatch
# -------------


def dispatch_tool(ctx: Context, name: str, arguments: dict):
    try:
        return REGISTRY.dispatch(ctx, name, arguments)
    except:
        # Propagate any exceptions to the agent LLM
        exc_str = traceback.format_exc()
        print(exc_str)
        return exc_str


# -------------
# Agentic loop
# -------------


def _repair_json(s: str) -> str:
    """Fix model's common mistake: missing closing quote before } or ]."""
    # Count unmatched quotes to detect unclosed strings
    depth = 0
    in_string = False
    i = 0
    result = []
    while i < len(s):
        c = s[i]
        if c == "\\" and in_string:
            result.append(s[i : i + 2])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        elif c in "{[" and not in_string:
            depth += 1
        elif c in "}]" and not in_string:
            depth -= 1
        elif c in "}]" and in_string:
            # Closing bracket inside an unclosed string — insert missing quote
            result.append('"')
            in_string = False
            depth -= 1
        result.append(c)
        i += 1
    return "".join(result)


def _parse_raw_tool_calls(content: str) -> list[dict]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    for candidate in (raw, _repair_json(raw)):
        try:
            data = json.loads(candidate)
            return [{"name": data["name"], "arguments": json.dumps(data["arguments"])}]
        except (json.JSONDecodeError, KeyError):
            continue
    return []


SYSTEM_PROMPT = """You are a personal document assistant.
You will be given questions about personal matters and should answer
them based on the available documents. Some rules:

1. When any information is unavailable or inaccessible, immediately use
  `find_documents(query, n)` to locate relevant documents, then
  `ask_document` to read values out of them. Do not attempt to
  answer without first checking available resources.
2. NEVER try to guess document name.
3. AWLAYS give references to the documents you used.
"""


class Agent:

    def __init__(
        self, llm: LLM, backend: StorageBackend, vqa_llm: Optional[LLM] = None
    ):
        self.llm = llm
        self.ctx = Context(
            vqa=VQA(vqa_llm or llm),
            index=DocumentIndex(backend, llm),
        )
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def create_local() -> "Agent":
        """Create fully local agent, using vLLM and LocalStorageBackend"""
        from paperwerk.storage import LocalStorageBackend

        llm = LLM(base_url=VLLM_URL, api_key="(none)", model=VQA_MODEL_NAME)
        backend = LocalStorageBackend("/data/paperwerk/storage")
        return Agent(llm, backend)

    def create_hybrid() -> "Agent":
        """Create an Agent with Claude for the main LLM and vLLM for VQA.
        Use the local backend
        """
        from paperwerk.storage import LocalStorageBackend

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")
        llm = LLM(
            base_url="https://api.anthropic.com/v1/",
            api_key=api_key,
            model="claude-sonnet-4-6",
        )
        vqa_llm = LLM(base_url=VLLM_URL, api_key="(none)", model=VQA_MODEL_NAME)
        backend = LocalStorageBackend("/data/paperwerk/storage")
        return Agent(llm, backend, vqa_llm=vqa_llm)

    def __repr__(self):
        return "Agent()"

    def run(
        self,
        user_message: str,
    ) -> str:

        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.llm.invoke(
                self.messages,
                tools=REGISTRY.to_openai(),
                tool_choice="auto",
            )

            choice = response.choices[0]
            self.messages.append(choice.message.model_dump(exclude_unset=False))

            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
            elif choice.message.content and "<tool_call>" in choice.message.content:
                # Qwen3-VL sometimes generates invalid tool calls, which breaks
                # the parsing mechanism. This is a hack to fix it.
                tool_calls = _parse_raw_tool_calls(choice.message.content)
                if not tool_calls:
                    return choice.message.content
            else:
                return choice.message.content

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    name = tool_call["name"]
                    arguments = json.loads(tool_call["arguments"])
                    tool_call_id = f"fallback-{name}"
                else:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                result = dispatch_tool(self.ctx, name, arguments)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result),
                    }
                )

    def run_interactive(self):
        user_message = None
        while user_message != "/exit":
            user_message = input(":prompt: ")
            try:
                out = self.run(user_message)
                print(":response: " + out)
            except Exception:
                print(traceback.format_exc())


if __name__ == "__main__" and "__file__" in globals():
    agent = Agent.create_hybrid()
    agent.run_interactive()
    answer = agent.run(
        "Find the tax return document, then read the 'name' and 'ssn' fields from it."
    )
    print(answer)
