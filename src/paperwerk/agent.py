import json
import re
from dataclasses import dataclass
import traceback
from PIL import Image
from pdf2image import convert_from_path
from paperwerk.async_utils import run_async
from paperwerk.extractor import Extractor, visualize
from paperwerk.llm import LLM
from paperwerk.storage import DocumentIndex, StorageBackend
from paperwerk.tools import REGISTRY, tool
from paperwerk.vqa import VQA

VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "/data/models/vqa-20260529-qwen3vl-4b/"


@dataclass
class Context:
    extractor: Extractor
    vqa: VQA
    index: DocumentIndex


@tool
def extract_values(ctx: Context, document_path: str, keys: list[str]) -> list[dict]:
    """Extract values for the given keys from a document (PDF or image)
    stored in the document storage.

    Args:
        document_path: Path to the document file in storage.
        keys: List of keys to extract from the document.
    """
    with ctx.index.as_local(document_path) as path:
        items = ctx.extractor(path, keys)
    for item in items:
        item.meta["filename"] = document_path
    return [item.model_dump() for item in items]


@tool
def visualize_extraction(ctx: Context, items: list[dict], output_path: str) -> str:
    """Visualize extracted values by drawing bounding boxes on the source document page.
    Returns the path to the saved visualization image.
    This tool can ONLY be used on the output from extract_values() function.

    Args:
        items: List of extracted Grounded items as returned by extract_values.
        output_path: Path to save the visualization image.
            Can be absolute or relative.
            The path should ALWAYS be used exactly as provided by user.
    """
    from paperwerk.extractor import Grounded

    grounded = [Grounded(**item) for item in items]
    first = grounded[0]
    filename = first.meta["filename"]
    _, ext = filename.rsplit(".", 1)
    with ctx.index.as_local(filename) as path:
        if ext.lower() == "pdf":
            page = first.meta["page"]
            images = convert_from_path(path, fmt="jpeg")
            image = images[page]
        else:
            image = Image.open(path)
        visualize(image, grounded).save(output_path)
    return output_path


@tool
def ask_document(ctx: Context, document_path: str, question: str) -> str | list[str]:
    """Ask a free-form question about a document (PDF or image) and get
    a short, precise answer.

    Args:
        document_path: Path to the document file in storage.
        question: Question to ask about the document.
    """
    with ctx.index.as_local(document_path) as path:
        return ctx.vqa(path, question)


@tool
def find_documents(ctx: Context, query: str, n: int = 5) -> list[dict]:
    """Find up to n documents in the index that are relevant to the query.

    Returns a list of {"path", "title"} entries. The "path" can be passed
    to other tools (e.g. extract_values, ask_document) to act on the
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
  `ask_document` or `extract_values` to read them. Do not attempt to
  answer without first checking available resources.
2. NEVER try to guess document name.
3. AWLAYS give references to the documents you used.
"""


class Agent:

    def __init__(self, llm: LLM, backend: StorageBackend):
        self.llm = llm
        self.ctx = Context(
            extractor=Extractor(llm),
            vqa=VQA(llm),
            index=DocumentIndex(backend, llm),
        )
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
    from paperwerk.storage import LocalStorageBackend

    llm = LLM(base_url=VLLM_URL, api_key="(none)", model=MODEL_NAME)
    backend = LocalStorageBackend("/data/paperwerk/storage")
    agent = Agent(llm, backend)
    agent.run_interactive()
    answer = agent.run(
        "Find the tax return document, then extract the 'name' and 'ssn' fields from it."
    )
    print(answer)
