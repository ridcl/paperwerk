import json
from dataclasses import dataclass
import traceback
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_path
from pile.extractor import Extractor, visualize
from pile.storage import DocumentStorage
from pile.summarizer import Summarizer
from pile.vqa import VQA


VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "/data/models/kvp10k-qwen3vl-4b/"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_values",
            "description": (
                "Extract values for the given keys from a document (PDF or image) "
                "stored in the document storage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to the document file in storage.",
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to extract from the document.",
                    },
                },
                "required": ["document_path", "keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_extraction",
            "description": (
                "Visualize extracted values by drawing bounding boxes on the source document page. "
                "Returns the path to the saved visualization image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of extracted Grounded items as returned by extract_values.",
                        "items": {"type": "object"},
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the visualization image.",
                    },
                },
                "required": ["items", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_document",
            "description": (
                "Ask a free-form question about a document (PDF or image) "
                "and get a short, precise answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to the document file in storage.",
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the document.",
                    },
                },
                "required": ["document_path", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all indexed documents.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_summaries",
            "description": "List all indexed documents and their summaries.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_details",
            "description": "Show detailed information about a specific document, including per-page summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Full path to the document file.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
]


@dataclass
class Context:
    extractor: Extractor
    vqa: VQA
    storage: DocumentStorage


def ask_document(ctx: Context, document_path: str, question: str) -> str | list[str]:
    return ctx.vqa(document_path, question)


def extract_values(ctx: Context, document_path: str, keys: list[str]) -> list[dict]:
    items = ctx.extractor(document_path, keys)
    return [item.model_dump() for item in items]


def visualize_extraction(ctx: Context, items: list[dict], output_path: str) -> str:
    from pile.extractor import Grounded

    grounded = [Grounded(**item) for item in items]
    first = grounded[0]
    filename = first.meta["filename"]
    _, ext = filename.rsplit(".", 1)
    if ext.lower() == "pdf":
        page = first.meta["page"]
        images = convert_from_path(filename, fmt="jpeg")
        image = images[page]
    else:
        image = Image.open(filename)
    visualize(image, grounded).save(output_path)
    return output_path


def list_files(ctx: Context) -> list[str]:
    return ctx.storage.list_files()


def file_summaries(ctx: Context) -> list[str]:
    return ctx.storage.summaries()


def file_details(ctx: Context, filename: str) -> list[str]:
    return ctx.storage.details(filename)


# -------------
# Tool dispatch
# -------------


def dispatch_tool(ctx: Context, name: str, arguments: dict):
    try:
        if name == "ask_document":
            return ask_document(ctx, **arguments)
        if name == "extract_values":
            return extract_values(ctx, **arguments)
        if name == "visualize_extraction":
            return visualize_extraction(ctx, **arguments)
        if name == "list_files":
            return list_files(ctx, **arguments)
        if name == "file_details":
            return file_details(ctx, **arguments)
        if name == "file_details":
            return file_details(ctx, **arguments)
        raise ValueError(f"Unknown tool: {name}")
    except:
        # Propagate any exceptions to the agent LLM
        exc_str = traceback.format_exc()
        print(exc_str)
        return exc_str


# -------------
# Agentic loop
# -------------

SYSTEM_MESSAGE = """You are a personal document assistant.
You will be given questions about personal matters and should answer
them based on the available documents. Use tools to list documents
and extract information available in them. Start by listing all files
and always give reference to the document that you used. For example,
if you used `/foo/bar/baz.pdf`, reference it at the end of response as:

:link:/foo/bar/baz.pdf
"""


class Agent:

    def __init__(
        self,
        model: str = MODEL_NAME,
        base_url: str = VLLM_URL,
        root_dir: str = "/data/Documents",
    ):
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="")
        summarizer = Summarizer(base_url=base_url, model=model)
        self.ctx = Context(
            extractor=Extractor(model=model, base_url=base_url),
            vqa=VQA(model=model, base_url=base_url),
            storage=DocumentStorage(root_dir=root_dir, summarizer=summarizer),
        )
        self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    def __repr__(self):
        return "Agent()"

    def run(
        self,
        user_message: str,
    ) -> str:

        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            self.messages.append(choice.message.model_dump(exclude_unset=False))

            if choice.finish_reason == "tool_calls":
                for tool_call in choice.message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    result = dispatch_tool(self.ctx, tool_call.function.name, arguments)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
            else:
                return choice.message.content

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
    agent = Agent()
    answer = agent.run(
        "List the available documents, then extract the 'name' and 'ssn' fields from the tax return."
    )
    print(answer)
