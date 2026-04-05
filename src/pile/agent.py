import json
import glob
from dataclasses import dataclass
import traceback
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_path
from pile.extractor import Extractor, visualize


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
            "name": "list_files",
            "description": "List files available in document storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Optional subdirectory to list. Defaults to root.",
                        "default": "/",
                    }
                },
                "required": [],
            },
        },
    },
]


@dataclass
class Context:
    extractor: Extractor


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


def list_files(ctx: Context, directory: str = "/") -> list[str]:
    """Dummy file listing: returns a static set of example files."""
    extensions = ["pdf", "jpeg", "jpg", "png"]
    filenames = []
    for ext in extensions:
        filenames += glob.glob(f"/data/Documents/**/*.{ext}")
    return filenames


# -------------
# Tool dispatch
# -------------


def dispatch_tool(ctx: Context, name: str, arguments: dict):
    if name == "extract_values":
        return extract_values(ctx, **arguments)
    if name == "visualize_extraction":
        return visualize_extraction(ctx, **arguments)
    if name == "list_files":
        return list_files(ctx, **arguments)
    raise ValueError(f"Unknown tool: {name}")


# -------------
# Agentic loop
# -------------

SYSTEM_MESSAGE = """You are a personal document assistant.
You will be given questions about personal matters and should answer
them based on the available documents. Use tools to list documents
and extract information available in them.
"""


class Agent:

    def __init__(
        self,
        model: str = MODEL_NAME,
        base_url: str = VLLM_URL,
    ):
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="")
        self.ctx = Context(extractor=Extractor(model=model, base_url=base_url))
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
                out = self.run(":response: " + user_message)
                print(out)
            except Exception:
                print(traceback.format_exc())


if __name__ == "__main__" and "__file__" in globals():
    agent = Agent()
    answer = agent.run(
        "List the available documents, then extract the 'name' and 'ssn' fields from the tax return."
    )
    print(answer)
