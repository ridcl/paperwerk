import json
from openai import OpenAI

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


# ---------------------------
# Dummy tool implementations
# ---------------------------


def extract_values(document_path: str, keys: list[str]) -> dict:
    """Dummy extraction: returns placeholder values for each requested key."""
    return {key: f"<extracted:{key}>" for key in keys}


def list_files(directory: str = "/") -> list[str]:
    """Dummy file listing: returns a static set of example files."""
    return [
        "/docs/passport.pdf",
        "/docs/tax_return_2024.pdf",
        "/docs/employment_contract.pdf",
        "/docs/bank_statement_march.pdf",
    ]


# -------------
# Tool dispatch
# -------------


def dispatch_tool(name: str, arguments: dict):
    if name == "extract_values":
        return extract_values(**arguments)
    if name == "list_files":
        return list_files(**arguments)
    raise ValueError(f"Unknown tool: {name}")


# -------------
# Agentic loop
# -------------


class Agent:

    def __init__(
        self,
        model: str = "/data/models/kvp10k-qwen3vl-4b/",
        base_url: str = "http://localhost:8000/v1",
    ):
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="")
        self.messages = []

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
                    result = dispatch_tool(tool_call.function.name, arguments)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
            else:
                return choice.message.content


if __name__ == "__main__" and "__file__" in globals():
    agent = Agent()
    answer = agent.run(
        "List the available documents, then extract the 'name' and 'ssn' fields from the tax return."
    )
    print(answer)
