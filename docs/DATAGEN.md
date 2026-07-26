## Data generation utilities


> ℹ️ Data generation requires `poppler` and `playwright` system packages, as well as `playwright` dependencies. Look at Dockerfile.train for full config or just follow the error messages.

Paperwerk provides several helpers for synthetic data generation using LLMs. Most functions accept an instance of `LLM` class. You can point it to a local vLLM, e.g.:

```python
from paperwerk.llm import LLM

llm = LLM(
    base_url="http://localhost:8000/v1/",
    api_key="(none)",
    model="google/gemma-4-E4B-it",
)
```

or a API-based model such as Claude:

```python
import os
from paperwerk.llm import LLM

api_key = os.environ.get("ANTHROPIC_API_KEY")
llm = LLM(
    base_url="https://api.anthropic.com/v1/",
    api_key=api_key,
    model="claude-sonnet-4-6",
)
```

Quick usage example:

```python
from paperwerk.datagen import (
    make_template, random_values, render, augment, annotate
)

input_path = "path/to/example-doc.pdf"
output_path = "path/to/generated-doc.pdf"


# 1. Create a Junja2 template from an image or PDF file.
# The template follows the structure of the example document, but replaces all
# concrete values with placeholders that you can populate with your own data.
# The list of placeholders is returned as field_names: list[str].
template, field_names = await make_template(llm, input_path)

# 2. Generate values.
# random_values() provides best effort generation, but doesn't guarantee internal
# consistency or fidelity. You may want to create a different value generation process
# that is better aligned with a specific task.
values = await random_values(llm, field_names)

# 3. Render the template with the given values to a PDF file.
# Each field: Field contains:
#  * name
#  * value
#  * page (zero-indexed)
#  * bbox ([x0, y0, x1, y1], scaled to 0..1 w.r.t. page size)
pdf_bytes, fields = await render(template, values)

# 4. Optionally, augment the file.
# Note that fields may change their bbox after the augmentation.
pdf_bytes, fields = augment(pdf_bytes, fields, profile="phone_photo")

# 5. Optionally, annotate the document with fields
pdf_bytes = annotate(pdf_bytes, fields)

# 6. Save the result
with open(output_path, "wb") as fp:
    fp.write(pdf_bytes)
```
