from pathlib import Path
from pdf2image import convert_from_path
from datasets import load_dataset
from paperwerk.agent import Agent
from paperwerk.vqa import visualize

# Copy of https://github.com/opendatalab/CiteVQA with
# PDFs pre-downloaded (669 of 711)
CITE_VQA_ROOT = Path("/data/paperwerk/CiteVQA")


def main():
    ds = load_dataset("opendatalab/CiteVQA")
    df = ds["validation"].to_pandas()
    row = df[df.language == "en"].iloc[3]
    pdf_path = CITE_VQA_ROOT / row.PDF_Source[0]
    page_idx = row.Evidence[0]["source_page_id"]
    image = convert_from_path(pdf_path)[page_idx]
    question = row.Question

    agent = Agent.create_local()
    vqa = agent.ctx.vqa
    answers = vqa.ask(image, [question])
    visualize(image, answers).save("output/out.jpeg")
