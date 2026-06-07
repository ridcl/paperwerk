from pathlib import Path
from pdf2image import convert_from_path
from datasets import load_dataset
from paperwerk.agent import Agent
from paperwerk.vqa import Answer, visualize

# Copy of https://github.com/opendatalab/CiteVQA with
# PDFs pre-downloaded (669 of 711)
CITE_VQA_ROOT = Path("/data/paperwerk/CiteVQA")


def main():
    ds = load_dataset("opendatalab/CiteVQA")
    df = ds["validation"].to_pandas()
    sub = df[
        (df.language == "en")
        & (df.Question_Type.isin(["Multimodal Parsing", "Factual Retrieval"]))
    ]
    # row = sub.iloc[2]
    row = sub.iloc[8]

    pdf_path = CITE_VQA_ROOT / row.PDF_Source[0]
    page_idx = row.Evidence[0]["source_page_id"] - 1
    image = convert_from_path(pdf_path)[page_idx]
    question = row.Question

    agent = Agent.create_local()
    vqa = agent.ctx.vqa
    answers = vqa.ask(image, [question])
    out_img = visualize(image, answers)

    def evidence_to_answer(ev):
        ev_bbox = ev["bbox"]
        bbox = ev_bbox[1], ev_bbox[0], ev_bbox[3], ev_bbox[2]
        return Answer(query="", value="", box_2d=bbox)

    gt_answers = [evidence_to_answer(ev) for ev in row.Evidence]
    out_img = visualize(out_img, gt_answers, color="blue")
    out_img.save("output/out.jpeg")
