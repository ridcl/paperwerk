from pathlib import Path
from PIL import Image
from datasets import load_dataset
from paperwerk.agent import Agent
from paperwerk.vqa import Answer, visualize

# BBox-DocVQA / SciEGQA (https://github.com/yuwenhan07/SciEGQA,
# https://huggingface.co/datasets/Yuwh07/BBox_DocVQA_Train) with the page
# PNGs pre-extracted from images.tar, laid out as the dataset's own
# {category}/{doc_name}/{doc_name}_{page}.png convention.
BBOX_DOCVQA_ROOT = Path("/data/paperwerk/BBoxDocVQA")

# The vqa-20260529 model was trained on pages downscaled to this longest-side
# cap (see src/training/vqa_20260529.py), and the server enforces it. These
# dataset PNGs are high-DPI renders (~2500x3500), so we must match the cap.
# box_2d is normalized to 0..1000, so downscaling leaves coordinates valid.
MAX_IMAGE_SIZE = 2 * 896


def main():
    ds = load_dataset("Yuwh07/BBox_DocVQA_Train")
    df = ds["train"].to_pandas()
    # The VQA model is trained on single-page examples, so keep samples whose
    # evidence lives on a single page.
    sub = df[df.evidence_page.map(len) == 1]
    row = sub.iloc[5]

    page = int(row.evidence_page[0])  # 1-based, matches the PNG file name
    image_path = (
        BBOX_DOCVQA_ROOT
        / "images"
        / row.category
        / row.doc_name
        / f"{row.doc_name}_{page}.png"
    )
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    question = row.query

    agent = Agent.create_local()
    vqa = agent.ctx.vqa
    answers = vqa.ask(image, [question])
    out_img = visualize(image, answers)

    # rel_bbox is [xmin, ymin, xmax, ymax] already normalized to 0..1000, which
    # is exactly the model's box_2d scale — no axis reordering needed.
    def rel_bbox_to_answer(rel_bbox):
        return Answer(query="", value="", box_2d=[round(v) for v in rel_bbox])

    gt_answers = [rel_bbox_to_answer(b) for b in row.rel_bbox[0]]
    out_img = visualize(out_img, gt_answers, color="blue")
    out_img.save("output/out.jpeg")


# cd /data/paperwerk/BBoxDocVQA
# wget -c https://huggingface.co/datasets/Yuwh07/BBox_DocVQA_Train/resolve/main/images.tar
# tar -xf images.tar
