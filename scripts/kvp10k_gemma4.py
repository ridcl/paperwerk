"""KVP10K experiment: evaluate and fine-tune Gemma 4 E4B on KVP extraction using Unsloth.

Steps:
1. Load KVP10K dataset from local parquet
2. Evaluate base Gemma 4 E4B-it on test split
   - F1 of exact value match (per document, then macro-averaged)
   - Mean IoU of predicted vs. ground-truth bounding boxes (on exact-match hits)
3. Fine-tune on train split with LoRA via Unsloth + SFTTrainer
4. Re-evaluate fine-tuned model on test split

Installation (run once before this script):
    uv add --active unsloth timm accelerate
"""

import io
import json
import logging
import os

import unsloth  # must be first to apply all kernel patches before other imports
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "/data/kvp10k-with-images.parquet"

MODEL_ID = "unsloth/gemma-4-E4B-it"
OUTPUT_DIR = "/data/models/kvp10k-gemma4-e4b"
LORA_CKPT_DIR = "/data/cache/kvp10k_lora_ckpts_gemma4"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 4096
MAX_IMAGE_SIZE = 896
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_SAMPLES = 500

LORA_RANK = 16
LORA_ALPHA = 16
MAX_STEPS = 1_000
EVAL_EVERY_N_STEPS = 500

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATASET_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[~df["split"].isin(["train"])].reset_index(drop=True)
    logger.info("Train: %d rows, Test/Val: %d rows", len(train_df), len(test_df))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def _make_prompt(keys: list[str]) -> str:
    key_list = "\n".join(f"- {k}" for k in keys)
    return (
        "Extract the following key-value pairs from this document. "
        'Return a JSON list where each item has "key", "value", and "box_2d" fields. '
        'The "box_2d" is the bounding box of the value text as [y0, x0, y1, x1] '
        "with coordinates in 0–1000 range.\n"
        f"Keys to extract:\n{key_list}"
    )


def _gt_bbox_to_model(bbox: list[float]) -> list[int]:
    """Ground-truth [x0,y0,x1,y1] (0–1 normalised) → Gemma [y0,x0,y1,x1] (0–1000)."""
    x0, y0, x1, y1 = bbox
    return [round(y0 * 1000), round(x0 * 1000), round(y1 * 1000), round(x1 * 1000)]


def _model_bbox_to_normalized(bbox: list[int]) -> list[float]:
    """Gemma [y0,x0,y1,x1] (0–1000) → [x0,y0,x1,y1] (0–1 normalised)."""
    y0, x0, y1, x1 = bbox
    return [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]


def _make_target(kvps: list[dict]) -> str:
    targets = [
        {
            "key": kvp["key"],
            "value": kvp["value"],
            "box_2d": _gt_bbox_to_model(kvp["bounding_box"]),
        }
        for kvp in kvps
    ]
    return json.dumps(targets)


def make_conversation(row: pd.Series) -> dict:
    """Convert a dataset row to an Unsloth-compatible messages dict for training."""
    image = _load_image(row["image"])
    kvps = list(row["kvps"])
    keys = [kvp["key"] for kvp in kvps]
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _make_prompt(keys)},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _make_target(kvps)}],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x0, y0, x1, y1] normalised bounding boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compute_metrics(
    predictions: list[str],
    ground_truths: list[list[dict]],
) -> dict[str, float]:
    """Compute macro-averaged F1 (exact value match) and mean IoU.

    F1 is computed per document over key-level precision/recall, then averaged.
    IoU is computed only for keys where the value matches exactly.
    """
    all_f1: list[float] = []
    all_iou: list[float] = []
    n_parse_errors = 0

    for pred_str, gt_kvps in zip(predictions, ground_truths):
        try:
            pred_list = json.loads(pred_str)
            if not isinstance(pred_list, list):
                pred_list = []
        except (json.JSONDecodeError, ValueError):
            pred_list = []
            n_parse_errors += 1

        pred_by_key = {
            str(p.get("key", "")).strip().lower(): p
            for p in pred_list
            if isinstance(p, dict)
        }

        tp = 0
        for gt in gt_kvps:
            key_norm = str(gt["key"]).strip().lower()
            pred = pred_by_key.get(key_norm)
            if pred is None:
                continue
            if str(pred.get("value", "")).strip() == str(gt["value"]).strip():
                tp += 1
                try:
                    pred_norm = _model_bbox_to_normalized(pred["box_2d"])
                    all_iou.append(_iou(pred_norm, list(gt["bounding_box"])))
                except (KeyError, TypeError, ValueError):
                    pass

        n_pred = len(pred_list)
        n_gt = len(gt_kvps)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        denom = precision + recall
        all_f1.append(2 * precision * recall / denom if denom > 0 else 0.0)

    if n_parse_errors:
        logger.warning("JSON parse errors: %d / %d", n_parse_errors, len(predictions))

    return {
        "f1_exact_match": float(np.mean(all_f1)) if all_f1 else 0.0,
        "iou": float(np.mean(all_iou)) if all_iou else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize_predictions(
    image: Image.Image,
    output: str,
    gt_kvps: list[dict],
    output_path: str,
) -> None:
    """Draw GT boxes (blue) and predicted boxes (red) on image and save."""
    try:
        items = json.loads(output)
        if not isinstance(items, list):
            items = []
    except (json.JSONDecodeError, ValueError):
        items = []

    W, H = image.size
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    for kvp in gt_kvps:
        bb = kvp.get("bounding_box")
        if bb is None or len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        draw.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline="blue", width=2)

    for item in items:
        bbox = item.get("box_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        y0, x0, y1, x1 = bbox  # Gemma uses [y0, x0, y1, x1]
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 12)), item.get("key", ""), fill="red")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    vis.save(output_path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model,
    processor,
    df: pd.DataFrame,
    output_dir: str | None = None,
    max_samples: int = EVAL_MAX_SAMPLES,
) -> dict[str, float]:
    df = df.head(max_samples)
    predictions: list[str] = []
    ground_truths: list[list[dict]] = []

    model = FastVisionModel.for_inference(model)

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), desc="Evaluating", total=df.shape[0])
    ):
        image = _load_image(row["image"])
        kvps = list(row["kvps"])
        keys = [kvp["key"] for kvp in kvps]

        # During inference the image is passed to processor() directly;
        # the message only needs a placeholder {"type": "image"}.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _make_prompt(keys)},
                ],
            }
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                use_cache=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        predictions.append(output_text)
        ground_truths.append(kvps)

        if output_dir is not None:
            visualize_predictions(
                image,
                output_text,
                kvps,
                os.path.join(output_dir, f"row_{i:03}.jpeg"),
            )

    model = FastVisionModel.for_training(model)
    return compute_metrics(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def train(train_df: pd.DataFrame, model, processor) -> object:
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    logger.info("Preparing training dataset (%d rows)...", len(train_df))
    train_dataset = [
        make_conversation(row)
        for _, row in tqdm(train_df.iterrows(), desc="Converting", total=len(train_df))
    ]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            max_grad_norm=1.0,
            warmup_steps=50,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="steps",
            save_steps=EVAL_EVERY_N_STEPS,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=RANDOM_SEED,
            output_dir=LORA_CKPT_DIR,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LEN,
        ),
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_mem = round(gpu.total_memory / 1024**3, 3)
        logger.info("GPU = %s. Max memory = %.1f GB.", gpu.name, max_mem)
        logger.info("%.3f GB reserved before training.", start_mem)

    logger.info("Starting LoRA fine-tuning for %d steps", MAX_STEPS)
    trainer_stats = trainer.train()
    logger.info("Training complete: %.1f s", trainer_stats.metrics["train_runtime"])

    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        logger.info(
            "Peak reserved memory: %.3f GB (%.1f%% of max).",
            used_mem,
            used_mem / max_mem * 100,
        )

    logger.info("Saving LoRA adapter to %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    train_df, eval_df = load_splits()

    logger.info("Loading %s...", MODEL_ID)
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        # Non-quantized layers (layer norms, embeddings) load directly in bf16.
        # This prevents TRL from allocating a full float32→bf16 copy at SFTTrainer
        # init, which would OOM on a single 24 GB GPU.
        dtype=torch.bfloat16,
        # Spread the model across both GPUs sequentially (GPU 0 fills first).
        device_map="sequential",
        use_gradient_checkpointing="unsloth",
    )
    processor = get_chat_template(processor, "gemma-4")

    # # --- Evaluate base model ---
    # logger.info(
    #     "Evaluating base model on %d test samples...",
    #     min(EVAL_MAX_SAMPLES, len(eval_df)),
    # )
    # metrics_before = evaluate(model, processor, eval_df, output_dir="output/base_model")
    # logger.info(
    #     "Base model  — F1: %.4f  IoU: %.4f",
    #     metrics_before["f1_exact_match"],
    #     metrics_before["iou"],
    # )

    # --- Fine-tune ---
    model = FastVisionModel.for_training(model)
    model = train(train_df, model, processor)

    # --- Evaluate fine-tuned model ---
    logger.info("Evaluating fine-tuned model...")
    metrics_after = evaluate(model, processor, eval_df, output_dir="output/ft_model")
    logger.info(
        "Fine-tuned  — F1: %.4f  IoU: %.4f",
        metrics_after["f1_exact_match"],
        metrics_after["iou"],
    )

    print("\n=== Results ===")
    print(
        f"Fine-tuned:  F1={metrics_after['f1_exact_match']:.4f}"
        f"  IoU={metrics_after['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
