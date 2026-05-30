"""LoRA fine-tuning of Qwen3.5-4B on the CUAD+KVP10k VQA dataset — Unsloth port.

Sibling of ``vqa_20260529.py`` (the Qwen3-VL-4B version). Same task, data
formatting, prompt, target schema, evaluation metrics, and resume logic — the
only differences are the model and the few things its architecture forces:

    Qwen3-VL version (vqa_20260529.py)   Qwen3.5 version (this file)
    ----------------------------------   --------------------------------------
    FastVisionModel.from_pretrained      FastModel.from_pretrained
    MODEL_ID = Qwen3-VL-4B-Instruct      MODEL_ID = Qwen/Qwen3.5-4B
    LoRA on q/k + gate/up/down           LoRA on q/k/v/o + gate/up/down
    plain instruct (no thinking)         thinking model -> disabled for eval gen

About Qwen3.5-4B (released Feb 2026):
  - Natively multimodal (image-text-to-text), so the document-VQA-with-images
    task maps directly; loaded through Unsloth's generic ``FastModel`` (which
    returns a multimodal processor), not ``FastVisionModel``.
  - Hybrid attention: a 3:1 stack of Gated DeltaNet (linear attention) layers to
    Gated Attention (full attention) layers, plus a sparse MoE FFN. The Gated
    DeltaNet layers need Triton kernels from ``flash-linear-attention`` (a project
    dependency) — transformers imports them automatically when the model loads.
  - It is a *thinking* model by default (emits ``<think>...</think>`` before the
    answer). Our targets are pure JSON, so eval generation passes
    ``enable_thinking=False`` to keep the model from prefixing reasoning. Training
    on complete assistant turns (the JSON) is naturally non-thinking; see the note
    in ``train`` about validating the collator-rendered sample.

The merged 16-bit checkpoint it writes loads directly in vLLM / transformers
without Unsloth or PEFT.

Dataset schema (produced by datagen.builders.vqa_20260524_cuad_kvp10k):
    images      list[binary]   — page image bytes, one entry per page
    queries     list[string]   — answerable + unanswerable queries (shuffled)
    answers     list[struct{               — one entry per answerable evidence
                   query:        string,    —   the query this answers
                   value:        string,    —   literal or derived answer
                   bounding_box: list[f64], —   [x0,y0,x1,y1] normalised 0–1
                   index:        int32,     —   which `images` entry holds it
                }]
    source      string
    variant     string
    page_start  int32
    page_end    int32
    split       string

Both data builders currently emit one image per datapoint, but every path here
handles an arbitrary number of pages (each answer carries the `index` of its
page).
"""

import os

# Qwen3.5's Gated DeltaNet uses Triton kernels (per Unsloth's docs). But when
# `tilelang` is installed (vLLM pulls it), flash-linear-attention's backend
# dispatcher prefers its tilelang backend, whose nvcc-based JIT fails against the
# CUDA-13 toolkit headers ("CUDA compiler and CUDA toolkit headers are
# incompatible") on the DeltaNet backward. Disabling it forces fla's pure-Triton
# path (compiled via Triton's own ptxas, no nvcc) — the path Unsloth expects, and
# the one verified to train end-to-end on this transformers-5.5 / torch-cu13 /
# vLLM-0.22 stack. Set before unsloth/transformers/fla import so the backend
# registry sees it; setdefault lets the environment override it.
os.environ.setdefault("FLA_TILELANG", "0")

# Unsloth patches transformers/trl on import, so it MUST come first — importing
# it after them silently disables the optimizations (and Unsloth warns loudly).
import unsloth  # noqa: E402, F401  (side-effecting; keep above transformers/trl)
from unsloth import FastModel  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from tqdm import tqdm  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "/data/paperwerk/vqa_20260524.parquet"

MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "/data/models/vqa-20260529-qwen3_5-4b"
LORA_CKPT_DIR = "/data/cache/vqa_20260529_qwen3_5_lora_ckpts"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1  # JAX trained on a per-step batch of 1
MAX_SEQ_LEN = 4096
MAX_IMAGE_SIZE = 896
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_SAMPLES = 500

# 16-bit LoRA (not 4-bit QLoRA): Unsloth recommends against 4-bit for Qwen3.5,
# and 16-bit keeps the merged checkpoint a clean, vLLM-servable model.
LOAD_IN_4BIT = False

# Qwen3.5 is a *thinking* model; our targets are JSON, so generation disables it.
ENABLE_THINKING = False

LORA_RANK = 16
LORA_ALPHA = 2 * LORA_RANK
# Unsloth's recommended Qwen3.5 target set: the Gated Attention projections
# (q/k/v/o) + the FFN/expert projections (gate/up/down). These names match the
# full-attention + MoE-FFN layers; the Gated DeltaNet (linear-attention) layers
# use different module names and stay frozen, as does the vision encoder.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MAX_STEPS = 10_000
EVAL_EVERY_N_STEPS = 500
WARMUP_STEPS = 50
PEAK_LR = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

RANDOM_SEED = 42
TEST_FRACTION = 0.05

# Qwen ChatML role markers — used by the collator to mask everything except the
# assistant turn, so loss is computed only on the completion (as the JAX run did
# via encode_messages(loss_roles={"assistant"})).
_INSTRUCTION_PART = "<|im_start|>user\n"
_RESPONSE_PART = "<|im_start|>assistant\n"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    # `images` is list<binary> with 32-bit offsets. Across the whole file its
    # decoded bytes exceed the 2 GiB offset limit, so pyarrow must split the
    # column into chunks — but it can't emit a *chunked* nested array and raises
    # "Nested data conversions not implemented for chunked array outputs", even
    # inside read_table. Reading in bounded batches keeps each batch's binary
    # under the limit (one contiguous array per batch), so the nested conversion
    # stays on the supported path; concatenate the per-batch frames.
    pf = pq.ParquetFile(DATASET_PATH)
    frames = [batch.to_pandas() for batch in pf.iter_batches(batch_size=512)]
    df = pd.concat(frames, ignore_index=True)
    # Random split by index (ignores the parquet's own `split` column) so the
    # held-out set is drawn uniformly across sources and variants. Seeded for
    # reproducibility.
    test_df = df.sample(frac=TEST_FRACTION, random_state=RANDOM_SEED)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Data formatting  (identical to the Qwen3-VL version)
# ---------------------------------------------------------------------------


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def _load_images(images_bytes: Any) -> list[Image.Image]:
    """Decode and downscale every page image of a datapoint, in order."""
    return [_load_image(b) for b in images_bytes]


def _make_prompt(queries: list[str], n_images: int) -> str:
    query_list = "\n".join(f"- {q}" for q in queries)
    return f"Extract values:\n{query_list}"


def _gt_bbox_to_qwen(bbox: list[float]) -> list[int]:
    """Ground-truth [x0,y0,x1,y1] (0–1 normalised) → Qwen [x0,y0,x1,y1] (0–1000)."""
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000), round(y0 * 1000), round(x1 * 1000), round(y1 * 1000)]


def _qwen_bbox_to_normalized(bbox: list[int]) -> list[float]:
    """Qwen [x0,y0,x1,y1] (0–1000) → [x0,y0,x1,y1] (0–1 normalised)."""
    x0, y0, x1, y1 = bbox
    return [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]


def _make_target(answers: list[dict]) -> str:
    targets = [
        {
            "query": ans["query"],
            "value": ans["value"],
            "box_2d": _gt_bbox_to_qwen(list(ans["bounding_box"])),
            "index": int(ans["index"]),
        }
        for ans in answers
    ]
    return json.dumps(targets)


def make_conversation(row: pd.Series) -> list[dict]:
    """Build the 2-turn chat (user prompt+images, assistant JSON target).

    The message structure — content lists of ``{"type": "image", "image": PIL}``
    and ``{"type": "text", "text": ...}`` — is exactly what
    ``UnslothVisionDataCollator`` consumes (it runs the processor's chat template
    and image preprocessing internally).
    """
    images = _load_images(row["images"])
    queries = list(row["queries"])
    answers = list(row["answers"])
    user_content: list[dict] = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": _make_prompt(queries, len(images))})
    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _make_target(answers)}],
        },
    ]


# ---------------------------------------------------------------------------
# Metrics  (identical to the Qwen3-VL version)
# ---------------------------------------------------------------------------


def _norm(s: Any) -> str:
    return str(s).strip().lower()


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
    """Compute macro-averaged F1 (exact query+value match) and mean IoU.

    Each prediction is greedily matched against the unused ground-truth answers
    on normalised (query, value) — so a query with several answers needs each
    one matched separately, and an answer produced for an unanswerable query
    (one with no ground truth) lowers precision. F1 is computed per document,
    then averaged. IoU is measured only for matched pairs that also agree on
    the page `index`.

    Args:
        predictions: Raw model output strings (expected to be JSON).
        ground_truths: List of answer lists; each answer has
            {query, value, bounding_box, index}.
    """
    all_f1: list[float] = []
    all_iou: list[float] = []
    n_parse_errors = 0

    for pred_str, gt_answers in zip(predictions, ground_truths):
        try:
            pred_list = json.loads(pred_str)
            if not isinstance(pred_list, list):
                pred_list = []
        except (json.JSONDecodeError, ValueError):
            pred_list = []
            n_parse_errors += 1

        preds = [p for p in pred_list if isinstance(p, dict)]
        gt_unused = list(range(len(gt_answers)))

        tp = 0
        for pred in preds:
            pq_, pv = _norm(pred.get("query", "")), _norm(pred.get("value", ""))
            match_i = next(
                (
                    gi
                    for gi in gt_unused
                    if _norm(gt_answers[gi]["query"]) == pq_
                    and _norm(gt_answers[gi]["value"]) == pv
                ),
                None,
            )
            if match_i is None:
                continue
            tp += 1
            gt_unused.remove(match_i)
            gt = gt_answers[match_i]
            try:
                if int(pred.get("index", 0)) == int(gt.get("index", 0)):
                    pred_norm = _qwen_bbox_to_normalized(pred["box_2d"])
                    all_iou.append(_iou(pred_norm, list(gt["bounding_box"])))
            except (KeyError, TypeError, ValueError):
                pass

        n_pred = len(preds)
        n_gt = len(gt_answers)
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
# Evaluation
# ---------------------------------------------------------------------------


def visualize_predictions(
    images: list[Image.Image],
    output: str,
    gt_answers: list[dict],
    output_path: str,
) -> None:
    """Draw GT boxes (blue) and predicted boxes (red) on each page and save.

    Boxes are assigned to pages by their `index`. With one page, writes a single
    file at ``output_path``; with several, writes ``<stem>_p{i}<ext>`` per page.
    """
    try:
        items = json.loads(output)
        if not isinstance(items, list):
            items = []
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse model output for visualization")
        items = []

    pages = [im.copy() for im in images]
    draws = [ImageDraw.Draw(im) for im in pages]

    for ans in gt_answers:
        idx = int(ans.get("index", 0))
        bb = ans.get("bounding_box")
        if not 0 <= idx < len(pages) or bb is None or len(bb) != 4:
            continue
        W, H = pages[idx].size
        x0, y0, x1, y1 = bb
        draws[idx].rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline="blue", width=2)

    for item in items:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", 0))
        bbox = item.get("box_2d")
        if not 0 <= idx < len(pages) or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        W, H = pages[idx].size
        x0, y0, x1, y1 = bbox
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draws[idx].rectangle([px0, py0, px1, py1], outline="red", width=2)
        draws[idx].text((px0, max(0, py0 - 12)), str(item.get("query", "")), fill="red")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if len(pages) == 1:
        pages[0].save(output_path)
    else:
        stem, ext = os.path.splitext(output_path)
        for i, im in enumerate(pages):
            im.save(f"{stem}_p{i}{ext}")


@torch.inference_mode()
def _generate(model, processor, images: list[Image.Image], messages: list[dict]) -> str:
    """Greedy-decode the model's answer for one (images, prompt) example.

    Applies the chat template with a generation prompt and ``enable_thinking``
    off (Qwen3.5 would otherwise prefix a ``<think>`` block that breaks JSON
    parsing), feeds the (text, images) through the processor, and returns the
    decoded completion (the prompt tokens are stripped off).
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=EVAL_MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    trimmed = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def show_row(model, processor, row: pd.Series) -> None:
    """Generate for a single row and dump a visualization to output/out.jpeg."""
    FastModel.for_inference(model)
    images = _load_images(row["images"])
    messages = make_conversation(row)
    del messages[-1]  # drop the assistant turn — we want the model to produce it
    output = _generate(model, processor, images, messages)
    print(output)
    visualize_predictions(images, output, list(row["answers"]), "output/out.jpeg")
    print("Saved to output/out.jpeg")


def evaluate(
    model,
    processor,
    df: pd.DataFrame,
    output_dir: str | None = None,
    max_samples: int = EVAL_MAX_SAMPLES,
) -> dict[str, float]:
    FastModel.for_inference(model)
    df = df.head(max_samples)
    predictions: list[str] = []
    ground_truths: list[list[dict]] = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), desc="Evaluating", total=df.shape[0])
    ):
        images = _load_images(row["images"])
        messages = make_conversation(row)
        del messages[-1]
        output = _generate(model, processor, images, messages)
        predictions.append(output)
        ground_truths.append(list(row["answers"]))
        if output_dir is not None:
            visualize_predictions(
                images,
                output,
                list(row["answers"]),
                os.path.join(output_dir, f"row_{i:03}.jpeg"),
            )

    return compute_metrics(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


class _VQADataset(torch.utils.data.Dataset):
    """Lazily yields ``{"messages": conversation}`` rows for the collator.

    Decoding happens in ``make_conversation`` on access, so page images are
    materialised one batch at a time instead of holding every decoded image in
    memory at once. ``UnslothVisionDataCollator`` consumes the ``messages`` and
    produces masked, padded model inputs.
    """

    def __init__(self, df: pd.DataFrame):
        self._rows = [row for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return {"messages": make_conversation(self._rows[idx])}


def train(model, processor, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

    FastModel.for_training(model)

    train_dataset = _VQADataset(train_df)
    eval_dataset = _VQADataset(eval_df.head(20))

    # The collator masks the prompt (image + user text) and computes loss only on
    # the assistant JSON — the PyTorch equivalent of the JAX completion_mask.
    # Note: Qwen3.5 is a thinking model, but our assistant turns are pure JSON and
    # are rendered as complete turns here (no generation prompt), so no <think> is
    # introduced. If you change targets or see empty <think></think> in a rendered
    # training sample, render with the processor's non-thinking template.
    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part=_INSTRUCTION_PART,
        response_part=_RESPONSE_PART,
    )

    config = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=PEAK_LR,
        # JAX used warmup -> cosine decay to 1e-5; HF cosine decays to ~0, which
        # is close enough for a 10k-step run with the same 50-step warmup.
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,  # matches optax.clip_by_global_norm(1.0)
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=MAX_SEQ_LEN,
        # Vision SFT essentials: keep raw image columns and skip TRL's text-only
        # dataset prep — our collator does all preprocessing.
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_N_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="steps",
        save_steps=EVAL_EVERY_N_STEPS,
        logging_steps=10,
        output_dir=LORA_CKPT_DIR,
        report_to="tensorboard",
        logging_dir="/tmp/tensorboard/vqa_20260529_qwen3_5",
        seed=RANDOM_SEED,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )

    # Resume from the most recent checkpoint in LORA_CKPT_DIR if one exists, so a
    # re-run continues instead of restarting at step 0 (it restores model/adapter
    # weights, optimizer, LR schedule, RNG, and global step). get_last_checkpoint
    # returns None on a fresh run, and train(resume_from_checkpoint=None) starts
    # from scratch — whereas passing True would error when no checkpoint exists.
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = (
        get_last_checkpoint(LORA_CKPT_DIR) if os.path.isdir(LORA_CKPT_DIR) else None
    )
    if last_checkpoint:
        logger.info("Resuming LoRA fine-tuning from checkpoint: %s", last_checkpoint)
    else:
        logger.info("Starting LoRA fine-tuning for %d steps", MAX_STEPS)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving merged 16-bit model to %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Bake the LoRA adapters into the base weights and write a plain 16-bit
    # checkpoint — directly loadable by vLLM / transformers, no Unsloth needed.
    model.save_pretrained_merged(OUTPUT_DIR, processor, save_method="merged_16bit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    train_df, eval_df = load_splits()
    eval_df = eval_df.head(20)

    # --- Load base model ---
    logger.info("Loading base model…")
    model, processor = FastModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=LOAD_IN_4BIT,
        # Unsloth's checkpointing — long-context friendly, lower VRAM.
        use_gradient_checkpointing="unsloth",
        max_seq_length=MAX_SEQ_LEN,
    )

    # --- Evaluate base model ---
    logger.info("Evaluating base model on %d test samples…", len(eval_df))
    metrics_before = evaluate(model, processor, eval_df, output_dir="output/base_model")
    logger.info(
        "Base model  — F1: %.4f  IoU: %.4f",
        metrics_before["f1_exact_match"],
        metrics_before["iou"],
    )

    # --- Fine-tune (attach LoRA in place and train) ---
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # vision tower frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        use_rslora=False,
        random_state=RANDOM_SEED,
    )
    # TEMP: visualize one example before training (mirrors the JAX script).
    eval_rows = [row for _, row in eval_df.iterrows()]
    show_row(model, processor, eval_rows[min(8, len(eval_rows) - 1)])
    train(model, processor, train_df, eval_df)

    # --- Evaluate fine-tuned model (LoRA active, in memory) ---
    logger.info("Evaluating fine-tuned model…")
    metrics_after = evaluate(model, processor, eval_df, output_dir="output/ft_model")
    logger.info(
        "Fine-tuned  — F1: %.4f  IoU: %.4f",
        metrics_after["f1_exact_match"],
        metrics_after["iou"],
    )

    print("\n=== Results ===")
    print(
        f"Base model:  F1={metrics_before['f1_exact_match']:.4f}  "
        f"IoU={metrics_before['iou']:.4f}"
    )
    print(
        f"Fine-tuned:  F1={metrics_after['f1_exact_match']:.4f}  "
        f"IoU={metrics_after['iou']:.4f}"
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
