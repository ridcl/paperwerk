"""Train a VQA model with visual grounding.

Expected dataset schema:
    images      list[binary]   — page image bytes, one entry per page
    queries     list[string]   — answerable + unanswerable queries (shuffled)
    answers     list[struct{               — one entry per answerable evidence
                   query:        string,    —   the query this answers
                   value:        string,    —   literal or derived answer
                   bounding_box: list[f64], —   [x0,y0,x1,y1] normalised 0-1
                   index:        int32,     —   which `images` entry holds it
                }]
    source      string
    variant     string
    page_start  int32
    page_end    int32

"""

## isort: skip_file  (import order below is deliberate — unsloth MUST come first)
import unsloth  # noqa: F401  (side-effecting; keep above transformers/trl)
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
)
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import io
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

# Message/target formatting lives in the serving package so the prompt, target
# JSON schema, and bbox conventions stay identical between training and serving.
from paperwerk.vqa import (  # noqa: E402
    make_target,
    make_user_message,
    pil_image_part,
    qwen_bbox_to_normalized,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (mirrors vqa_20260529_jax.py)
# ---------------------------------------------------------------------------

DATASET = "ridcl/vqa_kvp10k_synth"

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = "/data/models/vqa-20260622-qwen3vl-4b"
LORA_CKPT_DIR = "/data/cache/vqa_20260622_lora_ckpts"

GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3

# Each datapoint is a whole document: all of its pages are packed into one
# example, mirroring how VQA is applied to a full document at serving time.

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1

MAX_SEQ_LEN = 128_000
MAX_IMAGE_SIZE = 1792 if GPU_MEMORY_GB > 70 else 896
EVAL_MAX_NEW_TOKENS = 8_000
EVAL_MAX_SAMPLES = 500

LOAD_IN_4BIT = False

LORA_RANK = 16
LORA_ALPHA = 2 * LORA_RANK
LORA_TARGET_MODULES = ["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

MAX_STEPS = 25_000
EVAL_EVERY_N_STEPS = 500
# Stop early if eval_loss hasn't improved for this many evals (× EVAL_EVERY_N_STEPS
# steps). A few evals of patience rides out normal eval-loss noise; combined with
# load_best_model_at_end, the final model is the lowest-eval_loss checkpoint, not
# the overfit tail.
EARLY_STOP_PATIENCE = 5
WARMUP_STEPS = 50
PEAK_LR = 1e-4
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


def _concat_document(rows: list[pd.Series]) -> dict:
    """Merge all page rows of one document into a single datapoint.

    Concatenates the rows' images in page order, unions their query lists
    (de-duplicated, order preserved), and re-indexes every answer onto its
    page's position within the merged image sequence: a row landing at image
    offset ``o`` shifts each of its answers from local ``index`` to ``o + index``
    (single-page rows collapse this to the slot number; the offset keeps it
    correct for rows that already carry several pages).
    """
    images: list[bytes] = []
    queries: list[str] = []
    seen_queries: set[str] = set()
    answers: list[dict] = []
    for row in rows:
        offset = len(images)  # where this row's pages land in the merged document
        images.extend(row["images"])
        for q in row["queries"]:
            if q not in seen_queries:
                seen_queries.add(q)
                queries.append(q)
        for ans in row["answers"]:
            ans = dict(ans)
            ans["index"] = offset + int(ans["index"])
            answers.append(ans)
    first, last = rows[0], rows[-1]
    return {
        "images": images,
        "queries": queries,
        "answers": answers,
        "source": first["source"],
        "variant": first["variant"],
        "page_start": int(first["page_start"]),
        "page_end": int(last["page_end"]),
    }


def _group_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each document's page rows into one all-pages datapoint.

    Rows sharing a ``source`` are one document; they are ordered by page and
    concatenated whole (no windowing), so the model trains on every page of a
    document at once. Grouping is positional over the pages that survived
    datagen, so a page dropped upstream simply shifts later pages down one slot.
    """
    rows: list[dict] = []
    for _, doc in df.groupby("source", sort=False):
        ordered = sorted(
            (r for _, r in doc.iterrows()), key=lambda r: int(r["page_start"])
        )
        rows.append(_concat_document(ordered))
    return pd.DataFrame(rows).reset_index(drop=True)


def _hf_to_df(repo: str) -> pd.DataFrame:
    """Load one HF Hub VQA dataset into a DataFrame, batching the pandas convert.

    `images` is list<binary> with 32-bit offsets; across a full dataset the
    decoded bytes can exceed pyarrow's 2 GiB offset limit, at which point a
    single `to_pandas()` raises "Nested data conversions not implemented for
    chunked array outputs". `to_pandas(batched=True)` converts one bounded batch
    at a time (a contiguous binary array per batch), keeping the nested
    conversion on the supported path; concatenate the per-batch frames.
    """
    ds = load_dataset(repo)["vqa_kvp10k_synth"]
    frames = list(ds.to_pandas(batched=True, batch_size=512))
    return pd.concat(frames, ignore_index=True)


def _row_images_ok(images_bytes: Any) -> bool:
    """True iff every image in the row decodes — guards against corrupt bytes.

    The source dataset can carry an occasional truncated/corrupt page image
    (valid header, undecodable stream), which otherwise raises deep inside the
    DataLoader worker (`Image.open(...).convert(...)`) and kills the whole run.
    We force a full decode here (open is lazy) so a bad row is caught up front.
    """
    for b in images_bytes:
        try:
            Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return False
    return True


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _hf_to_df(DATASET)
    # Drop rows with an undecodable image before splitting, so one corrupt page
    # can't crash a long training run mid-epoch.
    n_before = len(df)
    df = df[df["images"].map(_row_images_ok)].reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Dropped %d row(s) with undecodable images", n_before - len(df))
    # Split by `source` (one document): all of a document's pages stay on a single
    # side, so no page leaks between train and test. Since each document collapses
    # to one datapoint, this is also a clean datapoint-level split. Seeded for repro.
    sources = df["source"].drop_duplicates()
    test_sources = set(sources.sample(frac=TEST_FRACTION, random_state=RANDOM_SEED))
    train_df = _group_documents(df[~df["source"].isin(test_sources)])
    test_df = _group_documents(df[df["source"].isin(test_sources)])
    logger.info(
        "Train: %d documents, Test: %d documents",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Data formatting  (identical to the JAX version)
# ---------------------------------------------------------------------------


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def _load_images(images_bytes: Any) -> list[Image.Image]:
    """Decode and downscale every page image of a datapoint, in order."""
    return [_load_image(b) for b in images_bytes]


def make_conversation(row: pd.Series) -> list[dict]:
    """Build the 2-turn chat (user prompt+images, assistant JSON target).

    The message structure — content lists of ``{"type": "image", "image": PIL}``
    and ``{"type": "text", "text": ...}`` (via ``pil_image_part``) — is exactly
    what ``UnslothVisionDataCollator`` consumes (it runs the processor's chat
    template and image preprocessing internally). The prompt and target JSON come
    from ``paperwerk.vqa`` so serving reproduces them byte-for-byte.
    """
    images = _load_images(row["images"])
    queries = list(row["queries"])
    answers = list(row["answers"])
    return [
        make_user_message(images, queries, image_part=pil_image_part),
        {
            "role": "assistant",
            "content": [{"type": "text", "text": make_target(answers)}],
        },
    ]


# ---------------------------------------------------------------------------
# Metrics  (identical to the JAX version)
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
                    pred_norm = qwen_bbox_to_normalized(pred["box_2d"])
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

    Mirrors the JAX sampler call: applies the chat template with a generation
    prompt, feeds the (text, images) through the processor, and returns the
    decoded completion (the prompt tokens are stripped off).
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


@torch.inference_mode()
def _generate_plain(
    model, processor, images: list[Image.Image], queries: list[str]
) -> str:
    """Greedy-decode answers for (images, queries) via the stock HF path.

    Reproduces the standalone check that confirmed the trained checkpoint is
    healthy: it builds the user turn straight from ``make_user_message`` (so the
    prompt/image formatting is byte-identical to training and to ``_generate``),
    forces the model into eval mode, then greedily decodes. The decoding call
    itself is the same as ``_generate``; the one material difference is that this
    helper calls ``model.eval()`` instead of relying on the caller to switch the
    model to inference mode.

    That difference is the whole point. Calling ``_generate``/``model.generate``
    directly on an Unsloth ``FastVisionModel`` still in its *training*
    configuration (gradient checkpointing on, training-mode kernels) emits
    degenerate output — repeated null bytes on image inputs, prompt echo on
    text. Running through this plain eval-mode path (equivalently: a stock
    ``transformers`` load of the merged 16-bit checkpoint) yields valid grounded
    JSON for both single- and multi-image inputs. Use it to sanity-check a
    checkpoint without the Unsloth inference-mode toggle. For an Unsloth model
    in-session, ``FastVisionModel.for_inference(model)`` is the canonical switch.
    """
    model.eval()
    user = make_user_message(images, queries, image_part=pil_image_part)
    text = processor.apply_chat_template(
        [user], tokenize=False, add_generation_prompt=True
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
    FastVisionModel.for_inference(model)
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
    FastVisionModel.for_inference(model)
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
    materialised one batch at a time (matching the JAX ``_DataLoader``) instead
    of holding every decoded image in memory at once. ``UnslothVisionDataCollator``
    consumes the ``messages`` and produces masked, padded model inputs.
    """

    def __init__(self, df: pd.DataFrame):
        self._rows = [row for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return {"messages": make_conversation(self._rows[idx])}


def train(model, processor, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

    FastVisionModel.for_training(model)

    train_dataset = _VQADataset(train_df)
    eval_dataset = _VQADataset(eval_df.head(20))

    # The collator masks the prompt (image + user text) and computes loss only on
    # the assistant JSON — the PyTorch equivalent of the JAX completion_mask.
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
        # Restore the lowest-eval_loss checkpoint as the final model (instead of the
        # possibly-overfit last step) so the merged export below is the best one.
        # Requires save_strategy == eval_strategy and save_steps % eval_steps == 0.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        output_dir=LORA_CKPT_DIR,
        report_to="tensorboard",
        logging_dir="/tmp/tensorboard/vqa_20260622_qwen3vl",
        seed=RANDOM_SEED,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
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
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=LOAD_IN_4BIT,
        # Unsloth's checkpointing — long-context friendly, lower VRAM.
        use_gradient_checkpointing="unsloth",
        max_seq_length=MAX_SEQ_LEN,
    )

    # --- Fine-tune (attach LoRA in place and train) ---
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # vision tower frozen, as in the JAX run
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
    train(model, processor, train_df, eval_df)

    # --- Evaluate fine-tuned model (LoRA active, in memory) ---
    logger.info("Evaluating fine-tuned model…")
    model, processor = None, None
    processor = AutoProcessor.from_pretrained(OUTPUT_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        OUTPUT_DIR, dtype=torch.bfloat16, device_map="cuda"
    )
    metrics_after = evaluate(model, processor, eval_df, output_dir="output/ft_model")
    logger.info(
        "Fine-tuned  — F1: %.4f  IoU: %.4f",
        metrics_after["f1_exact_match"],
        metrics_after["iou"],
    )

    model.push_to_hub("paperwerk-vqa")
    processor.push_to_hub("paperwerk-vqa")


if __name__ == "__main__" and "__file__" in globals():
    main()
