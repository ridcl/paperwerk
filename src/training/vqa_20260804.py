"""LoRA fine-tuning of Qwen3-VL on the CUAD-synthetic + XBRL-tables VQA datasets.

Successor to ``vqa_20260529.py``: same task, formatting, prompt, target schema,
LoRA config, and metrics, but trained on a different data mix. KVP10k is dropped;
the model now learns from two HF Hub datasets combined:

    ridcl/cuad-synthetic-vqa  — synthetic CUAD-style contracts (the CUAD rows of
                                the old combined parquet, republished standalone
                                by ``datagen.builders.cuad_synthetic_vqa_20260622``)
    ridcl/xbrl-tables         — SEC inline-XBRL financial tables with grounded,
                                metadata-exact answers (``xbrl_tables_20260619``)

Both datasets share the unified VQA schema (``images``, ``queries``, ``answers``,
``source``, ``variant``, ``page_start``, ``page_end``), so they concatenate
directly; the trainer does its own document-level train/test split (see
``load_splits``).

This is the PyTorch/Unsloth counterpart of ``vqa_20260529_jax.py`` (JAX + Flax
via tunix/fabrique). It trains the *same* task with the *same* data formatting,
prompt, target schema, LoRA target modules, and evaluation metrics — only the
training engine differs:

    JAX version                         Unsloth version (this file)
    ---------------------------------   ---------------------------------------
    fabrique.load_model / sampler       FastVisionModel.from_pretrained
    qwix.apply_lora_to_model            FastVisionModel.get_peft_model
    tunix peft_trainer.PeftTrainer      trl.SFTTrainer + UnslothVisionDataCollator
    encode_messages (loss on assistant) collator train_on_responses_only=True
    save_qwen3vl_lora_merged            model.save_pretrained_merged(merged_16bit)

The merged 16-bit checkpoint it writes loads directly in vLLM / transformers
without Unsloth or PEFT.

Dataset schema (shared by both HF datasets above):
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

The two datasets share this schema exactly but populate it differently:
CUAD-synthetic rows are one page each (``page_start == page_end``, every answer
``index == 0``), whereas XBRL-tables rows are already multi-page windows. At
serving time we apply VQA to several consecutive pages at once, so data
preparation here groups each document's rows into overlapping page windows of up
to ``MAX_PAGES`` images (see ``load_splits``), concatenating their
images/queries/answers and re-indexing every answer onto its page's position
within the merged sequence. CUAD's single-page rows get tiled into multi-page
windows; an XBRL row already at/over ``MAX_PAGES`` passes through as its own
window. Every path downstream already handles an arbitrary number of pages (each
answer carries the `index` of its page), matching the JAX version.
"""

# isort: skip_file  (import order below is deliberate — unsloth MUST come first)
# Unsloth patches transformers/trl on import, so it MUST come first — importing
# it after them silently disables the optimizations (and Unsloth warns loudly).
import unsloth  # noqa: F401  (side-effecting; keep above transformers/trl)
from transformers import AutoModelForImageTextToText, AutoProcessor
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from tqdm import tqdm  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

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

# Page windowing: train on sequences of consecutive pages, mirroring how VQA is
# applied at serving time. Each document's single-page rows are tiled into
# windows of up to MAX_PAGES images. PAGE_STRIDE < MAX_PAGES makes consecutive
# windows overlap (MAX_PAGES=5, PAGE_STRIDE=3 -> pages 0-4, 3-7, 6-10, ...), so a
# value whose evidence sits near a window boundary is still seen whole in some
# window. Set PAGE_STRIDE == MAX_PAGES for non-overlapping windows.
MAX_PAGES = 3
PAGE_STRIDE = 2

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
# NB: a window now holds up to MAX_PAGES pages, so a packed example can carry
# ~MAX_PAGES times the vision tokens of a single page. With MAX_SEQ_LEN below the
# collator truncates anything past the limit — if that starts clipping assistant
# targets, lower MAX_IMAGE_SIZE (fewer vision tokens/page) before raising the
# length, since this 4096 was chosen to fit the <80 GB GPUs' memory budget.
MAX_SEQ_LEN = 32768 if GPU_MEMORY_GB > 70 else 4096
MAX_IMAGE_SIZE = 1792 if GPU_MEMORY_GB > 70 else 896
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_SAMPLES = 500

LOAD_IN_4BIT = False

LORA_RANK = 16
LORA_ALPHA = 2 * LORA_RANK
LORA_TARGET_MODULES = ["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

MAX_STEPS = 100_000
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


def _concat_window(rows: list[pd.Series]) -> dict:
    """Merge consecutive page rows of one document into a single datapoint.

    Concatenates the rows' images in page order, unions their query lists
    (de-duplicated, order preserved), and re-indexes every answer onto its
    page's position within the merged image sequence: a row landing at image
    offset ``o`` shifts each of its answers from local ``index`` to ``o + index``
    (rows are single-page today, so this is just the slot number, but the offset
    keeps it correct should a source row ever carry multiple pages). Answers are
    copied rather than mutated because overlapping windows revisit the same row.
    """
    images: list[bytes] = []
    queries: list[str] = []
    seen_queries: set[str] = set()
    answers: list[dict] = []
    for row in rows:
        offset = len(images)  # where this row's pages land in the merged window
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


def _document_windows(rows: list[pd.Series]) -> list[dict]:
    """Tile one document's page rows into overlapping ≤MAX_PAGES-image windows.

    Rows are ordered by page, then a window grows from each start until adding
    the next row would exceed MAX_PAGES images; the start then advances by
    PAGE_STRIDE rows (< MAX_PAGES ⇒ overlap). A single row larger than MAX_PAGES
    still forms its own window rather than stalling. Grouping is positional over
    the pages that survived datagen, so a window spans available consecutive
    rows even if an intermediate page was dropped upstream.
    """
    rows = sorted(rows, key=lambda r: int(r["page_start"]))
    n = len(rows)
    windows: list[dict] = []
    start = 0
    while start < n:
        end, n_imgs = start, 0
        while end < n:
            k = len(rows[end]["images"])
            if end > start and n_imgs + k > MAX_PAGES:
                break
            n_imgs += k
            end += 1
        windows.append(_concat_window(rows[start:end]))
        if end >= n:
            break
        start += PAGE_STRIDE
    return windows


def _windowed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace single-page rows with multi-page windows, one group per document."""
    rows: list[dict] = []
    for _, doc in df.groupby("source", sort=False):
        rows.extend(_document_windows([r for _, r in doc.iterrows()]))
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
        logger.warning(
            "Dropped %d row(s) with undecodable images", n_before - len(df)
        )
    # Split by `source`, not by row: a document's overlapping page windows share
    # images, so a row-level split would leak held-out pages into training. For
    # CUAD a `source` is one variant of one document and groups all its page rows,
    # so each document's pages stay on a single side. For XBRL each window is its
    # own `source`, so two context-sharing windows of one filing can still land on
    # opposite sides (a minor, accepted page-leakage risk). Seeded for repro.
    sources = df["source"].drop_duplicates()
    test_sources = set(sources.sample(frac=TEST_FRACTION, random_state=RANDOM_SEED))
    train_df = _windowed(df[~df["source"].isin(test_sources)])
    test_df = _windowed(df[df["source"].isin(test_sources)])
    logger.info(
        "Train: %d windows (%d docs), Test: %d windows (%d docs)",
        len(train_df),
        len(sources) - len(test_sources),
        len(test_df),
        len(test_sources),
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
