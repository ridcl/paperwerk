import io
import json
import logging
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pyarrow.parquet as pq
import qwix
from flax import nnx
from PIL import Image, ImageDraw
from tqdm import tqdm
from tunix.rl import reshard as reshard_lib
from tunix.sft import metrics_logger, peft_trainer

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl.loading import load_model, resolve_model_dir
from fabrique.models.qwen3vl.sampler import Qwen3VLSampler, load_sampler
from fabrique.models.qwen3vl.utils import encode_messages
from fabrique.models.qwen3vl.vision import VisionGridData
from fabrique.saving import save_qwen3vl_lora_merged
from fabrique.utils import show_hbm_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# Dataset schema (produced by datagen.builders.vqa_20260524_cuad_kvp10k):
#   images      list[binary]   — page image bytes, one entry per page
#   queries     list[string]   — answerable + unanswerable queries (shuffled)
#   answers     list[struct{               — one entry per answerable evidence
#                  query:        string,    —   the query this answers
#                  value:        string,    —   literal or derived answer
#                  bounding_box: list[f64], —   [x0,y0,x1,y1] normalised 0–1
#                  index:        int32,     —   which `images` entry holds it
#               }]
#   source      string
#   variant     string
#   page_start  int32
#   page_end    int32
#   split       string
#
# Differences vs. the kvp10k-only format this script supersedes:
#   - single `image` (bytes) -> `images` (list of bytes); a datapoint may span
#     several pages, so each answer carries the `index` of its page.
#   - `kvps[*].key` / `keys` -> `answers[*].query` / `queries`.
#   - a query may have several answers (e.g. every row of a table) and the
#     `queries` list also holds unanswerable queries that have no answer.
#
# Note: both data builders currently emit one image per datapoint, so the
# evaluation path feeds a single image per prompt through the public sampler
# API (which binds one image per prompt). Training already handles any number
# of images via `encode_messages`.

DATASET_PATH = "/data/paperwerk/vqa_20260524.parquet"

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = "/data/models/vqa-20260529-qwen3vl-4b"
LORA_CKPT_DIR = "/data/cache/vqa_20260529_lora_ckpts"

MESH = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))

BATCH_SIZE = 1
MAX_SEQ_LEN = 4096
MAX_IMAGE_SIZE = 896
EVAL_CACHE_SIZE = 4096
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_SAMPLES = 500

LORA_RANK = 16
LORA_ALPHA = float(2 * LORA_RANK)
_LORA_TARGETS = ".*q_proj|.*k_proj|.*gate_proj|.*up_proj|.*down_proj"
MAX_STEPS = 10_000
EVAL_EVERY_N_STEPS = 500

RANDOM_SEED = 42
TEST_FRACTION = 0.05

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
# Data formatting
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
    page_phrase = (
        "this document page" if n_images == 1 else f"these {n_images} document pages"
    )
    return (
        f"Answer the following queries about {page_phrase}. "
        'Return a JSON list where each item has "query", "value", "box_2d", and '
        '"index" fields:\n'
        '  - "query": the exact query string being answered, copied verbatim.\n'
        '  - "value": the answer — either the literal text from the document or '
        "a short derived value (e.g. a year extracted from a full date, or "
        '"Yes"/"No" for a yes/no question).\n'
        '  - "box_2d": the bounding box of the supporting evidence as '
        "[x0, y0, x1, y1] with coordinates in the 0–1000 range.\n"
        '  - "index": the 0-based page number (index into the provided images) '
        "where the evidence is located.\n"
        "A single query may have several answers — emit one item per answer. "
        "Some queries cannot be answered from the document; omit those entirely "
        "rather than guessing.\n"
        f"Queries:\n{query_list}"
    )


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
# Metrics
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
            pq, pv = _norm(pred.get("query", "")), _norm(pred.get("value", ""))
            match_i = next(
                (
                    gi
                    for gi in gt_unused
                    if _norm(gt_answers[gi]["query"]) == pq
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


def show_row(lora_model, processor, row):
    images = _load_images(row["images"])
    messages = make_conversation(row)
    del messages[-1]
    lora_model.config.remat_config = model_lib.RematConfig.NONE
    try:
        sampler = Qwen3VLSampler(lora_model, processor, cache_size=EVAL_CACHE_SIZE)
        prompt = sampler._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = sampler(
            prompts=[prompt],
            images=images,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
        )[0]
        print(output)
    finally:
        lora_model.config.remat_config = model_lib.RematConfig.BLOCK

    visualize_predictions(images, output, list(row["answers"]), "output/out.jpeg")
    print("Saved to output/out.jpeg")


def evaluate(
    sampler,
    df: pd.DataFrame,
    output_dir: str | None = None,
    max_samples: int = EVAL_MAX_SAMPLES,
) -> dict[str, float]:
    df = df.head(max_samples)
    predictions: list[str] = []
    ground_truths: list[list[dict]] = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), desc="Evaluating", total=df.shape[0])
    ):
        images = _load_images(row["images"])
        messages = make_conversation(row)
        del messages[-1]
        prompt = sampler._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = sampler(
            prompts=[prompt],
            images=images,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
        )[0]
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


def _gen_model_input_fn(batch) -> dict:
    return {
        "input_tokens": jnp.array(batch.input_tokens),
        "padding_mask": jnp.array(batch.input_mask).astype(jnp.bool_),
        "completion_mask": jnp.array(batch.completion_mask),
        "positions": jnp.array(batch.positions),
        "pixel_values": jnp.array(batch.pixel_values, dtype=jnp.bfloat16),
        "vision_grid": batch.vision_grid,
    }


def _loss_fn(
    model: model_lib.Qwen3VL,
    input_tokens: jax.Array,
    positions: jax.Array,
    pixel_values: jax.Array,
    vision_grid: VisionGridData,
    padding_mask: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
    logits, _ = model(
        input_tokens,
        positions,
        pixel_values,
        vision_grid,
        cache=None,
        padding_mask=padding_mask,
    )
    logits = logits[:, :-1, :]
    targets = input_tokens[:, 1:]
    mask = completion_mask[:, 1:].astype(jnp.float32)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    )
    return jnp.sum(token_loss * mask) / jnp.sum(mask)


class _DataLoader:
    """Iterator that encodes VQA rows as batched EncodedBatch objects."""

    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        vcfg: model_lib.VisionModelConfig,
        batch_size: int,
        max_seq_len: int,
        num_epochs: int = 1,
    ):
        self._df = df
        self._processor = processor
        self._vcfg = vcfg
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._num_epochs = num_epochs

    def __iter__(self):
        for epoch in range(self._num_epochs):
            df = self._df.sample(frac=1, random_state=RANDOM_SEED + epoch)
            buffer: list[list[dict]] = []
            for _, row in df.iterrows():
                buffer.append(make_conversation(row))
                if len(buffer) == self._batch_size:
                    yield encode_messages(
                        self._processor,
                        buffer,
                        loss_roles={"assistant"},
                        vcfg=self._vcfg,
                        max_seq_len=self._max_seq_len,
                        padding=True,
                        pad_to_multiple_of=1024,
                        truncation=True,
                    )
                    buffer = []


def _get_lora_model(
    base_model: model_lib.Qwen3VL,
    mesh: jax.sharding.Mesh,
) -> model_lib.Qwen3VL:
    lora_provider = qwix.LoraProvider(
        module_path=_LORA_TARGETS, rank=LORA_RANK, alpha=LORA_ALPHA
    )
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **base_model.get_model_input()
    )
    # Fix sharding metadata: LoRA A/B are rank-2 but inherit rank-3 specs from
    # Einsum weights — trim the extra axis so nnx.get_partition_spec is valid.
    for _, node in nnx.iter_graph(lora_model):
        if isinstance(node, nnx.Variable) and node.has_metadata("out_sharding"):
            sharding = node.get_metadata()["out_sharding"]
            if sharding and len(sharding) > len(node.shape):
                node.set_metadata("out_sharding", tuple(sharding[: len(node.shape)]))
    with mesh:
        graph_def, state = nnx.split(lora_model)
        default_memory_kind = jax.devices()[0].default_memory().kind
        dst_shardings = jax.tree_util.tree_map(
            lambda x: jax.sharding.NamedSharding(
                mesh, x, memory_kind=default_memory_kind
            ),
            nnx.get_partition_spec(state),
        )
        lora_model = nnx.merge(
            graph_def, reshard_lib.reshard_pytree(state, dst_shardings)
        )
    return lora_model


def train(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

    config = model_lib.ModelConfig.qwen3vl_4b()
    config.remat_config = model_lib.RematConfig.BLOCK
    processor, base_model = load_model(MODEL_ID, mesh=MESH, config=config)
    show_hbm_usage()

    lora_model = _get_lora_model(base_model, mesh=MESH)
    show_hbm_usage()

    # TEMP: Visualize before training
    eval_rows = [row for _, row in eval_df.iterrows()]
    show_row(lora_model, processor, eval_rows[min(8, len(eval_rows) - 1)])

    num_epochs = math.ceil(MAX_STEPS / len(train_df))
    train_loader = _DataLoader(
        train_df,
        processor,
        config.vision_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_epochs=num_epochs,
    )
    eval_loader = _DataLoader(
        eval_df.head(20),
        processor,
        config.vision_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_epochs=1,
    )

    logging_opts = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/vqa_20260529_qwen3vl",
        flush_every_n_steps=EVAL_EVERY_N_STEPS,
    )
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        metrics_logging_options=logging_opts,
        checkpoint_root_directory=LORA_CKPT_DIR,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=2e-4,
                warmup_steps=50,
                decay_steps=MAX_STEPS,
                end_value=1e-5,
            ),
            weight_decay=0.01,
        ),
    )
    trainer = peft_trainer.PeftTrainer(
        lora_model, optimizer, training_config
    ).with_gen_model_input_fn(_gen_model_input_fn)
    trainer.loss_fn = _loss_fn
    trainer.eval_loss_fn = _loss_fn

    logger.info("Starting LoRA fine-tuning for %d steps", MAX_STEPS)
    with MESH:
        trainer.train(train_loader, eval_ds=eval_loader)

    logger.info("Saving merged LoRA model to %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_qwen3vl_lora_merged(
        model_id_or_dir=MODEL_ID,
        output_dir=OUTPUT_DIR,
        lora_model=lora_model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    train_df, eval_df = load_splits()
    eval_df = eval_df.head(20)

    # --- Evaluate base model ---
    logger.info("Loading base model for evaluation…")
    sampler = load_sampler(MODEL_ID, mesh=MESH, cache_size=EVAL_CACHE_SIZE)
    logger.info(
        "Evaluating base model on %d test samples…", min(EVAL_MAX_SAMPLES, len(eval_df))
    )

    metrics_before = evaluate(sampler, eval_df, output_dir="output/base_model")
    logger.info(
        "Base model  — F1: %.4f  IoU: %.4f",
        metrics_before["f1_exact_match"],
        metrics_before["iou"],
    )
    del sampler  # free HBM before training

    # --- Fine-tune ---
    train(train_df, eval_df)

    # --- Evaluate fine-tuned model ---
    logger.info("Loading fine-tuned model for evaluation…")
    sampler_ft = load_sampler(OUTPUT_DIR, mesh=MESH, cache_size=EVAL_CACHE_SIZE)
    metrics_after = evaluate(sampler_ft, eval_df, output_dir="output/ft_model")
    logger.info(
        "Fine-tuned  — F1: %.4f  IoU: %.4f",
        metrics_after["f1_exact_match"],
        metrics_after["iou"],
    )

    print("\n=== Results ===")
    print(
        f"Base model:  F1={metrics_before['f1_exact_match']:.4f}  IoU={metrics_before['iou']:.4f}"
    )
    print(
        f"Fine-tuned:  F1={metrics_after['f1_exact_match']:.4f}  IoU={metrics_after['iou']:.4f}"
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
