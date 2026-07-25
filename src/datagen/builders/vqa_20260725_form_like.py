"""Build the `vqa-form-like` dataset: synthetic filled forms → unified VQA schema.

This is a self-contained *builder*. It renders synthetic documents from the
form-like Jinja templates under ``datagen/assets/templates`` (filling every
field with values invented by a **locally-running Gemma-4** served by vLLM),
rasterizes each page to an image, and emits one datapoint per document in the
same schema as the other VQA builders (``vqa_20260524_kvp10k.py`` /
``vqa_20260524_cuad.py``), so this parquet can be mixed with them downstream::

    images       <- every rendered page of the document (image bytes)
    queries      <- each field name, unique, in first-appearance order
    answers      <- one per filled field occurrence:
                       query        = field name
                       value        = synthesized value (as rendered)
                       bounding_box = [x0, y0, x1, y1], per-page, 0..1, top-left
                       index        = 0-based page index within `images`
    source       <- "<class>/<template>#<n>"
    variant      <- "form-like"
    page_start   <- 0
    page_end     <- n_pages - 1
    split        <- --split (default "train")

Only *form-like* document classes are used — forms, invoices, certificates,
product lists, and the like (short, structured; not prose). The exact set is
the ``FORM_LIKE_CLASSES`` constant below, curated by structural + semantic
analysis of the template corpus. **Edit that list freely** to add/remove
classes; the builder samples a random valid template file from each listed
class at run time (skipping parse-broken or near-empty templates).

Signature-named fields are rendered as procedural SVG scrawls
(``datagen.signatures``) rather than text, and are excluded from queries /
answers. With ``--augment-ratio`` a fraction of documents get scanner/phone
augmentation (``datagen.augment``); the rest are clean.

Prerequisite — Gemma-4 served locally by vLLM (see README)::

    docker run --gpus all --shm-size=16g -p 8000:8000 paperwerk-serve:latest \\
      /venv/bin/vllm serve google/gemma-4-E4B-it --max-model-len 65536 \\
      --enable-auto-tool-choice --tool-call-parser gemma4 --tensor-parallel-size 2

Run::

    python -m datagen.builders.vqa_20260725_form_like -n 200
    python -m datagen.builders.vqa_20260725_form_like -n 500 --augment-ratio 0.5
    python -m datagen.builders.vqa_20260725_form_like -n 100 --no-signatures \\
        -o /data/paperwerk/vqa_form_like.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment
from pdf2image import convert_from_bytes

from paperwerk.llm import LLM

import datagen
from datagen.augment import PROFILES, augment, augment_geometric
from datagen.render import render
from datagen.signatures import inject_signatures
from datagen.templates import discover_fields
from datagen.values import synthesize_values

# ---------------------------------------------------------------------------
# Form-like template classes (EDITABLE).
#
# Curated from the template corpus: classes whose documents are short and
# full of structured fields (names, dates, amounts, quantities) — forms,
# invoices, certificates, requests, worksheets, slips, datasheets, product
# lists, etc. Deliberately excludes prose-heavy types (biography, cv/resume,
# reports, papers, manuals, guides, newsletters, ...). Add or remove entries
# as you see fit; unknown names are simply skipped with a warning.
# ---------------------------------------------------------------------------
FORM_LIKE_CLASSES: list[str] = [
    "academic_schedule_change_form", "academic_transcript_request", "accident_form",
    "account_reset_form", "ach_authorization_form", "additional_resources_form",
    "address_change_form", "adjustment_request", "administrative_performance_appraisal_form",
    "admission_application", "adoption_credit", "advising_worksheet", "affidavit",
    "affidavit_form", "affidavit_of_financial_support", "alumni_information_form",
    "apartment_request", "appeal_form", "application", "application_for_employment_authorization",
    "application_for_i20", "application_form", "application_packet", "asset_form",
    "asset_worksheet", "assistance_animal_health_professional_form",
    "audio_services_request_form", "authorization_form", "authorization_statement",
    "award_change_form", "background_check_form", "background_investigator_request_form",
    "banking_options", "beneficiary_designation", "bid_response", "billing_inquiry_form",
    "birthday_cake_order_form", "board_membership_roster", "budget_request",
    "building_permit_application", "business_vendor_registration_form", "campus_forms_request",
    "cancer_registry_form", "catering_order_form", "central_registry_release_of_information_form",
    "certificate", "certificate_of_compliance", "certificate_of_exemption", "certification_form",
    "certification_of_retirement", "chain_of_custody_form", "change_form", "change_of_address",
    "change_of_address_form", "change_of_address_request_form", "change_of_information_form",
    "change_of_name_application", "change_of_program_form", "chargeback_request",
    "check_payment_remittance_slip", "check_request", "check_request_form", "checklist",
    "chemical_hygiene_permit", "child_support_verification_form", "claim_form",
    "college_transcript_request", "commercial_invoice", "complaint_form", "compliance_form",
    "conformance_statement", "consent_form", "contact_list",
    "controlled_substances_discrepancy_form", "course_change_request",
    "course_enrollment_application", "course_enrollment_request", "course_evaluation_request",
    "course_replacement_request", "course_request", "course_roster_form",
    "credit_card_authorization", "credit_card_authorization_form", "credit_card_request_form",
    "critical_illness_claim_form", "customs_declaration", "data_retrieval_request", "data_sheet",
    "datasheet", "declaration_of_conformity", "degree_audit_form", "degree_verification_request",
    "dependent_verification_worksheet", "deposit_form", "diploma_request", "diploma_request_form",
    "direct_charge_worksheet", "direct_deposit_authorization",
    "direct_deposit_authorization_form", "direct_deposit_form", "direct_pay_request",
    "disability_claim_form", "disability_claim_statement", "disability_verification",
    "disability_verification_form", "doctoral_defense_form", "document_request",
    "documentation_submission_form", "donation_form", "elevation_certificate",
    "emergency_contact_form", "employee_gate_permit_application",
    "employee_medical_certification", "employee_withholding_allowance_certificate",
    "employer_data_sheet", "enrollment_form", "enrollment_forms", "enrollment_verification_form",
    "entry_forms", "environmental_product_declaration", "equipment_request_form",
    "eu_declaration_of_conformity", "evaluation_form", "event_application",
    "event_ticket_request", "exam_form", "exemption_certificate", "expense_reimbursement_form",
    "faculty_conference_travel_supplemental_form", "fafsa_form", "fafsa_verification_worksheet",
    "fee_payment_form", "fee_transfer_details", "field_experience_application",
    "financial_affidavit", "financial_aid_application", "financial_aid_form",
    "financial_aid_request", "financial_disclosure_form", "form", "form_1099c", "form_5500",
    "form_700u", "fsa_application", "funding_request_form", "general_approval_application",
    "graduate_certificate_application", "grant_application", "guest_account_request_form",
    "guided_study_form", "health_assessment_form", "health_forms_checklist",
    "health_history_form", "health_information_disclosure_request", "health_information_form",
    "health_questionnaire", "health_screening_form", "honors_thesis_form", "hotel_listing",
    "hsa_transfer_request", "iecex_certificate_of_conformity", "immigration_form",
    "immunization_exemption_certificate", "immunization_form", "incentive_reimbursement_form",
    "income_expense_verification_form", "income_worksheet", "independent_contractor_form",
    "indoor_air_quality_request_form", "insurance_enrollment_form",
    "insurance_waiver_request_form", "intake_form", "internship_appeal_form",
    "internship_application", "internship_application_form", "invoice", "irs_form",
    "irs_tax_form", "job_application", "job_requisition_form", "lab_use_request",
    "leave_request_form", "license_explanation_form", "life_insurance_enrollment_form",
    "loan_application", "loan_deferment_request", "long_term_disability_claim_form",
    "lost_and_found_form", "mail_order_form", "mailing_information_form",
    "mandatory_information_form", "marital_status_form", "material_safety_data_sheet",
    "meal_plan_accommodation_verification_form", "meal_plan_request", "medical_evaluation_form",
    "medical_form", "medical_health_history_form", "medical_history_form",
    "medical_questionnaire", "membership_application", "membership_form", "merchant_application",
    "mileage_reimbursement_form", "minor_authorization_form", "minor_evaluation_request_form",
    "motor_vehicle_purchase_approval_form", "name_change_form", "name_change_request",
    "new_employee_registration_form", "new_fund_request_form", "non_degree_application",
    "observation_form", "onboarding_form", "order_form", "osha_form",
    "out_of_state_tuition_waiver_application", "parent_asset_worksheet",
    "parent_statement_of_income", "parental_assets_worksheet", "parking_appeal_form",
    "parking_form", "parking_permit_application", "parking_registration", "parking_request_form",
    "pass_request_form", "passport_application", "pay_stub", "payment_request_form",
    "payment_slip", "payroll_application", "payroll_deduction_authorization",
    "payroll_deduction_form", "peer_teaching_evaluation_form", "performance_evaluation",
    "performance_review_form", "permission_to_publish_form", "personal_data_change_request",
    "personal_data_form", "petition_form", "petty_cash_request", "plan_of_study_form",
    "practicum_goals_objectives_and_tasks_form", "pre_designation_form",
    "pre_trip_travel_info_form", "prescription_order_form", "price_list", "proctor_approval_form",
    "product_certification", "product_data_sheet", "product_datasheet", "product_sheet",
    "professional_judgment_request_form", "program_application", "program_approval_form",
    "program_request", "project_change_request", "promotion_application", "promotion_checklist",
    "proposal_transmittal_form", "purchase_order", "purchase_request", "questionnaire",
    "quotation", "quotation_request", "quote_request", "readmission_application",
    "reasonable_adjustment_application", "receipt_form", "recommendation_form",
    "reduced_course_load_request_form", "reference_check_form", "referral_form",
    "refund_application_form", "refund_request", "registration_form", "registration_request_form",
    "reimbursement_form", "reimbursement_request", "reimbursement_request_form",
    "release_of_information_form", "religious_exemption_form", "remittance_slip",
    "remote_work_request_form", "replacement_diploma_request",
    "request_for_access_to_official_record", "request_for_bid", "request_for_bids",
    "request_for_portability", "request_for_quotation", "request_for_quotations",
    "request_for_temporary_paid_leaves", "request_form", "request_to_prevent_disclosure",
    "requirements_worksheet", "research_activities_safety_checklist", "research_fund_request",
    "research_maintenance_form", "reservation_form", "reservation_request",
    "retired_staff_request_form", "retirement_contribution_election_form",
    "retirement_plan_election_form", "room_use_request", "safety_data_sheet",
    "safety_information_form", "sales_and_use_tax_certificate", "schedule_change_form",
    "scholarship_application", "scholarship_award", "scholarship_donation_form",
    "scholarship_listing", "scholarship_recommendation_form", "school_performance_fact_sheet",
    "service_award_form", "service_request_form", "sevis_form_i20", "sick_leave_transfer_form",
    "sign_language_request_form", "sign_up_form", "social_security_number_change",
    "spare_parts_list", "speaker_course_submission_form", "special_circumstances_request",
    "special_consideration_request_form", "sponsorship_form", "spouse_dependent_discount_request",
    "statement_of_award", "statement_of_educational_purpose", "stock_gift_form",
    "student_application", "student_checklist", "student_financial_aid_form",
    "student_financial_services_form", "student_form", "student_refund_request",
    "student_verification_worksheet", "student_worksheet", "submission_form", "subscription_form",
    "table_space_request_form", "tax_collection_form", "tax_deferred_annuity_election_form",
    "tax_document", "tax_election_form", "tax_exemption_application", "tax_exemption_certificate",
    "tax_exemption_form", "tax_filing_statement", "tax_form", "tax_identification_form",
    "tax_return_form", "tax_return_transcript_request", "tax_transcript_request",
    "taxpayer_identification_form", "teacher_certification_authorization", "technical_data_sheet",
    "test_certificate", "thesis_proposal_approval_form", "third_party_access_request_form",
    "third_party_billing_statement", "time_sheet", "tort_claim_form", "transcript",
    "transcript_order_form", "transcript_release_form", "transcript_request",
    "transcript_request_form", "transfer_application", "transfer_information_request",
    "transfer_request", "transfer_request_form", "travel_authorization_form", "travel_form",
    "trusted_contact_authorization_form", "tuition_award_request",
    "tuition_discount_verification_form", "tuition_insurance_form", "tuition_waiver_affidavit",
    "tuition_waiver_form", "type_approval_certificate", "university_admission_application",
    "university_application_form", "university_complaint_form", "user_access_deletion_form",
    "vacation_request", "vaccination_form", "vaccine_records_request_form",
    "vehicle_request_form", "vendor_application", "vendor_authorization",
    "vendor_foreign_source_statement", "vendor_form", "vendor_information_request",
    "vendor_registration", "vendor_request_form", "verification_form",
    "verification_of_dependent", "verification_worksheet",
    "veterans_enrollment_certification_form", "volunteer_form", "volunteer_identification_form",
    "w2_form", "w2_transmittal", "w2g", "w4_form", "waiver_request",
    "webform_application_checklist", "wireless_device_request", "withdrawal_form",
    "withholding_certificate", "work_study_program_form", "worksheet",
]

# ---------------------------------------------------------------------------
# Local Gemma-4 via vLLM. The container runs on the host network, so the
# host's localhost is reachable directly; override with env vars if it moves.
# vLLM ignores the API key but the OpenAI client requires a non-empty value.
# ---------------------------------------------------------------------------
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1/")
_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"
_DEFAULT_OUTPUT = Path("/data/paperwerk/vqa_20260725_form_like.parquet")
_VARIANT = "form-like"

# Same unified schema as vqa_20260524_kvp10k.py so the parquets are mixable.
_PARQUET_SCHEMA = pa.schema(
    [
        ("images", pa.list_(pa.binary())),
        ("queries", pa.list_(pa.string())),
        (
            "answers",
            pa.list_(
                pa.struct(
                    [
                        ("query", pa.string()),
                        ("value", pa.string()),
                        ("bounding_box", pa.list_(pa.float64())),
                        ("index", pa.int32()),
                    ]
                )
            ),
        ),
        ("source", pa.string()),
        ("variant", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("split", pa.string()),
    ]
)

_FLUSH_EVERY = 32  # write to parquet every N datapoints to bound memory
_MAX_ATTEMPTS = 12  # per-slot template retries before giving up on that slot


def build_template_pool(min_fields: int) -> list[tuple[str, Path]]:
    """Resolve FORM_LIKE_CLASSES to concrete, usable (class, template) pairs."""
    env = Environment(autoescape=True)
    pool: list[tuple[str, Path]] = []
    missing: list[str] = []
    for cls in FORM_LIKE_CLASSES:
        class_dir = _TEMPLATES_ROOT / cls
        if not class_dir.is_dir():
            missing.append(cls)
            continue
        for tpl in sorted(class_dir.glob("*.html.j2")):
            if not tpl.with_suffix("").with_suffix(".json").exists():
                continue
            text = tpl.read_text()
            try:
                env.parse(text)
            except Exception:
                continue  # skip malformed-as-generated templates
            if len(discover_fields(text)) < min_fields:
                continue  # skip near-blank templates
            pool.append((cls, tpl))
    if missing:
        print(f"warning: {len(missing)} listed class(es) not found: {missing[:8]}"
              + (" ..." if len(missing) > 8 else ""))
    return pool


def _augment_plan(rng: random.Random) -> dict:
    return {
        "profile": rng.choice(PROFILES),
        "quality": round(rng.uniform(0.25, 0.9), 3),
        "geometric": rng.random() < 0.5,
        "geo_quality": round(rng.uniform(0.4, 0.9), 3),
    }


def _encode(img, image_format: str, jpeg_quality: int) -> bytes:
    buf = BytesIO()
    rgb = img.convert("RGB")
    if image_format == "jpeg":
        rgb.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        rgb.save(buf, format="PNG")
    return buf.getvalue()


def _render_to_datapoint(
    template_html: str,
    data: dict,
    plan: dict | None,
    *,
    seed: int,
    dpi: int,
    image_format: str,
    jpeg_quality: int,
) -> tuple[list[bytes], list]:
    """Sync CPU stage: render (+optional augment) → (page image bytes, fields)."""
    pdf, fields = render(template_html, data, seed=seed)
    if plan is not None:
        if plan["geometric"]:
            pdf, fields = augment_geometric(
                pdf, fields, quality=plan["geo_quality"], seed=seed
            )
        pdf, fields = augment(
            pdf, fields, profile=plan["profile"], quality=plan["quality"], seed=seed
        )
    pages = convert_from_bytes(pdf, dpi=dpi)
    images = [_encode(p, image_format, jpeg_quality) for p in pages]
    return images, fields


def _to_datapoint(
    images: list[bytes],
    fields: list,
    *,
    source: str,
    split: str,
) -> dict | None:
    """Assemble a unified VQA datapoint; None if nothing was filled."""
    n_pages = len(images)
    answers: list[dict] = []
    queries: list[str] = []
    for f in fields:
        if not f.value.strip():
            continue  # skip empty / signature (SVG) fields
        if not (0 <= f.page < n_pages):
            continue
        if f.name not in queries:
            queries.append(f.name)
        answers.append(
            {
                "query": f.name,
                "value": f.value,
                "bounding_box": [float(c) for c in f.bbox],
                "index": int(f.page),
            }
        )
    if not answers:
        return None
    return {
        "images": images,
        "queries": queries,
        "answers": answers,
        "source": source,
        "variant": _VARIANT,
        "page_start": 0,
        "page_end": n_pages - 1,
        "split": split,
    }


async def _build(args: argparse.Namespace) -> None:
    pool = build_template_pool(args.min_fields)
    if not pool:
        sys.exit("error: no usable templates resolved from FORM_LIKE_CLASSES")
    print(
        f"pool: {len(pool)} template(s) across "
        f"{len({c for c, _ in pool})} form-like class(es)"
    )

    rng = random.Random(args.seed)
    # Per-slot augmentation flags: exactly round(N * ratio) augmented, shuffled.
    n_aug = round(args.limit * args.augment_ratio)
    aug_flags = [True] * n_aug + [False] * (args.limit - n_aug)
    rng.shuffle(aug_flags)

    cpu_sem = asyncio.Semaphore(args.cpu_concurrency)

    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    llm = LLM(
        base_url=_VLLM_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=args.concurrency,
    )
    print(f"value generation via {llm!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(args.output), _PARQUET_SCHEMA)
    write_lock = asyncio.Lock()
    pending: list[dict] = []
    done = 0

    async def _flush(force: bool = False) -> None:
        nonlocal pending
        async with write_lock:
            if pending and (force or len(pending) >= _FLUSH_EVERY):
                batch, pending = pending, []
                table = pa.Table.from_pylist(batch, schema=_PARQUET_SCHEMA)
                await asyncio.to_thread(writer.write_table, table)

    async def _slot(slot: int) -> None:
        nonlocal done
        augmented = aug_flags[slot]
        slot_rng = random.Random(f"{args.seed}:{slot}")
        seed = args.seed + slot
        # Sample templates WITH replacement (a run of N docs typically far
        # exceeds the pool size); each pick gets fresh Gemma values + seed, so
        # reuses still differ. Each slot retries a bounded number of times to
        # skip templates that fail to render before giving up.
        for _ in range(_MAX_ATTEMPTS):
            cls, tpl_path = slot_rng.choice(pool)
            try:
                schema = discover_fields(tpl_path.read_text())
                template_html = tpl_path.read_text()
                if args.signatures:
                    template_html, sig_fields = inject_signatures(
                        template_html, schema, slot_rng
                    )
                else:
                    sig_fields = []
                synth_fields = [f for f in schema if f not in sig_fields]
                if not synth_fields:
                    continue
                data = await synthesize_values(llm, synth_fields)
                plan = _augment_plan(slot_rng) if augmented else None
                async with cpu_sem:
                    images, fields = await asyncio.to_thread(
                        _render_to_datapoint,
                        template_html, data, plan,
                        seed=seed, dpi=args.dpi,
                        image_format=args.image_format,
                        jpeg_quality=args.jpeg_quality,
                    )
            except Exception as e:  # noqa: BLE001 - retry with next template
                print(f"[slot {slot:05d}] {cls} failed ({e!r}); retrying")
                continue

            source = f"{cls}/{tpl_path.stem.replace('.html', '')}#{slot:05d}"
            dp = _to_datapoint(images, fields, source=source, split=args.split)
            if dp is None:
                print(f"[slot {slot:05d}] {cls} produced no answers; retrying")
                continue
            pending.append(dp)
            done += 1
            tag = (
                f"aug:{plan['profile']}" + ("+geo" if plan["geometric"] else "")
                if plan else "clean"
            )
            print(
                f"[slot {slot:05d}] ({done}/{args.limit}) {cls} "
                f"[{tag}, {len(dp['images'])}p, {len(dp['answers'])} qa]"
            )
            await _flush()
            return
        print(f"[slot {slot:05d}] gave up after {_MAX_ATTEMPTS} failed attempt(s)")

    try:
        await asyncio.gather(*(_slot(i) for i in range(args.limit)))
        await _flush(force=True)
    finally:
        writer.close()

    print(f"\ndone: wrote {done} datapoint(s) -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-n", "--limit", type=int, default=100,
                        help="Number of documents to generate (default: 100)")
    parser.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"Output parquet (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--augment-ratio", type=float, default=0.0,
                        help="Fraction of docs to scanner/photo-augment (default: 0.0)")
    sig = parser.add_mutually_exclusive_group()
    sig.add_argument("--signatures", dest="signatures", action="store_true",
                     help="Render signature fields as SVG scrawls (default)")
    sig.add_argument("--no-signatures", dest="signatures", action="store_false",
                     help="Leave signature fields as plain text values")
    parser.set_defaults(signatures=True)
    parser.add_argument("--split", default="train", help="Split label (default: train)")
    parser.add_argument("--seed", type=int, default=20260725, help="RNG seed")
    parser.add_argument("--dpi", type=int, default=150, help="Page raster DPI (default: 150)")
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png",
                        help="Encoding for page images (default: png)")
    parser.add_argument("--jpeg-quality", type=int, default=92,
                        help="JPEG quality when --image-format jpeg (default: 92)")
    parser.add_argument("--min-fields", type=int, default=4,
                        help="Skip templates with fewer discoverable fields (default: 4)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent Gemma requests (default: 10)")
    parser.add_argument("--cpu-concurrency", type=int, default=4,
                        help="Max concurrent render/augment jobs (default: 4)")
    args = parser.parse_args()
    if not 0.0 <= args.augment_ratio <= 1.0:
        sys.exit("error: --augment-ratio must be in [0, 1]")
    asyncio.run(_build(args))


if __name__ == "__main__":
    main()
