from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = BASE_DIR / "result3" / "drug_geneset"
DEFAULT_DESCRIPTION_DIR = BASE_DIR / "result3" / "descripition"
DEFAULT_OUTPUT_DIR = BASE_DIR / "result3" / "drug_geneset_checked"
DEFAULT_OPENAI_API_KEY_FILE = BASE_DIR / "openai_api_key.txt"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_GENECOUNT_THRESHOLD = 100
DEFAULT_MAX_OUTPUT_TOKENS = 2000
DEFAULT_MAX_RETRIES = 2

_OPENAI_CLIENT = None


def load_api_key(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", normalize_text(value))
    return value.strip("_") or "drug"


def normalize_gene_symbol(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9-]", "", text)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        gene = normalize_gene_symbol(item)
        if gene and gene not in seen:
            seen.add(gene)
            result.append(gene)
    return result


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model output is not valid JSON:\n{text}")
        return json.loads(match.group(0))


def get_openai_client():
    global _OPENAI_CLIENT

    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    api_key = load_api_key(DEFAULT_OPENAI_API_KEY_FILE) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OpenAI API key is missing. Put it in {DEFAULT_OPENAI_API_KEY_FILE} or set OPENAI_API_KEY."
        )

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def read_gene_csv(csv_path: Path) -> Tuple[str, List[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        genes: List[str] = []
        drug_name = ""

        for row in reader:
            if not drug_name:
                drug_name = normalize_text(row.get("DrugName"))
            gene = normalize_gene_symbol(row.get("Gene"))
            if gene:
                genes.append(gene)

    if not drug_name:
        drug_name = csv_path.stem
    return drug_name, dedupe_keep_order(genes)


def load_description_context(drug_name: str, csv_stem: str, description_dir: Path) -> Dict[str, Any]:
    candidate_paths = [
        description_dir / f"{csv_stem}.json",
        description_dir / f"{slugify(drug_name)}.json",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        pathway_moa = data.get("pathway_moa", {})
        gene_block = data.get("gene", {})
        return {
            "path": str(path),
            "final_moa_artifact": pathway_moa.get("final_moa_artifact", "") if isinstance(pathway_moa, dict) else "",
            "final_gene_artifact": gene_block.get("final_gene_artifact", "") if isinstance(gene_block, dict) else "",
            "final_gene_set": dedupe_keep_order(gene_block.get("final_gene_set", [])) if isinstance(gene_block, dict) else [],
        }

    return {
        "path": "",
        "final_moa_artifact": "",
        "final_gene_artifact": "",
        "final_gene_set": [],
    }


def build_system_prompt(threshold: int) -> str:
    return f"""
You are a biomedical gene-set curation assistant.

Task:
1. Review one drug's current gene set.
2. If the gene set is clearly too small for the drug's mechanism and response context, supplement only important missing genes.
3. If the current set is already adequate, keep it unchanged.

Threshold guidance:
- This review is only triggered when GeneCount is below {threshold}.
- Do not try to force the final count above {threshold}.

What counts as important missing genes:
- direct drug targets
- core pathway mediators tightly linked to the mechanism
- major transport, metabolism, resistance, apoptosis, checkpoint, or signaling genes if strongly relevant

What not to do:
- do not invent genes
- do not add broad filler genes
- do not remove existing genes
- do not add weakly related genes
- keep additions conservative, usually much fewer than 40 genes

Output rules:
- Return JSON only.
- No markdown.
- Use HGNC-style gene symbols.
- Use exactly this schema:
{{
  "drug_name": "string",
  "decision": "keep" or "supplement",
  "rationale": "one short sentence",
  "added_genes": ["GENE1", "GENE2"],
  "updated_genes": ["FINAL", "GENE", "SET"],
  "confidence": "high" or "medium" or "low"
}}
""".strip()


def get_output_schema() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "drug_gene_set_deficiency_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "drug_name": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": ["keep", "supplement"],
                },
                "rationale": {"type": "string"},
                "added_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "updated_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": [
                "drug_name",
                "decision",
                "rationale",
                "added_genes",
                "updated_genes",
                "confidence",
            ],
        },
    }


def build_user_payload(
    drug_name: str,
    genes: List[str],
    threshold: int,
    description_context: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task": "supplement_low_gene_count_drug_gene_set",
        "drug_name": drug_name,
        "threshold": threshold,
        "gene_count": len(genes),
        "current_genes": genes,
        "description_context": {
            "final_moa_artifact": description_context.get("final_moa_artifact", ""),
            "final_gene_artifact": description_context.get("final_gene_artifact", ""),
            "final_gene_set": description_context.get("final_gene_set", []),
        },
        "instructions": [
            "Keep all existing genes.",
            "Supplement only if clearly important genes are missing.",
            "Return a conservative, mechanism-grounded final gene set.",
        ],
    }


def run_deficiency_review(
    drug_name: str,
    genes: List[str],
    threshold: int,
    description_context: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    client = get_openai_client()
    input_items = [
        {"role": "system", "content": build_system_prompt(threshold)},
        {
            "role": "user",
            "content": safe_json_dumps(
                build_user_payload(
                    drug_name=drug_name,
                    genes=genes,
                    threshold=threshold,
                    description_context=description_context,
                )
            ),
        },
    ]

    last_error = None
    final_text = ""
    response = None

    for attempt in range(DEFAULT_MAX_RETRIES):
        attempt_input = input_items
        if attempt > 0:
            attempt_input = input_items + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid or truncated. Retry with much shorter rationale and fewer additions."
                    ),
                }
            ]

        response = client.responses.create(
            model=model,
            input=attempt_input,
            text={"format": get_output_schema()},
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )

        final_text = (response.output_text or "").strip()
        try:
            return {
                "raw_output_text": final_text,
                "actor_result": parse_json_response(final_text),
                "response_id": response.id,
                "actor_model": model,
            }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to parse model JSON after {DEFAULT_MAX_RETRIES} attempts: {last_error}")


def merge_final_genes(original_genes: List[str], actor_result: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    original_set = set(original_genes)
    final_genes = list(original_genes)

    model_updated = dedupe_keep_order(actor_result.get("updated_genes", []))
    model_added = dedupe_keep_order(actor_result.get("added_genes", []))

    for gene in model_updated + model_added:
        if gene and gene not in final_genes:
            final_genes.append(gene)

    final_genes = dedupe_keep_order(final_genes)
    added_genes = [gene for gene in final_genes if gene not in original_set]
    return final_genes, added_genes


def write_output_csv(
    output_path: Path,
    drug_name: str,
    final_genes: List[str],
    original_genes: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_set = set(original_genes)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["DrugName", "Gene", "GeneSource"])
        writer.writeheader()
        for gene in final_genes:
            writer.writerow({
                "DrugName": drug_name,
                "Gene": gene,
                "GeneSource": "original" if gene in original_set else "supplemented",
            })


def save_review_json(
    output_path: Path,
    drug_name: str,
    original_genes: List[str],
    final_genes: List[str],
    added_genes: List[str],
    threshold: int,
    review_payload: Dict[str, Any],
    description_context: Dict[str, Any],
) -> None:
    actor_result = review_payload.get("actor_result", {}) or {}
    payload = {
        "drug_name": drug_name,
        "threshold": threshold,
        "original_gene_count": len(original_genes),
        "final_gene_count": len(final_genes),
        "added_gene_count": len(added_genes),
        "added_genes": added_genes,
        "decision": normalize_text(actor_result.get("decision", "")),
        "confidence": normalize_text(actor_result.get("confidence", "")),
        "rationale": normalize_text(actor_result.get("rationale", "")),
        "description_path": description_context.get("path", ""),
        "raw_model_output": review_payload.get("raw_output_text", ""),
    }
    output_path.write_text(safe_json_dumps(payload), encoding="utf-8")


def process_one_csv(
    csv_path: Path,
    description_dir: Path,
    output_dir: Path,
    threshold: int,
    model: str,
) -> Dict[str, Any]:
    drug_name, original_genes = read_gene_csv(csv_path)
    description_context = load_description_context(drug_name, csv_path.stem, description_dir)

    if len(original_genes) >= threshold:
        final_genes = list(original_genes)
        added_genes: List[str] = []
        review_payload = {
            "actor_model": "",
            "raw_output_text": "",
            "actor_result": {
                "drug_name": drug_name,
                "decision": "keep",
                "rationale": f"GeneCount already meets threshold {threshold}.",
                "added_genes": [],
                "updated_genes": final_genes,
                "confidence": "high",
            },
        }
    else:
        review_payload = run_deficiency_review(
            drug_name=drug_name,
            genes=original_genes,
            threshold=threshold,
            description_context=description_context,
            model=model,
        )
        final_genes, added_genes = merge_final_genes(
            original_genes=original_genes,
            actor_result=review_payload.get("actor_result", {}) or {},
        )

    output_csv = output_dir / csv_path.name
    output_json = output_dir / f"{csv_path.stem}_review.json"
    write_output_csv(
        output_path=output_csv,
        drug_name=drug_name,
        final_genes=final_genes,
        original_genes=original_genes,
    )
    save_review_json(
        output_path=output_json,
        drug_name=drug_name,
        original_genes=original_genes,
        final_genes=final_genes,
        added_genes=added_genes,
        threshold=threshold,
        review_payload=review_payload,
        description_context=description_context,
    )

    return {
        "drug_name": drug_name,
        "original_gene_count": len(original_genes),
        "final_gene_count": len(final_genes),
        "added_gene_count": len(added_genes),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
        "review_decision": normalize_text(review_payload.get("actor_result", {}).get("decision", "")),
    }


def process_all_csvs(
    input_dir: Path,
    description_dir: Path,
    output_dir: Path,
    threshold: int,
    model: str,
    limit: Optional[int] = None,
    only_drug: Optional[str] = None,
) -> None:
    csv_files = sorted(input_dir.glob("*.csv"))
    if only_drug:
        target = normalize_text(only_drug).lower()
        csv_files = [path for path in csv_files if path.stem.lower() == target or slugify(path.stem).lower() == slugify(target).lower()]
    if limit is not None:
        csv_files = csv_files[:limit]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    total = len(csv_files)
    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{total}] Start: {csv_path.stem}")
        try:
            result = process_one_csv(
                csv_path=csv_path,
                description_dir=description_dir,
                output_dir=output_dir,
                threshold=threshold,
                model=model,
            )
            summary_rows.append(result)
            print(
                f"[{idx}/{total}] Done: {result['drug_name']} | "
                f"original={result['original_gene_count']} | "
                f"final={result['final_gene_count']} | "
                f"added={result['added_gene_count']}"
            )
        except Exception as exc:
            summary_rows.append({
                "drug_name": csv_path.stem,
                "original_gene_count": "",
                "final_gene_count": "",
                "added_gene_count": "",
                "output_csv": "",
                "output_json": "",
                "review_decision": "error",
                "error": str(exc),
            })
            print(f"[{idx}/{total}] Failed: {csv_path.stem} | {exc}")

    summary_path = output_dir / "review_summary.csv"
    fieldnames = [
        "drug_name",
        "original_gene_count",
        "final_gene_count",
        "added_gene_count",
        "output_csv",
        "output_json",
        "review_decision",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("\nAll done.")
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check result3 drug gene sets and use OpenAI to supplement important missing genes for low-count drugs."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--description_dir", type=Path, default=DEFAULT_DESCRIPTION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=int, default=DEFAULT_GENECOUNT_THRESHOLD)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drug_name", type=str, default=None)
    args = parser.parse_args()

    process_all_csvs(
        input_dir=args.input_dir,
        description_dir=args.description_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        model=args.model,
        limit=args.limit,
        only_drug=args.drug_name,
    )


if __name__ == "__main__":
    main()
