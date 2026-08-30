from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[2]
CLADD_DIR = BASE_DIR / "CLADD"

if str(CLADD_DIR) not in sys.path:
    sys.path.insert(0, str(CLADD_DIR))

from run_kg_reports import generate_kg_reports  # type: ignore  # noqa: E402


def _build_error_payload(drug_name: str, smiles: str, exc: Exception) -> Dict[str, Any]:
    return {
        "drug_name": drug_name,
        "query_info": {},
        "anchor_drug": {},
        "related_gene_set": [],
        "drugrel_report": "",
        "biorel_report": "",
        "message": str(exc),
    }


def _normalize_query_inputs(
    drug_name: Optional[str] = None,
    smiles: Optional[str] = None,
) -> Dict[str, str]:
    normalized_drug_name = str(drug_name or "").strip()
    normalized_smiles = str(smiles or "").strip()

    if normalized_drug_name and normalized_smiles:
        raise ValueError("Please provide either drug_name or smiles, not both.")
    if not normalized_drug_name and not normalized_smiles:
        raise ValueError("Please provide either drug_name or smiles.")

    return {
        "drug_name": normalized_drug_name,
        "smiles": normalized_smiles,
    }


def get_cladd_kg_reports(
    drug_name: Optional[str] = None,
    smiles: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Given either a drug name or a SMILES string, return both DrugRel and
    BioRel reports plus the supporting CLADD retrieval context.
    """
    try:
        query = _normalize_query_inputs(drug_name=drug_name, smiles=smiles)
    except Exception as exc:
        return _build_error_payload(str(drug_name or "").strip(), str(smiles or "").strip(), exc)

    try:
        result = generate_kg_reports(
            drug_name=query["drug_name"] or None,
            smiles=query["smiles"] or None,
            model="gpt-4o-mini",
            temperature=0.0,
            max_triplets=5,
            device=0,
        )
    except Exception as exc:
        return _build_error_payload(query["drug_name"], query["smiles"], exc)

    return {
        "drug_name": query["drug_name"],
        "query_info": result.get("query_info", {}),
        "anchor_drug": result.get("anchor_drug", {}),
        "related_gene_set": result.get("related_gene_set", []),
        "drugrel_report": result.get("drugrel_report", ""),
        "biorel_report": result.get("biorel_report", ""),
        "message": (
            "Matched successfully."
            if (result.get("drugrel_report") or result.get("biorel_report"))
            else "Empty CLADD reports."
        ),
    }


get_cladd_kg_reports_doc = {
    "type": "function",
    "name": "get_cladd_kg_reports",
    "description": (
        "Given either a drug name or a SMILES string, use CLADD/PrimeKG to find the most similar drug "
        "that exists in the KG as the anchor drug, summarize the KG evidence, and return both DrugRel "
        "and BioRel reports. Provide exactly one of `drug_name` or `smiles`. The result includes "
        "query_info, anchor_drug, related_gene_set, drugrel_report, and biorel_report."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Drug name to query in CLADD/PrimeKG. Use this or smiles.",
            },
            "smiles": {
                "type": "string",
                "description": "SMILES string to query in CLADD/PrimeKG. Use this or drug_name.",
            },
        },
        "required": [],
    },
}
