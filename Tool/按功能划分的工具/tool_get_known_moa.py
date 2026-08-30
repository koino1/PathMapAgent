from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import xml.etree.ElementTree as ET

from Tool.common import compact_records, dedupe_records, load_csv_rows, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRUGBANK_XML_PATH = PROJECT_ROOT / "local_data" / "DrugBank_fulldata.xml"
PRIMEKG_SMILES_PATH = PROJECT_ROOT / "CLADD" / "data" / "PrimeKG" / "primekg_drug_smiles_step2.csv"
DB_NS = {"db": "http://www.drugbank.ca"}
DRUGBANK_DRUG_TAG = f"{{{DB_NS['db']}}}drug"
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_TIMEOUT = 20
VALID_SOURCES = ("drugbank", "gdsc", "prism", "opentargets")

_gdsc_rows = load_csv_rows("GDSC_drugs.csv")
_prism_rows = load_csv_rows("Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv")


def _split_items(text: Any) -> List[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    raw = raw.replace(";", ",").replace("|", ",")
    items = [item.strip() for item in raw.split(",") if item.strip()]

    seen = set()
    result: List[str] = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


@lru_cache(maxsize=1)
def _load_primekg_smiles_rows() -> List[Dict[str, Any]]:
    if not PRIMEKG_SMILES_PATH.exists():
        return []

    with PRIMEKG_SMILES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            cleaned: Dict[str, Any] = {}
            for key, value in row.items():
                clean_key = str(key).replace("\ufeff", "").strip()
                cleaned[clean_key] = value.strip() if isinstance(value, str) else value
            rows.append(cleaned)
        return rows


def _lookup_primekg_smiles(drug_name: str) -> str:
    query = normalize_text(drug_name)
    if not query:
        return ""

    exact_match = ""
    partial_match = ""

    for row in _load_primekg_smiles_rows():
        name = normalize_text(row.get("Drugbank Name", ""))
        if not name:
            continue

        smiles = str(row.get("SMILES", "") or "").strip()
        if not smiles:
            continue

        if query == name:
            exact_match = smiles
            break
        if not partial_match and query in name:
            partial_match = smiles

    return exact_match or partial_match


def _lookup_gdsc_smiles(drug_name: str) -> str:
    query = normalize_text(drug_name)
    if not query:
        return ""

    exact_match = ""
    partial_match = ""

    for row in _gdsc_rows:
        name = normalize_text(row.get("DRUG_NAME", ""))
        if not name:
            continue

        smiles = str(row.get("SMILES", "") or "").strip()
        if not smiles:
            continue

        if query == name:
            exact_match = smiles
            break
        if not partial_match and query in name:
            partial_match = smiles

    return exact_match or partial_match


def _lookup_preferred_smiles(drug_name: str) -> str:
    return _lookup_gdsc_smiles(drug_name) or _lookup_primekg_smiles(drug_name)


def _format_open_targets_moa(row: Dict[str, Any]) -> str:
    mechanism = str(row.get("mechanismOfAction") or "").strip()
    action_type = str(row.get("actionType") or "").strip()
    target_name = str(row.get("targetName") or "").strip()

    if mechanism:
        return mechanism
    if action_type and target_name:
        return f"{action_type} of {target_name}"
    return ""


def _post_open_targets_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        OPEN_TARGETS_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=OPEN_TARGETS_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("errors") or []
    if errors:
        raise ValueError(f"Open Targets GraphQL errors: {errors}")
    return payload.get("data") or {}


def _text(node: Optional[ET.Element]) -> str:
    return (node.text or "").strip() if node is not None else ""


def _texts(parent: Optional[ET.Element], xpath: str) -> List[str]:
    if parent is None:
        return []
    return [_text(item) for item in parent.findall(xpath, DB_NS)]


def _first(parent: Optional[ET.Element], xpath: str) -> str:
    if parent is None:
        return ""
    return _text(parent.find(xpath, DB_NS))


def _sorted_unique_nonempty(values: List[str]) -> List[str]:
    return sorted({value for value in values if value != ""})


def _ensure_drugbank_xml_exists() -> None:
    if not DRUGBANK_XML_PATH.exists():
        raise FileNotFoundError(
            f"DrugBank XML file not found: {DRUGBANK_XML_PATH}. "
            "Please update DRUGBANK_XML_PATH to your local XML file."
        )


def _iter_drugbank_drugs():
    _ensure_drugbank_xml_exists()
    context = ET.iterparse(DRUGBANK_XML_PATH, events=("end",))
    for _, elem in context:
        if elem.tag == DRUGBANK_DRUG_TAG:
            yield elem
            elem.clear()


def _extract_drugbank_ids(drug: ET.Element) -> List[str]:
    return _sorted_unique_nonempty(_texts(drug, "db:drugbank-id"))


def _extract_synonyms(drug: ET.Element) -> List[str]:
    return _sorted_unique_nonempty(_texts(drug, "db:synonyms/db:synonym"))


def _extract_classification(drug: ET.Element) -> Dict[str, Any]:
    categories = _texts(drug, "db:categories/db:category/db:category")
    return {
        "description": _first(drug, "db:classification/db:description"),
        "direct_parent": _first(drug, "db:classification/db:direct-parent"),
        "categories": _sorted_unique_nonempty(categories),
    }


def _extract_pathways(drug: ET.Element) -> List[Dict[str, Any]]:
    pathways: List[Dict[str, Any]] = []
    for pathway in drug.findall("db:pathways/db:pathway", DB_NS):
        pathways.append(
            {
                "pathway_name": _first(pathway, "db:name"),
                "category": _first(pathway, "db:category"),
            }
        )
    return compact_records(pathways)


def _extract_drugbank_moa_record(drug: ET.Element) -> Dict[str, Any]:
    return compact_records(
        [
            {
                "source_of_evidence": "DrugBank",
                "drugbank_id": _extract_drugbank_ids(drug),
                "name": _first(drug, "db:name"),
                "synonyms": _extract_synonyms(drug),
                "description": _first(drug, "db:description"),
                "indication": _first(drug, "db:indication"),
                "mechanism_of_action": _first(drug, "db:mechanism-of-action"),
                "pharmacodynamics": _first(drug, "db:pharmacodynamics"),
                "metabolism": _first(drug, "db:metabolism"),
                "classification": _extract_classification(drug),
                "pathways": _extract_pathways(drug),
            }
        ]
    )[0]


def _dedupe_drugbank_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for record in records:
        key = (tuple(record.get("drugbank_id", [])), record.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _search_drugbank_moa(drug_name: str, max_results: int) -> Dict[str, Any]:
    try:
        query = normalize_text(drug_name)
        _ensure_drugbank_xml_exists()
    except FileNotFoundError as exc:
        return {
            "source": "DrugBank",
            "num_matches": 0,
            "matches": [],
            "message": str(exc),
        }
    except ET.ParseError as exc:
        return {
            "source": "DrugBank",
            "num_matches": 0,
            "matches": [],
            "message": f"Failed to parse DrugBank XML: {exc}",
        }

    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    try:
        for drug in _iter_drugbank_drugs():
            main_name = normalize_text(_first(drug, "db:name"))
            ids = [normalize_text(item) for item in _extract_drugbank_ids(drug)]
            synonyms = [normalize_text(item) for item in _extract_synonyms(drug)]

            is_exact = query == main_name or query in ids or query in synonyms
            is_partial = (
                query in main_name
                or any(query in dbid for dbid in ids)
                or any(query in synonym for synonym in synonyms)
            )
            if not is_exact and not is_partial:
                continue

            record = _extract_drugbank_moa_record(drug)
            if is_exact:
                exact_matches.append(record)
                if len(exact_matches) >= max_results:
                    break
            else:
                partial_matches.append(record)
    except ET.ParseError as exc:
        return {
            "source": "DrugBank",
            "num_matches": 0,
            "matches": [],
            "message": f"Failed to parse DrugBank XML: {exc}",
        }

    matches = _dedupe_drugbank_records(exact_matches or partial_matches)[:max_results]
    return {
        "source": "DrugBank",
        "num_matches": len(matches),
        "matches": matches,
        "message": "Matched successfully." if matches else "No matched drug found in DrugBank XML.",
    }


def _search_gdsc_moa(drug_name: str, max_results: int) -> Dict[str, Any]:
    query = normalize_text(drug_name)
    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    for row in _gdsc_rows:
        name = normalize_text(row.get("DRUG_NAME", ""))
        synonyms = normalize_text(row.get("DRUG_SYNONYMS", ""))
        record = {
            "drug_id": row.get("DRUG_ID"),
            "drug_name": row.get("DRUG_NAME"),
            "putative_target": row.get("PUTATIVE_TARGET"),
            "pathway_name": row.get("PATHWAY_NAME"),
            "drug_synonyms": row.get("DRUG_SYNONYMS"),
            "hgcn_targets": row.get("HGCN_TARGETS"),
        }

        synonym_set = {item.strip() for item in synonyms.split(",") if item.strip()}

        if name == query or query in synonym_set:
            exact_matches.append(record)
        elif query in name or (synonyms and query in synonyms):
            partial_matches.append(record)

    matches = exact_matches or partial_matches
    matches = dedupe_records(matches, keys=("drug_id", "drug_name", "putative_target", "pathway_name"))
    matches = compact_records(matches[:max_results])

    pathway_candidates: List[str] = []
    for match in matches:
        pathway_candidates.extend(_split_items(match.get("pathway_name")))

    return {
        "source": "GDSC",
        "num_matches": len(matches),
        "pathway_candidates": _unique_keep_order(pathway_candidates),
        "matches": matches,
        "message": "Matched successfully." if matches else "No matched drug found in GDSC file.",
    }


def _search_prism_moa(drug_name: str, max_results: int) -> Dict[str, Any]:
    query = normalize_text(drug_name)
    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    for row in _prism_rows:
        name = normalize_text(row.get("Drug.Name", ""))
        synonyms = normalize_text(row.get("Synonyms", ""))
        moa = normalize_text(row.get("MOA", ""))
        target = normalize_text(row.get("repurposing_target", ""))
        record = {
            "drug_name": row.get("Drug.Name"),
            "screen": row.get("screen"),
            "moa": row.get("MOA"),
            "repurposing_target": row.get("repurposing_target"),
            "ids": row.get("IDs"),
            "synonyms": row.get("Synonyms"),
        }

        synonym_set = {item.strip() for item in synonyms.split(",") if item.strip()}

        if name == query or query in synonym_set:
            exact_matches.append(record)
        elif query in name or (synonyms and query in synonyms) or query in moa or query in target:
            partial_matches.append(record)

    matches = exact_matches or partial_matches
    matches = dedupe_records(matches, keys=("drug_name", "screen", "dose", "ids"))
    matches = compact_records(matches[:max_results])

    moa_candidates: List[str] = []
    for match in matches:
        moa_candidates.extend(_split_items(match.get("moa")))

    return {
        "source": "PRISM",
        "num_matches": len(matches),
        "moa_candidates": _unique_keep_order(moa_candidates),
        "matches": matches,
        "message": "Matched successfully." if matches else "No matched compound found in PRISM file.",
    }


def _search_open_targets_moa(drug_name: str, max_results: int) -> Dict[str, Any]:
    search_query = """
    query SearchDrug($queryString: String!, $size: Int!) {
      search(queryString: $queryString, entityNames: ["drug"], page: {index: 0, size: $size}) {
        hits {
          id
          name
          description
        }
      }
    }
    """
    drug_query = """
    query DrugMoA($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
        mechanismsOfAction {
          rows {
            mechanismOfAction
            actionType
            targetName
          }
        }
      }
    }
    """

    try:
        search_data = _post_open_targets_graphql(
            query=search_query,
            variables={"queryString": drug_name, "size": max_results},
        )
    except (requests.RequestException, ValueError) as exc:
        return {
            "source": "OpenTargets",
            "num_matches": 0,
            "moa_candidates": [],
            "matches": [],
            "message": f"Open Targets request failed: {exc}",
        }

    hits = ((search_data.get("search") or {}).get("hits") or [])[:max_results]
    matches: List[Dict[str, Any]] = []
    moa_candidates: List[str] = []

    query_norm = normalize_text(drug_name)
    exact_hits = [
        hit
        for hit in hits
        if normalize_text(hit.get("name")) == query_norm
    ]

    for hit in exact_hits[:max_results]:
        chembl_id = hit.get("id")
        if not chembl_id:
            continue
        try:
            drug_data = _post_open_targets_graphql(
                query=drug_query,
                variables={"chemblId": chembl_id},
            )
        except (requests.RequestException, ValueError):
            continue

        drug_record = drug_data.get("drug") or {}
        rows = ((drug_record.get("mechanismsOfAction") or {}).get("rows") or [])
        row_candidates = _unique_keep_order(
            [
                candidate
                for candidate in (_format_open_targets_moa(row) for row in rows)
                if candidate
            ]
        )
        moa_candidates.extend(row_candidates)
        matches.append(
            compact_records(
                [
                    {
                        "chembl_id": chembl_id,
                        "drug_name": drug_record.get("name") or hit.get("name"),
                        "description": hit.get("description"),
                        "moa_candidates": row_candidates,
                    }
                ]
            )[0]
        )

    matches = compact_records(matches)
    return {
        "source": "OpenTargets",
        "num_matches": len(matches),
        "moa_candidates": _unique_keep_order(moa_candidates),
        "matches": matches,
        "message": "Matched successfully." if matches else "No matched drug found in Open Targets.",
    }


def get_known_moa(drug_name: str, max_results_per_source: int = 5) -> Dict[str, Any]:
    """
    Given a drug name, return its PrimeKG SMILES together with merged known
    MOA and pathway evidence from DrugBank, GDSC, PRISM, and Open Targets.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        return {
            "drug_name": drug_name,
            "smiles": "",
            "known_moa": [],
        }

    results_by_source: Dict[str, Dict[str, Any]] = {}

    for source_name in VALID_SOURCES:
        if source_name == "drugbank":
            result = _search_drugbank_moa(drug_name=drug_name, max_results=max_results_per_source)
        elif source_name == "gdsc":
            result = _search_gdsc_moa(drug_name=drug_name, max_results=max_results_per_source)
        elif source_name == "prism":
            result = _search_prism_moa(drug_name=drug_name, max_results=max_results_per_source)
        else:
            result = _search_open_targets_moa(drug_name=drug_name, max_results=max_results_per_source)

        results_by_source[source_name] = result

    known_moa_candidates: List[str] = []
    pathway_candidates: List[str] = []

    for match in results_by_source.get("drugbank", {}).get("matches", []):
        mechanism = match.get("mechanism_of_action")
        if mechanism:
            known_moa_candidates.append(mechanism)
        for pathway in match.get("pathways", []):
            pathway_name = pathway.get("pathway_name")
            if pathway_name:
                pathway_candidates.append(pathway_name)

    known_moa_candidates.extend(results_by_source.get("prism", {}).get("moa_candidates", []))
    known_moa_candidates.extend(results_by_source.get("opentargets", {}).get("moa_candidates", []))
    pathway_candidates.extend(results_by_source.get("gdsc", {}).get("pathway_candidates", []))

    known_moa = _unique_keep_order(known_moa_candidates + pathway_candidates)

    return {
        "drug_name": drug_name,
        "smiles": _lookup_preferred_smiles(drug_name),
        "known_moa": known_moa,
    }


get_known_moa_doc = {
    "type": "function",
    "name": "get_known_moa",
    "description": (
        "Given a drug name, return its SMILES from the bundled "
        "CLADD/data/PrimeKG/primekg_drug_smiles_step2.csv file "
        "and merged known mechanism/pathway evidence aggregated from DrugBank, GDSC, PRISM, "
        "and Open Targets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "A single drug name to search.",
            },
            "max_results_per_source": {
                "type": "integer",
                "description": "Maximum number of matched records to return from each source. Default is 5.",
            },
        },
        "required": ["drug_name"],
    },
}
