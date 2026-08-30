import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
PATHWAY_MAP_DIR = BASE_DIR / "result3" / "Pathway_map"
DIRECT_REACTOME_DIR = BASE_DIR / "result3" / "final_pathway_dirctly_reactome"
DESCRIPTION_DIR = BASE_DIR / "result3" / "descripition"
GENESET_CSV_PATH = BASE_DIR / "reactome_layer" / "pathway_gene_sets.csv"
OUTPUT_DIR = BASE_DIR / "result3" / "drug_geneset"


def load_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_pathway_name(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def normalize_gene_symbol(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9-]", "", text)


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        normalized = normalize_gene_symbol(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def split_genes(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []

    genes = []
    for piece in text.split(";"):
        gene = normalize_gene_symbol(piece)
        if gene:
            genes.append(gene)
    return dedupe_keep_order(genes)


def load_geneset_lookup(csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing geneset CSV: {csv_path}")

    genes_by_id: Dict[str, List[str]] = {}
    genes_by_name: Dict[str, List[str]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"PathwayID", "PathwayName", "Genes"}
        fieldnames = set(reader.fieldnames or [])
        missing = required_cols - fieldnames
        if missing:
            raise ValueError(f"Missing required columns in geneset CSV: {missing}")

        for row in reader:
            pathway_id = normalize_text(row.get("PathwayID"))
            pathway_name = normalize_pathway_name(row.get("PathwayName"))
            genes = split_genes(row.get("Genes"))

            if pathway_id and pathway_id not in genes_by_id:
                genes_by_id[pathway_id] = genes
            if pathway_name and pathway_name not in genes_by_name:
                genes_by_name[pathway_name] = genes

    return genes_by_id, genes_by_name


def extract_pathways_from_pathway_map_json(data: Dict[str, Any]) -> List[Dict[str, str]]:
    extracted: List[Dict[str, str]] = []

    for branch_name in ("pathway_moa", "gene"):
        branch_items = data.get(branch_name, [])
        if not isinstance(branch_items, list):
            continue

        for item in branch_items:
            if not isinstance(item, dict):
                continue
            pathway_id = normalize_text(item.get("PathwayID"))
            pathway_name = normalize_text(item.get("PathwayName"))
            if pathway_id or pathway_name:
                extracted.append({
                    "PathwayID": pathway_id,
                    "PathwayName": pathway_name,
                    "Source": "Pathway_map",
                })

    return extracted


def extract_pathways_from_direct_reactome_json(data: Dict[str, Any]) -> List[Dict[str, str]]:
    extracted: List[Dict[str, str]] = []

    for item in data.get("pathways", []):
        if not isinstance(item, dict):
            continue
        pathway_id = normalize_text(item.get("PathwayID"))
        pathway_name = normalize_text(item.get("PathwayName"))
        if pathway_id or pathway_name:
            extracted.append({
                "PathwayID": pathway_id,
                "PathwayName": pathway_name,
                "Source": "final_pathway_dirctly_reactome",
            })

    return extracted


def extract_final_gene_set_from_description(data: Dict[str, Any]) -> List[str]:
    gene_block = data.get("gene", {})
    if not isinstance(gene_block, dict):
        return []
    final_gene_set = gene_block.get("final_gene_set", [])
    if not isinstance(final_gene_set, list):
        return []
    return dedupe_keep_order([normalize_gene_symbol(item) for item in final_gene_set])


def deduplicate_pathways(pathways: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, str]] = []

    for item in pathways:
        key = (
            normalize_text(item.get("PathwayID")),
            normalize_pathway_name(item.get("PathwayName")),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def match_pathway_to_genes(
    pathway: Dict[str, str],
    genes_by_id: Dict[str, List[str]],
    genes_by_name: Dict[str, List[str]],
) -> List[str]:
    pathway_id = normalize_text(pathway.get("PathwayID"))
    pathway_name = normalize_pathway_name(pathway.get("PathwayName"))

    if pathway_id and pathway_id in genes_by_id:
        return genes_by_id[pathway_id]
    if pathway_name and pathway_name in genes_by_name:
        return genes_by_name[pathway_name]
    return []


def collect_drug_files() -> Dict[str, Dict[str, Path]]:
    drug_files: Dict[str, Dict[str, Path]] = {}

    for json_file in sorted(PATHWAY_MAP_DIR.glob("*.json")):
        drug_files.setdefault(json_file.stem, {})
        drug_files[json_file.stem]["pathway_map"] = json_file

        direct_path = DIRECT_REACTOME_DIR / json_file.name
        if direct_path.exists():
            drug_files[json_file.stem]["direct_reactome"] = direct_path

        description_path = DESCRIPTION_DIR / json_file.name
        if description_path.exists():
            drug_files[json_file.stem]["description"] = description_path

    return drug_files


def build_final_gene_set_for_drug(
    file_map: Dict[str, Path],
    genes_by_id: Dict[str, List[str]],
    genes_by_name: Dict[str, List[str]],
) -> Tuple[str, List[str]]:
    pathway_map_path = file_map.get("pathway_map")
    if pathway_map_path is None:
        raise ValueError("Missing pathway_map JSON for drug.")

    pathway_map_data = load_json(pathway_map_path)
    drug_name = normalize_text(pathway_map_data.get("drug_name")) or pathway_map_path.stem

    all_pathways = extract_pathways_from_pathway_map_json(pathway_map_data)

    direct_reactome_path = file_map.get("direct_reactome")
    if direct_reactome_path is not None:
        direct_data = load_json(direct_reactome_path)
        all_pathways.extend(extract_pathways_from_direct_reactome_json(direct_data))

    all_pathways = deduplicate_pathways(all_pathways)

    merged_genes: List[str] = []
    for pathway in all_pathways:
        merged_genes.extend(match_pathway_to_genes(pathway, genes_by_id, genes_by_name))

    description_path = file_map.get("description")
    if description_path is not None:
        description_data = load_json(description_path)
        merged_genes.extend(extract_final_gene_set_from_description(description_data))

    final_gene_set = dedupe_keep_order(merged_genes)
    return drug_name, final_gene_set


def save_drug_gene_csv(drug_file_stem: str, drug_name: str, genes: List[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{drug_file_stem}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["DrugName", "Gene"])
        writer.writeheader()
        for gene in genes:
            writer.writerow({
                "DrugName": drug_name,
                "Gene": gene,
            })
    return out_path


def batch_process_all_drugs(
    geneset_csv_path: Path = GENESET_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    genes_by_id, genes_by_name = load_geneset_lookup(geneset_csv_path)
    drug_files = collect_drug_files()

    if not drug_files:
        raise FileNotFoundError(f"No JSON files found in {PATHWAY_MAP_DIR}")

    total = len(drug_files)
    print(f"Found {total} drugs in Pathway_map.")

    for idx, (drug_file_stem, file_map) in enumerate(sorted(drug_files.items()), start=1):
        print(f"[{idx}/{total}] Start: {drug_file_stem}")
        try:
            drug_name, final_gene_set = build_final_gene_set_for_drug(
                file_map=file_map,
                genes_by_id=genes_by_id,
                genes_by_name=genes_by_name,
            )
            out_path = save_drug_gene_csv(
                drug_file_stem=drug_file_stem,
                drug_name=drug_name,
                genes=final_gene_set,
                output_dir=output_dir,
            )
            print(f"[{idx}/{total}] Done: {drug_file_stem} | genes={len(final_gene_set)} | saved={out_path.name}")
        except Exception as exc:
            print(f"[{idx}/{total}] Failed: {drug_file_stem} | {exc}")


def main() -> None:
    batch_process_all_drugs()


if __name__ == "__main__":
    main()
