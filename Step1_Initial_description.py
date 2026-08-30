from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None

from Tool.tool_geneagent_geneset import get_geneagent_geneset_info, get_geneagent_geneset_info_doc
from Tool.按功能划分的工具.tool_cladd_kg import get_cladd_kg_reports, get_cladd_kg_reports_doc
from Tool.按功能划分的工具.tool_cladd_molt5 import (
    get_cladd_molt5_description,
    get_cladd_molt5_description_doc,
)
from Tool.按功能划分的工具.tool_get_known_gene import get_known_gene, get_known_gene_doc
from Tool.按功能划分的工具.tool_get_known_moa import get_known_moa, get_known_moa_doc
from Tool.按功能划分的工具.tool_pubmed_articles import get_pubmed_articles, get_pubmed_articles_doc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DRUG_FILE = BASE_DIR / "local_data" / "GDSC_drugs.csv"
DEFAULT_RESULT_DIR = BASE_DIR / "result3" / "descripition"
DEFAULT_OPENAI_API_KEY_FILE = BASE_DIR / "openai_api_key.txt"
DEFAULT_ANTHROPIC_API_KEY_FILE = BASE_DIR / "claude_api_key.txt"

DEFAULT_ACTOR_MODEL = "gpt-5-mini"
DEFAULT_CRITIC_MODEL = "claude-haiku-4-5-20251001"
MAX_BRANCH_ROUNDS = 3
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_RECENT_CRITIC_HISTORY = 2
TEMPERATURE_ACTOR = 0
MAX_OUTPUT_TOKENS_ACTOR = 2000
MAX_OUTPUT_TOKENS_CRITIC = 500
MAX_PREVIOUS_LIST_ITEMS = 20
SUPPORTED_INPUT_TYPES = {"drug_name", "smiles"}
_CLADD_KG_CACHE: Dict[str, Dict[str, Any]] = {}


def load_api_key(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"Could not load API key from {path}: {exc}")
        return None


OPENAI_API_KEY = load_api_key(DEFAULT_OPENAI_API_KEY_FILE) or os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = load_api_key(DEFAULT_ANTHROPIC_API_KEY_FILE) or os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        f"OpenAI API key is missing. Put it in {DEFAULT_OPENAI_API_KEY_FILE} or set OPENAI_API_KEY."
    )

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        f"Anthropic API key is missing. Put it in {DEFAULT_ANTHROPIC_API_KEY_FILE} or set ANTHROPIC_API_KEY."
    )

if Anthropic is None:
    raise RuntimeError("anthropic package is not installed. Please install it first, e.g. pip install anthropic")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return value.strip("_") or "drug"


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model output is not valid JSON:\n{text}")
        return json.loads(match.group(0))


def _normalize_input_type(input_type: str) -> str:     #
    normalized = str(input_type or "").strip().lower()
    if normalized not in SUPPORTED_INPUT_TYPES:
        raise ValueError(f"Unsupported input_type: {input_type}. Expected one of {sorted(SUPPORTED_INPUT_TYPES)}.")
    return normalized


def _first_nonempty_value(row: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def read_input_values(csv_path: Path, input_type: str = "drug_name") -> List[str]:
    normalized_input_type = _normalize_input_type(input_type)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        values = []
        for row in reader:
            if normalized_input_type == "drug_name":
                value = _first_nonempty_value(row, ["DRUG_NAME", "drug_name", "Drug.Name", "Drugbank Name"])
            else:
                value = _first_nonempty_value(row, ["SMILES", "smiles", "query_smiles"])

            if value:
                values.append(value)
        return values


def get_output_json_path(drug_name: str, output_dir: Path) -> Path:  #根据输入值生成输出 JSON 文件路径。
    return output_dir / f"{slugify(drug_name)}.json"


def save_selected_outputs(drug_name: str, selected_outputs: Dict[str, Any], output_dir: Path) -> Path:  #保存最终的selected_outputs部分
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = get_output_json_path(drug_name, output_dir)
    json_path.write_text(safe_json_dumps(selected_outputs), encoding="utf-8")
    return json_path


def _extract_text_from_anthropic_message(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _safe_tool_call(tool_name: str, fn, **kwargs) -> Dict[str, Any]: #安全调用工具函数
    try:
        result = fn(**kwargs)
        return {
            "tool_name": tool_name,
            "arguments": kwargs,
            "result": result,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "tool_name": tool_name,
            "arguments": kwargs,
            "result": {"message": str(exc)},
            "status": "error",
        }


def _artifact_has_substance(branch_name: str, artifact: Optional[Dict[str, Any]]) -> bool:  #检查Actor当前生成的artifact是否为空，如果不为空，就认为它是一个可用结果。
    if not isinstance(artifact, dict):
        return False
    if branch_name == "pathway_moa":
        return bool(str(artifact.get("final_moa_artifact", "")).strip())
    if branch_name == "gene":
        return bool(str(artifact.get("final_gene_artifact", "")).strip() or artifact.get("final_gene_set", []))
    return False


def _build_not_found_artifact(drug_name: str, branch_name: str) -> Dict[str, Any]:
    if branch_name == "pathway_moa":
        return {
            "drug_name": drug_name,
            "branch": "pathway_moa",
            "final_moa_artifact": "",
            "message": f"No supported MoA artifact was produced after {MAX_BRANCH_ROUNDS} rounds.",
        }

    return {
        "drug_name": drug_name,
        "branch": "gene",
        "final_gene_set": [],
        "final_gene_artifact": "",
        "message": f"No supported gene artifact was produced after {MAX_BRANCH_ROUNDS} rounds.",
    }


def _compress_recent_critic_history(  #用来压缩最近几轮 Critic 历史。
    iterations: List[Dict[str, Any]],
    keep_last_n: int = MAX_RECENT_CRITIC_HISTORY,
) -> List[Dict[str, Any]]:
    recent = iterations[-keep_last_n:] if keep_last_n > 0 else []
    compressed: List[Dict[str, Any]] = []
    for item in recent:
        critic_result = item.get("critic_result", {}) or {}
        critic_json = critic_result.get("critic_result", {}) or {}
        compressed.append(
            {
                "round_index": item.get("round_index"),
                "decision": critic_json.get("decision", ""),
                "next_step_suggestion": critic_json.get("next_step_suggestion", ""),
            }
        )
    return compressed


def _compress_previous_artifact(   #压缩上一轮Actor生成的artifact
    branch_name: str,
    artifact: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(artifact, dict):
        return None

    if branch_name == "pathway_moa":
        return {
            "branch": "pathway_moa",
            "final_moa_artifact": artifact.get("final_moa_artifact", ""),
            "message": artifact.get("message", ""),
        }

    return {
        "branch": "gene",
        "final_gene_set": artifact.get("final_gene_set", [])[:MAX_PREVIOUS_LIST_ITEMS],
        "final_gene_artifact": artifact.get("final_gene_artifact", ""),
        "message": artifact.get("message", ""),
    }


def _compress_tool_result_for_branch(
    branch_name: str,
    tool_result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(tool_result, dict):
        return {}

    tool_name = tool_result.get("tool_name", "")
    if tool_name != "get_cladd_kg_reports":
        return tool_result

    result = tool_result.get("result", {}) or {}
    compressed_result: Dict[str, Any]

    if branch_name == "pathway_moa":
        compressed_result = {
            "drugrel_report": result.get("drugrel_report", ""),
            "biorel_report": result.get("biorel_report", ""),
            "message": result.get("message", ""),
        }
    else:
        compressed_result = {
            "related_gene_set": result.get("related_gene_set", []),
            "biorel_report": result.get("biorel_report", ""),
            "message": result.get("message", ""),
        }

    return {
        "tool_name": tool_name,
        "arguments": tool_result.get("arguments", {}),
        "result": compressed_result,
        "status": tool_result.get("status", ""),
    }


def _build_cladd_kg_cache_key(arguments: Dict[str, Any]) -> str:
    drug_name = str(arguments.get("drug_name") or "").strip()
    smiles = str(arguments.get("smiles") or "").strip()

    if drug_name:
        return f"drug_name::{drug_name}"
    if smiles:
        return f"smiles::{smiles}"
    return safe_json_dumps(arguments)

#####这里决定了不同的分支可以看到哪些工具##########

MOA_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_cladd_molt5_description": {
        "schema": get_cladd_molt5_description_doc,
        "fn": get_cladd_molt5_description,
    },
    "get_cladd_kg_reports": {
        "schema": get_cladd_kg_reports_doc,
        "fn": get_cladd_kg_reports,
    },
    "get_known_moa": {
        "schema": get_known_moa_doc,
        "fn": get_known_moa,
    },
    "get_pubmed_articles": {
        "schema": get_pubmed_articles_doc,
        "fn": get_pubmed_articles,
    },
}

GENE_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_known_gene": {
        "schema": get_known_gene_doc,
        "fn": get_known_gene,
    },
    "get_cladd_kg_reports": {
        "schema": get_cladd_kg_reports_doc,
        "fn": get_cladd_kg_reports,
    },
    "get_geneagent_geneset_info": {
        "schema": get_geneagent_geneset_info_doc,
        "fn": get_geneagent_geneset_info,
    },
    "get_pubmed_articles": {
        "schema": get_pubmed_articles_doc,
        "fn": get_pubmed_articles,
    },
}

SMILES_MOA_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_cladd_molt5_description": {
        "schema": get_cladd_molt5_description_doc,
        "fn": get_cladd_molt5_description,
    },
    "get_cladd_kg_reports": {
        "schema": get_cladd_kg_reports_doc,
        "fn": get_cladd_kg_reports,
    },
}

SMILES_GENE_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_cladd_kg_reports": {
        "schema": get_cladd_kg_reports_doc,
        "fn": get_cladd_kg_reports,
    },
    "get_geneagent_geneset_info": {
        "schema": get_geneagent_geneset_info_doc,
        "fn": get_geneagent_geneset_info,
    },
}


def _get_tool_registry(branch_name: str, input_type: str) -> Dict[str, Dict[str, Any]]: ##根据当前分支和输入类型，选择可用工具列表
    normalized_input_type = _normalize_input_type(input_type)
    if normalized_input_type == "smiles":
        if branch_name == "pathway_moa":
            return SMILES_MOA_TOOL_REGISTRY
        if branch_name == "gene":
            return SMILES_GENE_TOOL_REGISTRY
    else:
        if branch_name == "pathway_moa":
            return MOA_TOOL_REGISTRY
        if branch_name == "gene":
            return GENE_TOOL_REGISTRY
    raise ValueError(f"Unknown branch/input_type combination: {branch_name} / {input_type}")


MOA_ACTOR_PROMPT = """
You are a pharmacology expert. Your task is to produce the final drug mechanism description for a given drug based on the available evidence.

You may use only these tools:
1. get_cladd_molt5_description
2. get_cladd_kg_reports
3. get_known_moa
4. get_pubmed_articles

Tool-use policy:
1. Use the tool outputs only. Do not speculate.
2. When revising based on previous_critic_suggestion, preserve all correct and evidence-supported content from previous_artifact. Only revise parts that are unsupported, incorrect, incomplete, or explicitly identified for revision by the critic.
3. final_moa_artifact must be a concise integrated description synthesized from:
   - MolT5 molecular description
   - CLADD DrugRel report
   - CLADD BioRel report
   - known_moa list
4. Use get_pubmed_articles as a supplementary tool when the structured tools are insufficient or when you need extra literature support.
5. If one source is empty, still produce the best grounded integrated artifact from the remaining sources.
6. Read the full evidence range, including mechanism, pharmacokinetics, transport, resistance, toxicity, and disease context when present, but summarize around the core pharmacodynamic mechanism.
7. The first sentence of final_moa_artifact must state the primary direct target(s), pathway, or lesion responsible for the drug's main mechanism of action.
8. Broader context such as metabolism, transporters, resistance, toxicity, or clinical indication may be used to interpret the evidence, but should only appear briefly and only if it materially clarifies the core mechanism. Do not let these context items dominate the artifact.
9. Do not include tool-debugging or process notes in final_moa_artifact, such as statements about a caption being unrelated, a tool failing, or which source was trusted more.
10. Prefer the style of a mechanism summary rather than an evidence inventory: 1-3 sentences, centered on the core MoA, with optional brief supporting context.
11. The final JSON should contain only the integrated result, not a retained evidence summary.

Required final JSON schema:
{
  "drug_name": "string",
  "branch": "pathway_moa",
  "final_moa_artifact": "string"
}

Output rules:
- Return JSON only.
- final_moa_artifact must be grounded directly in tool outputs.
- Keep the entire JSON output within 2000 tokens.
""".strip()


GENE_ACTOR_PROMPT = """
You are a pharmacology expert. Your task is to produce the final drug mechanism description for a given drug based on the available evidence.

You may use only these tools:
1. get_known_gene
2. get_cladd_kg_reports
3. get_geneagent_geneset_info
4. get_pubmed_articles

Tool-use policy:
1. Use the tool outputs only. Do not speculate.
2. When revising based on previous_critic_suggestion, preserve all correct and evidence-supported content from previous_artifact. Only revise parts that are unsupported, incorrect, incomplete, or explicitly identified for revision by the critic.
3. You should first get known_gene and related_gene_set.
4. Build a combined gene set by merging:
   - known_gene from get_known_gene
   - related_gene_set gene names from get_cladd_kg_reports
5. If the combined gene set is empty or clearly insufficient, use get_pubmed_articles before get_geneagent_geneset_info to find additional candidate genes or clarify gene relevance.
6. After updating the combined gene set, if it is non-empty, call get_geneagent_geneset_info once using a comma-separated gene set string.
7. final_gene_artifact must be a concise integrated description grounded in:
   - known_gene
   - related_gene_set
   - GeneAgent final description if available
8. If no genes are available after the structured tools and PubMed follow-up, return an empty final_gene_set and an empty final_gene_artifact.

Required final JSON schema:
{
  "drug_name": "string",
  "branch": "gene",
  "final_gene_set": ["string"],
  "final_gene_artifact": "string"
}

Output rules:
- Return JSON only.
- final_gene_set must be a flat de-duplicated list of gene symbols.
- final_gene_artifact must be grounded directly in tool outputs.
- Keep the entire JSON output within 2000 tokens.
""".strip()


SMILES_MOA_ACTOR_PROMPT = """
You are a pharmacology expert. Your task is to produce the final drug mechanism description for a given drug based on the available evidence.

You use only these tools:
1. get_cladd_molt5_description
2. get_cladd_kg_reports

Tool-use policy:
1. Use the tool outputs only. Do not speculate.
2. When revising based on previous_critic_suggestion, preserve all correct and evidence-supported content from previous_artifact. Only revise parts that are unsupported, incorrect, incomplete, or explicitly identified for revision by the critic.
3. Call both tools using the input SMILES as the smiles argument.
4. Do not repeat an identical tool call unless the previous result was clearly unusable.
5. final_moa_artifact must be a concise integrated description synthesized from:
   - MolT5 molecular description
   - CLADD DrugRel report
   - CLADD BioRel report
6. Read the full evidence range, including mechanism, pharmacokinetics, transport, resistance, toxicity, and disease context when present, but summarize around the core pharmacodynamic mechanism.
7. The first sentence of final_moa_artifact must state the primary direct target(s), pathway, or lesion responsible for the drug's main mechanism of action.
8. Broader context such as metabolism, transporters, resistance, toxicity, or clinical indication may be used to interpret the evidence, but should only appear briefly and only if it materially clarifies the core mechanism. Do not let these context items dominate the artifact.
9. Do not include tool-debugging or process notes in final_moa_artifact, such as statements about a caption being unrelated, a tool failing, or which source was trusted more.
10. Prefer the style of a mechanism summary rather than an evidence inventory: 1-3 sentences, centered on the core MoA, with optional brief supporting context.
11. If one source is empty, still produce the best grounded integrated artifact from the remaining sources.

Required final JSON schema:
{
  "drug_name": "string",
  "branch": "pathway_moa",
  "final_moa_artifact": "string"
}

Output rules:
- Return JSON only.
- final_moa_artifact must be grounded directly in tool outputs.
- Keep the entire JSON output within 2000 tokens.
""".strip()


SMILES_GENE_ACTOR_PROMPT = """
You are a pharmacology expert. Your task is to produce the final drug mechanism description for a given drug based on the available evidence.

You use only these tools:
1. get_cladd_kg_reports
2. get_geneagent_geneset_info

Tool-use policy:
1. Use the tool outputs only. Do not speculate.
2. When revising based on previous_critic_suggestion, preserve all correct and evidence-supported content from previous_artifact. Only revise parts that are unsupported, incorrect, incomplete, or explicitly identified for revision by the critic.
3. Call get_cladd_kg_reports using the input SMILES as the smiles argument.
4. Use only the related_gene_set from CLADD KG output as the structured gene source in this mode.
5. If related_gene_set is non-empty, build a de-duplicated gene set from it and call get_geneagent_geneset_info once using a comma-separated gene set string.
6. final_gene_artifact must be a concise integrated description grounded in:
   - related_gene_set
   - CLADD BioRel report if available
   - GeneAgent final description if available
7. final_gene_set must contain the final de-duplicated gene set used for the conclusion.
8. If no related genes are available, return an empty final_gene_set and an empty final_gene_artifact.

Required final JSON schema:
{
  "drug_name": "string",
  "branch": "gene",
  "final_gene_set": ["string"],
  "final_gene_artifact": "string"
}

Output rules:
- Return JSON only.
- final_gene_set must be a flat de-duplicated list of gene symbols.
- Keep the entire JSON output within 2000 tokens.
""".strip()


MOA_CRITIC_PROMPT = """
You are a pharmacology expert. Your task is to evaluate whether the current description of the drug’s mechanism of action is accurate and complete based on the available evidence.
Return JSON only.

Required JSON schema:
{
  "branch": "pathway_moa",
  "decision": "stop or continue",
  "next_step_suggestion": "Preserve: ... Revise: ..."
}

Evaluation focus:
1. Whether final_moa_artifact is specific, concise, and grounded in the available tool outputs and tool_trace.
2. Whether the artifact avoids speculation.
3. Whether another round is actually likely to improve the integrated artifact.

Decision policy:
- Prefer stop when final_moa_artifact is already grounded and usable.
- Prefer continue only when there is a clear missing tool result or a clear integration problem.

Output rules:
- Return JSON only.
- next_step_suggestion must explicitly contain two parts:
  Preserve: what should be kept unchanged because it is already correct or well-supported.
  Revise: what should be improved, checked, added, removed, or rewritten in the next round.
""".strip()


GENE_CRITIC_PROMPT = """
You are a pharmacology expert. Your task is to evaluate whether the current description of the drug’s mechanism of action is accurate and complete based on the available evidence.
Return JSON only.

Required JSON schema:
{
  "branch": "gene",
  "decision": "stop or continue",
  "next_step_suggestion": "Preserve: ... Revise: ..."
}

Evaluation focus:
1. Whether final_gene_set is a plausible de-duplicated gene list grounded in the available tool outputs and tool_trace.
2. Whether final_gene_artifact is grounded in the available tool outputs and tool_trace.
3. Whether the artifact avoids speculation.
4. Whether another round is actually likely to improve the result.

Decision policy:
- Prefer stop when final_gene_artifact is already grounded and usable.
- Prefer continue only when there is a clear missing tool result or a clear integration problem.

Output rules:
- Return JSON only.
- next_step_suggestion must explicitly contain two parts:
  Preserve: what should be kept unchanged because it is already correct or well-supported.
  Revise: what should be improved, checked, added, removed, or rewritten in the next round.
""".strip()


SMILES_MOA_CRITIC_PROMPT = """
You are a pharmacology expert. Your task is to evaluate whether the current description of the drug’s mechanism of action is accurate and complete based on the available evidence.
Return JSON only.

Required JSON schema:
{
  "branch": "pathway_moa",
  "decision": "stop or continue",
  "next_step_suggestion": "Preserve: ... Revise: ..."
}

Evaluation focus:
1. Whether final_moa_artifact is specific, concise, and grounded in the available tool outputs and tool_trace.
2. Whether the artifact avoids speculation.
3. Whether another round is actually likely to improve the integrated artifact.

Decision policy:
- Prefer stop when final_moa_artifact is already grounded and usable.
- Prefer continue only when there is a clear missing tool result or a clear integration problem.

Output rules:
- Return JSON only.
- next_step_suggestion must explicitly contain two parts:
  Preserve: what should be kept unchanged because it is already correct or well-supported.
  Revise: what should be improved, checked, added, removed, or rewritten in the next round.
""".strip()


SMILES_GENE_CRITIC_PROMPT = """
You are a pharmacology expert. Your task is to evaluate whether the current description of the drug’s mechanism of action is accurate and complete based on the available evidence.
Return JSON only.

Required JSON schema:
{
  "branch": "gene",
  "decision": "stop or continue",
  "next_step_suggestion": "Preserve: ... Revise: ..."
}

Evaluation focus:
1. Whether final_gene_set is a plausible de-duplicated gene list grounded in the available tool outputs and tool_trace.
2. Whether final_gene_artifact is grounded in the available tool outputs and tool_trace.
3. Whether the artifact avoids speculation.
4. Whether another round is actually likely to improve the result.

Decision policy:
- Prefer stop when final_gene_artifact is already grounded and usable.
- Prefer continue only when there is a clear missing tool result or a clear integration problem.

Output rules:
- Return JSON only.
- next_step_suggestion must explicitly contain two parts:
  Preserve: what should be kept unchanged because it is already correct or well-supported.
  Revise: what should be improved, checked, added, removed, or rewritten in the next round.
""".strip()


def _get_actor_prompt(branch_name: str, input_type: str) -> str:
    normalized_input_type = _normalize_input_type(input_type)
    if normalized_input_type == "smiles":
        if branch_name == "pathway_moa":
            return SMILES_MOA_ACTOR_PROMPT
        if branch_name == "gene":
            return SMILES_GENE_ACTOR_PROMPT
    if branch_name == "pathway_moa":
        return MOA_ACTOR_PROMPT
    if branch_name == "gene":
        return GENE_ACTOR_PROMPT
    raise ValueError(f"Unknown actor branch/input_type: {branch_name} / {input_type}")


def _get_critic_prompt(branch_name: str, input_type: str) -> str:
    normalized_input_type = _normalize_input_type(input_type)
    if normalized_input_type == "smiles":
        if branch_name == "pathway_moa":
            return SMILES_MOA_CRITIC_PROMPT
        if branch_name == "gene":
            return SMILES_GENE_CRITIC_PROMPT
    if branch_name == "pathway_moa":
        return MOA_CRITIC_PROMPT
    if branch_name == "gene":
        return GENE_CRITIC_PROMPT
    raise ValueError(f"Unknown critic branch/input_type: {branch_name} / {input_type}")


def build_actor_messages(
    query_value: str,
    input_type: str,
    branch_name: str,
    round_index: int,
    previous_artifact: Optional[Dict[str, Any]],
    previous_critic: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized_input_type = _normalize_input_type(input_type)
    payload = {
        "input_type": normalized_input_type,
        "input_value": query_value,
        "branch": branch_name,
        "round_index": round_index,
        "previous_artifact": previous_artifact,
        "previous_critic_suggestion": previous_critic,
    }
    if normalized_input_type == "smiles":
        payload["smiles"] = query_value
    else:
        payload["drug_name"] = query_value
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": _get_actor_prompt(branch_name, normalized_input_type)}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": safe_json_dumps(payload)}]},
    ]


def _tool_output_message(call_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": safe_json_dumps(payload),
    }


def _prepare_tool_arguments(  #用来修正模型生成的工具参数
    tool_name: str,
    arguments: Dict[str, Any],
    query_value: str,
    input_type: str,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    prepared = dict(arguments or {})

    if tool_name in {"get_cladd_molt5_description", "get_cladd_kg_reports"}:
        prepared.pop("drug_name", None)
        prepared.pop("smiles", None)
        if normalized_input_type == "smiles":
            prepared["smiles"] = query_value
        else:
            prepared["drug_name"] = query_value
        return prepared

    if tool_name in {"get_known_gene", "get_known_moa"}:
        prepared.pop("smiles", None)
        prepared["drug_name"] = query_value
        return prepared

    return prepared


def run_branch_actor_with_tools(
    query_value: str,
    input_type: str,
    branch_name: str,
    round_index: int,
    previous_artifact: Optional[Dict[str, Any]],
    previous_critic: Optional[Dict[str, Any]],
    model: str = DEFAULT_ACTOR_MODEL,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    registry = _get_tool_registry(branch_name, normalized_input_type)
    tools = [item["schema"] for item in registry.values()]

    conversation_items: List[Dict[str, Any]] = build_actor_messages(
        query_value=query_value,
        input_type=normalized_input_type,
        branch_name=branch_name,
        round_index=round_index,
        previous_artifact=previous_artifact,
        previous_critic=previous_critic,
    )

    tool_trace: List[Dict[str, Any]] = []
    previous_response_id: Optional[str] = None

    for _ in range(MAX_TOOL_CALLS_PER_ROUND):
        request_kwargs = {
            "model": model,
            "input": conversation_items,
            "tools": tools,
            "tool_choice": "auto",
            "max_output_tokens": MAX_OUTPUT_TOKENS_ACTOR,
        }

        if previous_response_id is not None:
            request_kwargs["previous_response_id"] = previous_response_id

        response = openai_client.responses.create(**request_kwargs)
        previous_response_id = response.id

        output_items = getattr(response, "output", []) or []
        function_calls = [
            item for item in output_items
            if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            final_text = (response.output_text or "").strip()
            final_json = parse_json_response(final_text)
            return {
                "status": "artifact_generated",
                "drug_name": query_value,
                "input_type": normalized_input_type,
                "input_value": query_value,
                "branch": branch_name,
                "round_index": round_index,
                "model": model,
                "artifact": final_json,
                "tool_trace": tool_trace,
                "raw_output_text": final_text,
            }

        tool_outputs_for_next_turn: List[Dict[str, Any]] = []

        for fc in function_calls:
            tool_name = fc.name
            call_id = fc.call_id

            try:
                arguments = json.loads(fc.arguments or "{}")
            except Exception:
                arguments = {}
            arguments = _prepare_tool_arguments(
                tool_name=tool_name,
                arguments=arguments,
                query_value=query_value,
                input_type=normalized_input_type,
            )

            if tool_name not in registry:
                tool_result = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": {"message": f"Unknown tool: {tool_name}"},
                    "status": "error",
                }
            elif tool_name == "get_cladd_kg_reports":
                cache_key = _build_cladd_kg_cache_key(arguments)
                if cache_key in _CLADD_KG_CACHE:
                    tool_result = {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": copy.deepcopy(_CLADD_KG_CACHE[cache_key]),
                        "status": "ok",
                    }
                else:
                    tool_result = _safe_tool_call(
                        tool_name,
                        registry[tool_name]["fn"],
                        **arguments,
                    )
                    if tool_result.get("status") == "ok":
                        _CLADD_KG_CACHE[cache_key] = copy.deepcopy(tool_result.get("result", {}))
            else:
                tool_result = _safe_tool_call(
                    tool_name,
                    registry[tool_name]["fn"],
                    **arguments,
                )

            compressed_tool_result = _compress_tool_result_for_branch(branch_name, tool_result)
            tool_trace.append(compressed_tool_result)
            tool_outputs_for_next_turn.append(_tool_output_message(call_id, compressed_tool_result))

        conversation_items = tool_outputs_for_next_turn

    return {
        "status": "failed",
        "drug_name": query_value,
        "input_type": normalized_input_type,
        "input_value": query_value,
        "branch": branch_name,
        "round_index": round_index,
        "model": model,
        "artifact": _build_not_found_artifact(query_value, branch_name),
        "tool_trace": tool_trace,
        "raw_output_text": "Tool loop exceeded maximum allowed calls in this round.",
    }


def run_branch_critic_once(
    query_value: str,
    input_type: str,
    branch_name: str,
    round_index: int,
    tool_trace: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    recent_critic_history: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_CRITIC_MODEL,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    payload = {
        "drug_name": query_value,
        "input_type": normalized_input_type,
        "input_value": query_value,
        "branch": branch_name,
        "round_index": round_index,
        "recent_critic_history": recent_critic_history or [],
        "tool_trace": tool_trace,
        "current_artifact": artifact,
    }
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS_CRITIC,
        temperature=0,
        system=_get_critic_prompt(branch_name, normalized_input_type),
        messages=[{"role": "user", "content": safe_json_dumps(payload)}],
    )
    text = _extract_text_from_anthropic_message(message)
    critic_json = parse_json_response(text)
    return {
        "critic_model": model,
        "critic_result": critic_json,
    }


def run_branch_loop(
    query_value: str,
    input_type: str,
    branch_name: str,
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    iterations: List[Dict[str, Any]] = []
    previous_artifact: Optional[Dict[str, Any]] = None
    previous_critic: Optional[Dict[str, Any]] = None
    best_artifact: Optional[Dict[str, Any]] = None
    best_round_index: Optional[int] = None

    for round_index in range(1, MAX_BRANCH_ROUNDS + 1):
        actor_result = run_branch_actor_with_tools(
            query_value=query_value,
            input_type=normalized_input_type,
            branch_name=branch_name,
            round_index=round_index,
            previous_artifact=previous_artifact,
            previous_critic=previous_critic,
            model=actor_model,
        )

        recent_critic_history = _compress_recent_critic_history(
            iterations,
            keep_last_n=MAX_RECENT_CRITIC_HISTORY,
        )

        critic_result = run_branch_critic_once(
            query_value=query_value,
            input_type=normalized_input_type,
            branch_name=branch_name,
            round_index=round_index,
            tool_trace=actor_result.get("tool_trace", []),
            artifact=actor_result["artifact"],
            recent_critic_history=recent_critic_history,
            model=critic_model,
        )

        decision = str(critic_result["critic_result"].get("decision", "stop")).lower().strip()
        if decision not in {"stop", "continue"}:
            decision = "stop"

        iteration_record = {
            "round_index": round_index,
            "actor_result": actor_result,
            "critic_result": critic_result,
        }
        iterations.append(iteration_record)

        artifact_has_substance = _artifact_has_substance(branch_name, actor_result.get("artifact"))
        if artifact_has_substance:
            best_artifact = actor_result["artifact"]
            best_round_index = round_index

        if decision == "stop":
            if best_artifact is not None:
                return {
                    "branch": branch_name,
                    "status": "completed",
                    "iterations": iterations,
                    "final_artifact": best_artifact,
                    "final_critic": critic_result["critic_result"],
                    "final_artifact_from_round": best_round_index,
                }
            break

        if round_index == MAX_BRANCH_ROUNDS:
            break

        previous_artifact = _compress_previous_artifact(
            branch_name=branch_name,
            artifact=best_artifact if best_artifact is not None else actor_result["artifact"],
        )
        previous_critic = {
            "round_index": round_index,
            "decision": critic_result["critic_result"].get("decision", ""),
            "next_step_suggestion": critic_result["critic_result"].get("next_step_suggestion", ""),
        }

    if best_artifact is not None:
        return {
            "branch": branch_name,
            "status": "completed",
            "iterations": iterations,
            "final_artifact": best_artifact,
            "final_critic": iterations[-1]["critic_result"]["critic_result"] if iterations else {},
            "final_artifact_from_round": best_round_index,
        }

    return {
        "branch": branch_name,
        "status": "not_found_after_max_rounds",
        "iterations": iterations,
        "final_artifact": _build_not_found_artifact(query_value, branch_name),
        "final_critic": iterations[-1]["critic_result"]["critic_result"] if iterations else {},
    }


def run_dual_branch_generation_for_input(
    query_value: str,
    input_type: str = "drug_name",
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    pathway_result = run_branch_loop(
        query_value=query_value,
        input_type=normalized_input_type,
        branch_name="pathway_moa",
        actor_model=actor_model,
        critic_model=critic_model,
    )
    gene_result = run_branch_loop(
        query_value=query_value,
        input_type=normalized_input_type,
        branch_name="gene",
        actor_model=actor_model,
        critic_model=critic_model,
    )

    return {
        "status": "completed",
        "drug_name": query_value,
        "input_type": normalized_input_type,
        "input_value": query_value,
        "branches": {
            "pathway_moa": pathway_result,
            "gene": gene_result,
        },
        "selected_outputs": {
            "pathway_moa": pathway_result.get("final_artifact"),
            "gene": gene_result.get("final_artifact"),
        },
    }


def generate_for_single_input(
    query_value: str,
    input_type: str = "drug_name",
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    result_dir: Path = DEFAULT_RESULT_DIR,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    normalized_input_type = _normalize_input_type(input_type)
    result_json_path = get_output_json_path(query_value, result_dir)

    if skip_existing and result_json_path.exists():
        selected_outputs = json.loads(result_json_path.read_text(encoding="utf-8"))
        return {
            "status": "completed",
            "drug_name": query_value,
            "input_type": normalized_input_type,
            "input_value": query_value,
            "selected_outputs": selected_outputs,
        }

    result = run_dual_branch_generation_for_input(
        query_value=query_value,
        input_type=normalized_input_type,
        actor_model=actor_model,
        critic_model=critic_model,
    )

    if result.get("status") == "completed":
        save_selected_outputs(query_value, result["selected_outputs"], result_dir)

    return result


def batch_generate_initial_descriptions_agentic( #这是批量运行的入口，如果只运行csv中的前十个，那么可以让limit为10
    csv_path: Path = DEFAULT_DRUG_FILE,
    output_dir: Path = DEFAULT_RESULT_DIR,
    input_type: str = "drug_name",
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    limit: Optional[int] = 125,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    normalized_input_type = _normalize_input_type(input_type)
    input_values = read_input_values(csv_path, input_type=normalized_input_type)
    if limit is not None:
        input_values = input_values[:limit]

    total = len(input_values)
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    label = "drug_name" if normalized_input_type == "drug_name" else "smiles"

    for idx, query_value in enumerate(input_values, start=1):
        result_json_path = get_output_json_path(query_value, output_dir)

        if skip_existing and result_json_path.exists():
            print(f"[{idx}/{total}] Skipped: {query_value} ({label}, already exists in {output_dir})")
            selected_outputs = json.loads(result_json_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "status": "completed",
                    "drug_name": query_value,
                    "input_type": normalized_input_type,
                    "input_value": query_value,
                    "selected_outputs": selected_outputs,
                }
            )
            continue

        print(f"[{idx}/{total}] Starting generation: {query_value} ({label})")
        try:
            result = run_dual_branch_generation_for_input(
                query_value=query_value,
                input_type=normalized_input_type,
                actor_model=actor_model,
                critic_model=critic_model,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "drug_name": query_value,
                "input_type": normalized_input_type,
                "input_value": query_value,
                "actor_model": actor_model,
                "critic_model": critic_model,
                "error": str(exc),
            }

        results.append(result)

        if result.get("status") == "failed":
            print(f"[{idx}/{total}] Failed: {query_value} ({label}) | error={result.get('error')}")
        else:
            saved_path = save_selected_outputs(query_value, result["selected_outputs"], output_dir)
            print(f"[{idx}/{total}] Finished: {query_value} ({label}) | saved_to={saved_path.name}")

    return results


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV file containing drug names or SMILES.",
    )
    group.add_argument(
        "--drug-name",
        action="append",
        nargs="+",
        default=None,
        help="Single drug name input. Repeat the flag or pass multiple values after one flag.",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="drug_name",
        choices=sorted(SUPPORTED_INPUT_TYPES),
        help="Interpret input values as drug names or SMILES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for saved JSON outputs.",
    )
    parser.add_argument(
        "--actor-model",
        type=str,
        default=DEFAULT_ACTOR_MODEL,
        help="Model used for the actor branch.",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default=DEFAULT_CRITIC_MODEL,
        help="Model used for the critic branch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit when running from CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if the JSON file already exists.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()
    skip_existing = not args.overwrite

    if args.csv_path is not None:
        results = batch_generate_initial_descriptions_agentic(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            input_type=args.input_type,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            limit=args.limit,
            skip_existing=skip_existing,
        )
    else:
        drug_names = [value.strip() for group in (args.drug_name or []) for value in group if value.strip()]
        results = []
        for query_value in drug_names:
            print(f"Starting generation: {query_value} ({args.input_type})")
            result = generate_for_single_input(
                query_value=query_value,
                input_type=args.input_type,
                actor_model=args.actor_model,
                critic_model=args.critic_model,
                result_dir=args.output_dir,
                skip_existing=skip_existing,
            )
            results.append(result)
            print(f"Finished generation: {query_value} | status={result.get('status')}")

    print(
        safe_json_dumps(
            {
                "num_results": len(results),
                "output_dir": str(args.output_dir),
                "actor_model": args.actor_model,
                "critic_model": args.critic_model,
                "max_branch_rounds": MAX_BRANCH_ROUNDS,
                "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
                "input_type": args.input_type,
            }
        )
    )
