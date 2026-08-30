
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_JSON_DIR = BASE_DIR / "result3" / "descripition"
DEFAULT_OUTPUT_DIR = BASE_DIR / "result3" / "Pathway_map"

DEFAULT_ALL_PATHWAYS_CSV = BASE_DIR / "reactome_layer" / "all_pathways_with_layers.csv"
DEFAULT_RELATION_TXT = BASE_DIR / "reactome_layer" / "ReactomePathwaysRelation.txt"

DEFAULT_OPENAI_API_KEY_FILE = BASE_DIR / "openai_api_key.txt"
DEFAULT_ANTHROPIC_API_KEY_FILE = BASE_DIR / "claude_api_key.txt"

DEFAULT_ACTOR_MODEL = "gpt-5-mini"
DEFAULT_CRITIC_MODEL = "claude-haiku-4-5-20251001"

MAX_DEPTH = 5
MAX_RESELECT_PER_DEPTH = 2
MAX_RECENT_CRITIC_HISTORY = 2
MAX_PATHWAYS_PER_STEP = 5


# =========================
# Basic utils
# =========================
def load_api_key(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def compact_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


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


def _extract_text_from_anthropic_message(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_json_dumps(data), encoding="utf-8")


# =========================
# API clients
# =========================
OPENAI_API_KEY = load_api_key(DEFAULT_OPENAI_API_KEY_FILE) or os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = load_api_key(DEFAULT_ANTHROPIC_API_KEY_FILE) or os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key is missing.")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Anthropic API key is missing.")
if Anthropic is None:
    raise RuntimeError("anthropic package is not installed. Please install it first.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# =========================
# Load stage-1 result from result3/descripition
# =========================
def load_stage1_result(json_path: Path) -> Dict[str, Any]:  #读取Step1中的结果
    return json.loads(json_path.read_text(encoding="utf-8"))


def extract_final_artifact_only(stage1_result: Dict[str, Any], branch_name: str) -> Any: #提取某个 branch 的 final artifact
    # Expected format in result3/descripition:
    # {
    #   "pathway_moa": {"drug_name": "...", "final_moa_artifact": "..."},
    #   "gene": {"drug_name": "...", "final_gene_artifact": "..."}
    # }
    branch_result = stage1_result.get(branch_name)
    if not isinstance(branch_result, dict):
        raise ValueError(f"Missing branch '{branch_name}' in stage-1 result.")

    if branch_name == "pathway_moa":
        artifact = branch_result.get("final_moa_artifact")
    elif branch_name == "gene":
        artifact = branch_result.get("final_gene_artifact")
    else:
        raise ValueError(f"Unsupported branch '{branch_name}'.")

    if not artifact:
        raise ValueError(f"Missing final artifact for branch '{branch_name}'.")
    return artifact


# =========================
# Pathway table / relation graph
# =========================
def load_pathways_df(csv_path: Path) -> pd.DataFrame:  #读取 Reactome pathway 的基本信息表
    df = pd.read_csv(csv_path)
    required_cols = {"PathwayID", "PathwayName", "Layer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in pathway CSV: {missing}")
    df["Layer"] = df["Layer"].astype(int)
    return df


def load_relations(relation_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:  #读取 Reactome pathway 的父子关系文件
    children_map: Dict[str, List[str]] = {}
    parent_map: Dict[str, List[str]] = {}

    with relation_path.open("r", encoding="utf-8") as f:
        _ = next(f, None)
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parent, child = line.split(";", 1)
            parent = parent.strip()
            child = child.strip()
            children_map.setdefault(parent, []).append(child)
            parent_map.setdefault(child, []).append(parent)

    return children_map, parent_map


def get_layer0_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    sub = df[df["Layer"] == 0].copy()
    sub = sub.sort_values(["PathwayName"])
    return sub[["PathwayID", "PathwayName", "Layer"]].to_dict(orient="records")


def get_child_candidates( #获取某个 parent pathway 的 child pathways
    df: pd.DataFrame,
    children_map: Dict[str, List[str]],
    parent_id: str,
) -> List[Dict[str, Any]]:
    child_ids = children_map.get(parent_id, [])
    if not child_ids:
        return []

    sub = df[df["PathwayID"].isin(child_ids)].copy()
    sub = sub.sort_values(["Layer", "PathwayName"])
    return sub[["PathwayID", "PathwayName", "Layer"]].to_dict(orient="records")


def build_selected_pathway_child_candidates(
    selected_pathways: List[Dict[str, Any]],
    pathways_df: pd.DataFrame,
    children_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    grouped = []
    for item in selected_pathways:
        pathway_id = item.get("PathwayID")
        if not pathway_id:
            continue
        grouped.append({
            "pathway_id": pathway_id,
            "pathway_name": item.get("PathwayName", ""),
            "layer": item.get("Layer"),
            "child_candidates": get_child_candidates(
                pathways_df,
                children_map,
                pathway_id,
            ),
        })
    return grouped


# =========================
# Prompts
# =========================
MOA_ACTOR_PROMPT = f"""
You are a pharmacology expert. Your task is to select the Reactome pathways at the current layer that best match the final description of the drug’s mechanism of action.

At each step, you are given:
- the drug mechanism-of-action description from stage-1
- a candidate list of Reactome pathways restricted to the current layer

Your job is to choose the 1 to {MAX_PATHWAYS_PER_STEP} candidates that are best supported by the provided artifact at the current layer.

Selection rules:
1. Select only from the provided candidate list.
2. Do not invent pathway IDs or pathway names.
3. This 1–{MAX_PATHWAYS_PER_STEP} limit applies only to the current step, not to the total number of final pathways across all depths.
4. Do not remove a candidate only because another selected candidate is broader or more general.
5. If critic feedback is provided later, reselect only from the same candidate list for this step unless a new candidate list is explicitly provided.
6. If a reselect request includes fixed_selected_pathways and reselect_pathways, keep the fixed_selected_pathways unchanged and return only replacement pathways for the reselect_pathways in selected_pathways.

Return JSON only.

Required schema:
{{
  "branch": "pathway_moa",
  "depth": 0,
  "selected_pathways": [
    {{
      "pathway_id": "R-HSA-...",
      "pathway_name": "string",
      "confidence": "high or medium or low"
    }}
  ]
}}
""".strip()


GENE_ACTOR_PROMPT = f"""
You are a pharmacology expert. Your task is to select the Reactome pathways at the current layer that best match the gene evidence related to the drug’s mechanism of action.

At each step, you are given:
- the stage-1 gene evidence description related to the drug’s mechanism of action
- a candidate list of Reactome pathways restricted to the current layer

Your job is to choose the 1 to {MAX_PATHWAYS_PER_STEP} candidates that are best supported by the provided artifact at the current layer.

Selection rules:
1. Select only from the provided candidate list.
2. Do not invent pathway IDs or pathway names.
3. This 1–{MAX_PATHWAYS_PER_STEP} limit applies only to the current step, not to the total number of final pathways across all depths.
4. Do not remove a candidate only because another selected candidate is broader or more general.
5. If critic feedback is provided later, reselect only from the same candidate list for this step unless a new candidate list is explicitly provided.
6. If a reselect request includes fixed_selected_pathways and reselect_pathways, keep the fixed_selected_pathways unchanged and return only replacement pathways for the reselect_pathways in selected_pathways.

Return JSON only.

Required schema:
{{
  "branch": "gene",
  "depth": 0,
  "selected_pathways": [
    {{
      "pathway_id": "R-HSA-...",
      "pathway_name": "string",
      "confidence": "high or medium or low"
    }}
  ]
}}
""".strip()



MOA_CRITIC_PROMPT = """
You are a pharmacology expert. Your task is to evaluate whether each currently selected Reactome pathway is sufficiently supported by the final description of the drug’s mechanism of action, and to decide whether each pathway should be retained at the current layer or further explored through its child pathways.

You can see:
- the drug mechanism-of-action description from stage-1
- the full candidate pool at the current layer
- the actor's currently selected pathways
- the child pathway candidates for those selected pathways

Required schema:
{
  "branch": "pathway_moa",
  "depth": 0,
  "pathway_actions": [
    {
      "pathway_id": "R-HSA-...",
      "action": "keep or descend or reselect",
      "reason": "brief rationale for this action"
    }
  ]
}

Action rules:
- For every currently selected pathway, include a pathway_actions entry with action "keep", "descend", or "reselect".
- Use "keep" when the pathway is sufficiently supported and should remain at the current layer.
- Use "descend" when the pathway is sufficiently supported and should be refined into child pathways.
- Use "reselect" when the pathway is not sufficiently supported or should be replaced from the same current-layer candidate pool.

Evaluation focus:
1. Whether the selected pathways are supported by the artifact.
2. Whether the selected pathways are too broad.
3. Whether each approved pathway should be kept at the current layer or descended into its visible child candidates.

Important:
- Final outputs should keep only pathways that terminate at the current step.
- If a pathway is later refined into child pathways, the parent pathway should not be kept as a final pathway.
""".strip()


GENE_CRITIC_PROMPT = """
You are responsible for evaluating whether the current pathway selections are well supported by the gene evidence.

You can see:
- the stage-1 gene evidence description related to the drug’s mechanism of action
- the full candidate pool at the current layer
- the actor's currently selected pathways
- the child pathway candidates for those selected pathways

Required schema:
{
  "branch": "gene",
  "depth": 0,
  "pathway_actions": [
    {
      "pathway_id": "R-HSA-...",
      "action": "keep or descend or reselect",
      "reason": "brief rationale for this action"
    }
  ]
}

Action rules:
- For every currently selected pathway, include a pathway_actions entry with action "keep", "descend", or "reselect".
- Use "keep" when the pathway is sufficiently supported and should remain at the current layer.
- Use "descend" when the pathway is sufficiently supported and should be refined into child pathways.
- Use "reselect" when the pathway is not sufficiently supported or should be replaced from the same current-layer candidate pool.

Evaluation focus:
1. Whether the selected pathways are supported by the artifact and current evidence.
2. Whether the selected pathways are too broad.
3. Whether each approved pathway should be kept at the current layer or descended into its visible child candidates.

Important:
- Final outputs should keep only pathways that terminate at the current step.
- If a pathway is later refined into child pathways, the parent pathway should not be kept as a final pathway.
""".strip()


def get_actor_prompt(branch_name: str) -> str:
    if branch_name == "pathway_moa":
        return MOA_ACTOR_PROMPT
    if branch_name == "gene":
        return GENE_ACTOR_PROMPT
    raise ValueError(f"Unknown branch: {branch_name}")


def get_critic_prompt(branch_name: str) -> str:
    if branch_name == "pathway_moa":
        return MOA_CRITIC_PROMPT
    if branch_name == "gene":
        return GENE_CRITIC_PROMPT
    raise ValueError(f"Unknown branch: {branch_name}")


# =========================
# Pathway selection helpers
# =========================
def sanitize_selected_pathways(
    selected_pathways: Any,
    candidates: List[Dict[str, Any]],
    max_items: int = MAX_PATHWAYS_PER_STEP,
) -> List[Dict[str, Any]]:
    if not isinstance(selected_pathways, list):
        return []

    candidate_map = {c["PathwayID"]: c for c in candidates}
    out = []
    seen = set()

    for item in selected_pathways:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("pathway_id", "")).strip()
        if not pid or pid in seen:
            continue
        if pid not in candidate_map:
            continue

        c = candidate_map[pid]
        out.append({
            "PathwayID": c["PathwayID"],
            "PathwayName": c["PathwayName"],
            "Layer": c["Layer"],
            "confidence": item.get("confidence", "low"),
        })
        seen.add(pid)

        if len(out) >= max_items:
            break

    return out


def merge_pathway_lists(
    existing: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = []
    seen = set()

    for item in existing + new_items:
        pid = item.get("PathwayID")
        if not pid or pid in seen:
            continue
        merged.append(item)
        seen.add(pid)

    return merged


def convert_sanitized_pathways_to_actor_schema(
    pathways: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    for item in pathways:
        if not isinstance(item, dict):
            continue
        pathway_id = item.get("PathwayID")
        pathway_name = item.get("PathwayName")
        if not pathway_id or not pathway_name:
            continue
        out.append({
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "confidence": item.get("confidence", "low"),
        })
    return out

def is_descendant_of(
    child_id: str,
    parent_id: str,
    children_map: Dict[str, List[str]],
) -> bool:
    """
    Return True if child_id is a descendant of parent_id
    according to ReactomePathwaysRelation.txt.
    """
    if not child_id or not parent_id:
        return False

    stack = list(children_map.get(parent_id, []))
    visited = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current == child_id:
            return True

        stack.extend(children_map.get(current, []))

    return False


def rollback_uncovered_parents(
    parent_nodes: List[Dict[str, Any]],
    child_nodes: List[Dict[str, Any]],
    children_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    If a parent pathway was selected for descent but no selected child
    belongs to its descendant branch, keep that parent as a terminal output.
    """
    uncovered = []

    child_ids = [
        item.get("PathwayID")
        for item in child_nodes
        if isinstance(item, dict) and item.get("PathwayID")
    ]

    for parent in parent_nodes:
        parent_id = parent.get("PathwayID")
        if not parent_id:
            continue

        covered = any(
            is_descendant_of(
                child_id=child_id,
                parent_id=parent_id,
                children_map=children_map,
            )
            for child_id in child_ids
        )

        if not covered:
            uncovered.append(parent)

    return uncovered

def build_pathway_reason_map(branch_result: Dict[str, Any]) -> Dict[str, str]:
    keep_reason_map: Dict[str, str] = {}
    latest_reason_map: Dict[str, str] = {}
    iterations = branch_result.get("iterations", [])

    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        for action in iteration.get("critic_pathway_actions", []):
            if not isinstance(action, dict):
                continue
            pathway_id = str(action.get("pathway_id", "")).strip()
            reason = str(action.get("reason", "")).strip()
            if pathway_id and reason:
                latest_reason_map[pathway_id] = reason
                if str(action.get("action", "")).strip().lower() == "keep":
                    keep_reason_map[pathway_id] = reason

    reason_map: Dict[str, str] = {}
    pathway_ids = set(latest_reason_map) | set(keep_reason_map)
    for pathway_id in pathway_ids:
        reason_map[pathway_id] = keep_reason_map.get(pathway_id) or latest_reason_map.get(pathway_id, "")

    return reason_map


def build_selected_outputs(branch_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    final_selections = branch_result.get("final_selections", [])
    pathway_reason_map = build_pathway_reason_map(branch_result)

    pathways = []
    for item in final_selections:
        if isinstance(item, dict):
            pathway = dict(item)
            pathway_id = str(pathway.get("PathwayID", "")).strip()
            pathway["reason"] = pathway_reason_map.get(pathway_id, "")
            pathways.append(pathway)

    return pathways


def _split_pathways_by_critic_actions(
    critic_json: Dict[str, Any],
    selected_pathways: List[Dict[str, Any]],
    children_map: Dict[str, List[str]],
    depth: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_actions = critic_json.get("pathway_actions", [])

    action_map: Dict[str, Dict[str, str]] = {}
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict):
                continue
            pathway_id = str(item.get("pathway_id", "")).strip()
            action = str(item.get("action", "")).lower().strip()
            if not pathway_id or action not in {"keep", "descend", "reselect"}:
                continue
            action_map[pathway_id] = {
                "pathway_id": pathway_id,
                "action": action,
                "reason": str(item.get("reason", "")).strip(),
            }

    kept_nodes: List[Dict[str, Any]] = []
    descend_nodes: List[Dict[str, Any]] = []
    reselect_nodes: List[Dict[str, Any]] = []
    normalized_actions: List[Dict[str, Any]] = []

    for item in selected_pathways:
        pathway_id = item["PathwayID"]
        has_children = bool(children_map.get(pathway_id, []))
        action = action_map.get(pathway_id, {}).get("action", "reselect")
        reason = action_map.get(pathway_id, {}).get("reason", "")

        if action == "descend" and has_children and depth < MAX_DEPTH - 1:
            descend_nodes.append(item)
            normalized_actions.append({
                "pathway_id": pathway_id,
                "action": "descend",
                "reason": reason,
            })
        elif action == "descend":
            kept_nodes.append(item)
            normalized_actions.append({
                "pathway_id": pathway_id,
                "action": "keep",
                "reason": reason,
            })
        elif action == "keep":
            kept_nodes.append(item)
            normalized_actions.append({
                "pathway_id": pathway_id,
                "action": "keep",
                "reason": reason,
            })
        else:
            reselect_nodes.append(item)
            normalized_actions.append({
                "pathway_id": pathway_id,
                "action": "reselect",
                "reason": reason,
            })

    return kept_nodes, descend_nodes, reselect_nodes, normalized_actions




def _compact_iteration_record(
    depth: int,
    approved_selection_set: List[Dict[str, Any]],
    kept_nodes: List[Dict[str, Any]],
    descend_nodes: List[Dict[str, Any]],
    reselect_nodes: List[Dict[str, Any]],
    normalized_actions: List[Dict[str, Any]],
    final_attempt: Dict[str, Any],
) -> Dict[str, Any]:
    critic_bundle = final_attempt.get("critic_result", {}) if isinstance(final_attempt, dict) else {}
    critic_json = critic_bundle.get("critic_result", {}) if isinstance(critic_bundle, dict) else {}

    return {
        "depth": depth,
        "final_selected_pathways": approved_selection_set,
        "critic_pathway_actions": normalized_actions,
    }


# =========================
# Actor session state helpers
# =========================
def build_actor_initial_payload(
    branch_name: str,
    drug_name: str,
    depth: int,
    artifact: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    previous_critic: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "drug_name": drug_name,
        "depth": depth,
        "artifact": artifact,
        "candidates": candidates,
        "previous_critic_suggestion": previous_critic or {},
    }


def build_actor_reselect_payload(
    branch_name: str,
    drug_name: str,
    depth: int,
    previous_actor_result: Dict[str, Any],
    previous_sanitized_selection: List[Dict[str, Any]],
    critic_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    fixed_selected_pathways = critic_feedback.get("fixed_selected_pathways", [])
    reselect_pathways = critic_feedback.get("reselect_pathways", [])

    return {
        "message_type": "reselect_request",
        "fixed_selected_pathways": fixed_selected_pathways,
        "reselect_pathways": reselect_pathways,
        "instruction": "Keep fixed_selected_pathways unchanged. Replace only reselect_pathways. Return only replacement pathways in selected_pathways. JSON only.",
    }


# =========================
# Branch actors
# =========================
def run_plain_actor_no_tools(
    branch_name: str,
    drug_name: str,
    depth: int,
    artifact: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    previous_critic: Optional[Dict[str, Any]],
    model: str,
    session_state: Optional[Dict[str, Any]] = None,
    is_reselect: bool = False,
    previous_actor_result: Optional[Dict[str, Any]] = None,
    previous_sanitized_selection: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    session_state = session_state or {}

    if not is_reselect or not session_state.get("initialized"):
        payload = build_actor_initial_payload(
            branch_name=branch_name,
            drug_name=drug_name,
            depth=depth,
            artifact=artifact,
            candidates=candidates,
            previous_critic=previous_critic,
        )
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": get_actor_prompt(branch_name)},
                {"role": "user", "content": compact_json_dumps(payload)},
            ],
            max_output_tokens=1800,
        )
    else:
        payload = build_actor_reselect_payload(
            branch_name=branch_name,
            drug_name=drug_name,
            depth=depth,
            previous_actor_result=previous_actor_result or {},
            previous_sanitized_selection=previous_sanitized_selection or [],
            critic_feedback=previous_critic or {},
        )
        response = openai_client.responses.create(
            model=model,
            previous_response_id=session_state["last_response_id"],
            input=[
                {"role": "user", "content": compact_json_dumps(payload)},
            ],
            max_output_tokens=1800,
        )

    text = (response.output_text or "").strip()
    actor_json = parse_json_response(text)

    new_session_state = {
        "initialized": True,
        "last_response_id": response.id,
    }

    return {
        "actor_model": model,
        "actor_raw_output_text": text,
        "actor_result": actor_json,
        "session_state": new_session_state,
    }


def run_branch_actor_once(
    branch_name: str,
    drug_name: str,
    depth: int,
    artifact: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    previous_critic: Optional[Dict[str, Any]],
    model: str,
    session_state: Optional[Dict[str, Any]] = None,
    is_reselect: bool = False,
    previous_actor_result: Optional[Dict[str, Any]] = None,
    previous_sanitized_selection: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if branch_name not in {"pathway_moa", "gene"}:
        raise ValueError(f"Unknown branch: {branch_name}")
    return run_plain_actor_no_tools(
        branch_name=branch_name,
        drug_name=drug_name,
        depth=depth,
        artifact=artifact,
        candidates=candidates,
        previous_critic=previous_critic,
        model=model,
        session_state=session_state,
        is_reselect=is_reselect,
        previous_actor_result=previous_actor_result,
        previous_sanitized_selection=previous_sanitized_selection,
    )


# =========================
# Critics
# =========================
def compress_recent_critic_history(iterations: List[Dict[str, Any]], keep_last_n: int = 2) -> List[Dict[str, Any]]:
    recent = iterations[-keep_last_n:]
    out = []
    for item in recent:
        out.append({
            "depth": item.get("depth"),
            "pathway_actions": item.get("critic_pathway_actions", []),
        })
    return out


def run_branch_critic_once(
    branch_name: str,
    drug_name: str,
    depth: int,
    artifact: Dict[str, Any],
    current_candidates: List[Dict[str, Any]],
    actor_result: Dict[str, Any],
    selected_pathway_child_candidates: List[Dict[str, Any]],
    recent_critic_history: List[Dict[str, Any]],
    has_children: bool,
    model: str,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    critic_payload = {
        "message_type": "critic_evaluate_attempt",
        "drug_name": drug_name,
        "branch": branch_name,
        "depth": depth,
        "artifact": artifact,
        "current_candidates": current_candidates,
        "actor_result": actor_result,
        "selected_pathway_child_candidates": selected_pathway_child_candidates,
        "recent_critic_history": recent_critic_history,
        "has_children": has_children,
        "instruction": (
            "Evaluate the actor's currently selected pathways using the full current-layer candidate pool "
            "and the child candidates shown for those selected pathways. "
            "Return JSON only."
        ),
    }
    critic_messages = [{
        "role": "user",
        "content": compact_json_dumps(critic_payload),
    }]

    msg = anthropic_client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=0,
        system=get_critic_prompt(branch_name),
        messages=critic_messages,
    )
    text = _extract_text_from_anthropic_message(msg)
    critic_json = parse_json_response(text)

    return {
        "critic_model": model,
        "critic_raw_output_text": text,
        "critic_result": critic_json,
        "session_state": {},
    }


# =========================
# One branch hierarchical loop
# =========================
def run_hierarchical_branch_loop(
    drug_name: str,
    branch_name: str,
    branch_artifact: Dict[str, Any],
    pathways_df: pd.DataFrame,
    children_map: Dict[str, List[str]],
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> Dict[str, Any]:
    iterations: List[Dict[str, Any]] = []

    active_parent_nodes: List[Dict[str, Any]] = []
    final_selections: List[Dict[str, Any]] = []

    # Parents that were selected at the previous depth and intentionally withheld
    # because the algorithm attempted to refine them into child pathways.
    rollback_parent_nodes: List[Dict[str, Any]] = []

    for depth in range(MAX_DEPTH):
        if depth == 0:
            raw_candidates = get_layer0_candidates(pathways_df)
        else:
            all_children = []
            seen_child_ids = set()

            for parent in active_parent_nodes:
                parent_id = parent["PathwayID"]
                child_candidates = get_child_candidates(
                    pathways_df,
                    children_map,
                    parent_id,
                )

                for child in child_candidates:
                    cid = child["PathwayID"]
                    if cid not in seen_child_ids:
                        all_children.append(child)
                        seen_child_ids.add(cid)

            raw_candidates = all_children

        # If no candidates are available at this depth, restore all parents
        # that were being refined.
        if not raw_candidates:
            if rollback_parent_nodes:
                rollback_to_keep = [
                    p for p in rollback_parent_nodes
                    if int(p.get("Layer", 0)) > 0
                ]

                final_selections = merge_pathway_lists(
                    final_selections,
                    rollback_to_keep,
                )
            break

        candidates = raw_candidates

        current_attempts = []
        approved_selection_set: List[Dict[str, Any]] = []
        previous_critic_for_actor: Optional[Dict[str, Any]] = None
        actor_session_state: Dict[str, Any] = {}
        critic_session_state: Dict[str, Any] = {}
        last_actor_json: Optional[Dict[str, Any]] = None
        last_actor_selected: List[Dict[str, Any]] = []

        for attempt in range(1, MAX_RESELECT_PER_DEPTH + 2):
            is_reselect = attempt > 1
            fixed_selected_pathways = []
            num_replacements_needed = MAX_PATHWAYS_PER_STEP
            if is_reselect and isinstance(previous_critic_for_actor, dict):
                fixed_selected_pathways = previous_critic_for_actor.get("fixed_selected_pathways", []) or []
                reselect_pathways = previous_critic_for_actor.get("reselect_pathways", []) or []
                num_replacements_needed = len(reselect_pathways)

            actor_bundle = run_branch_actor_once(
                branch_name=branch_name,
                drug_name=drug_name,
                depth=depth,
                artifact=branch_artifact,
                candidates=candidates,
                previous_critic=previous_critic_for_actor,
                model=actor_model,
                session_state=actor_session_state,
                is_reselect=is_reselect,
                previous_actor_result=last_actor_json,
                previous_sanitized_selection=last_actor_selected,
            )
            actor_session_state = actor_bundle.get(
                "session_state",
                actor_session_state,
            )

            actor_json = actor_bundle["actor_result"]
            actor_selected_partial = sanitize_selected_pathways(
                actor_json.get("selected_pathways", []),
                candidates=candidates,
                max_items=num_replacements_needed if is_reselect else MAX_PATHWAYS_PER_STEP,
            )
            if is_reselect:
                actor_selected = merge_pathway_lists(
                    fixed_selected_pathways,
                    actor_selected_partial,
                )
                actor_json = dict(actor_json)
                actor_json["selected_pathways"] = convert_sanitized_pathways_to_actor_schema(
                    actor_selected,
                )
                actor_bundle["actor_result"] = actor_json
            else:
                actor_selected = actor_selected_partial

            selected_pathway_child_candidates = build_selected_pathway_child_candidates(
                selected_pathways=actor_selected,
                pathways_df=pathways_df,
                children_map=children_map,
            )

            has_any_child = any(
                children_map.get(item["PathwayID"], [])
                for item in actor_selected
            )

            critic_bundle = run_branch_critic_once(
                branch_name=branch_name,
                drug_name=drug_name,
                depth=depth,
                artifact=branch_artifact,
                current_candidates=candidates,
                actor_result=actor_json,
                selected_pathway_child_candidates=selected_pathway_child_candidates,
                recent_critic_history=compress_recent_critic_history(
                    iterations,
                    keep_last_n=MAX_RECENT_CRITIC_HISTORY,
                ),
                has_children=has_any_child,
                model=critic_model,
                session_state=critic_session_state,
            )
            critic_session_state = critic_bundle.get(
                "session_state",
                critic_session_state,
            )

            critic_json = critic_bundle["critic_result"]
            kept_nodes, descend_nodes, reselect_nodes, normalized_actions = _split_pathways_by_critic_actions(
                critic_json=critic_json,
                selected_pathways=actor_selected,
                children_map=children_map,
                depth=depth,
            )
            approved_set = kept_nodes + descend_nodes
            reselect_required = len(reselect_nodes) > 0

            attempt_record = {
                "attempt_index": attempt,
                "actor_result": actor_bundle,
                "critic_result": critic_bundle,
                "actor_selected_pathways": actor_selected,
                "critic_approved_pathways": approved_set,
                "critic_kept_pathways": kept_nodes,
                "critic_descend_pathways": descend_nodes,
                "critic_reselect_pathways": reselect_nodes,
                "critic_pathway_actions": normalized_actions,
                "critic_reselect_required": reselect_required,
            }
            current_attempts.append(attempt_record)

            approved_selection_set = approved_set
            last_actor_json = actor_json
            last_actor_selected = actor_selected

            if not reselect_required:
                break

            previous_critic_for_actor = {
                "depth": depth,
                "pathway_actions": critic_json.get("pathway_actions", []),
                "fixed_selected_pathways": approved_set,
                "reselect_pathways": reselect_nodes,
            }

        # If actor/critic failed to produce a valid approved set,
        # restore the parents that were being refined.
        if not current_attempts or not approved_selection_set:
            if rollback_parent_nodes:
                rollback_to_keep = [
                    p for p in rollback_parent_nodes
                    if int(p.get("Layer", 0)) > 0
                ]

                final_selections = merge_pathway_lists(
                    final_selections,
                    rollback_to_keep,
                )
            break

        final_attempt = current_attempts[-1]

        stopped_here_nodes = final_attempt.get("critic_kept_pathways", [])
        next_active_nodes = final_attempt.get("critic_descend_pathways", [])
        reselect_nodes = final_attempt.get("critic_reselect_pathways", [])
        normalized_actions = final_attempt.get("critic_pathway_actions", [])

        step_record = _compact_iteration_record(
            depth=depth,
            approved_selection_set=approved_selection_set,
            kept_nodes=stopped_here_nodes,
            descend_nodes=next_active_nodes,
            reselect_nodes=reselect_nodes,
            normalized_actions=normalized_actions,
            final_attempt=final_attempt,
        )
        iterations.append(step_record)

        # If the previous depth had multiple parents being refined,
        # check each parent separately. If none of the currently selected
        # pathways is a descendant of a parent, keep that parent.
        if depth > 0 and rollback_parent_nodes:
            uncovered_parents = rollback_uncovered_parents(
                parent_nodes=[
                    p for p in rollback_parent_nodes
                    if int(p.get("Layer", 0)) > 0
                ],
                child_nodes=approved_selection_set,
                children_map=children_map,
            )
            final_selections = merge_pathway_lists(
                final_selections,
                uncovered_parents,
            )

        final_selections = merge_pathway_lists(
            final_selections,
            stopped_here_nodes,
        )

        if not next_active_nodes:
            rollback_parent_nodes = []
            break

        rollback_parent_nodes = list(next_active_nodes)
        active_parent_nodes = next_active_nodes

    return {
        "branch": branch_name,
        "status": "completed",
        "iterations": iterations,
        "final_selections": final_selections,
    }



# =========================
# Dual-branch stage-2 runner
# =========================
def run_dual_branch_stage2_for_file(
    source_json_path: Path,
    pathways_csv: Path = DEFAULT_ALL_PATHWAYS_CSV,
    relation_txt: Path = DEFAULT_RELATION_TXT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> Dict[str, Any]:
    stage1_result = load_stage1_result(source_json_path)

    pathway_moa_branch = stage1_result.get("pathway_moa", {})
    gene_branch = stage1_result.get("gene", {})

    drug_name = None
    if isinstance(pathway_moa_branch, dict):
        drug_name = pathway_moa_branch.get("drug_name")
    if not drug_name and isinstance(gene_branch, dict):
        drug_name = gene_branch.get("drug_name")
    if not drug_name:
        raise ValueError(
            f"Missing branch-level drug_name in {source_json_path}. "
            "Expected pathway_moa.drug_name or gene.drug_name."
        )

    pathway_final_artifact = extract_final_artifact_only(stage1_result, "pathway_moa")
    gene_final_artifact = extract_final_artifact_only(stage1_result, "gene")

    pathways_df = load_pathways_df(pathways_csv)
    children_map, _ = load_relations(relation_txt)

    pathway_branch_result = run_hierarchical_branch_loop(
        drug_name=drug_name,
        branch_name="pathway_moa",
        branch_artifact=pathway_final_artifact,
        pathways_df=pathways_df,
        children_map=children_map,
        actor_model=actor_model,
        critic_model=critic_model,
    )

    gene_branch_result = run_hierarchical_branch_loop(
        drug_name=drug_name,
        branch_name="gene",
        branch_artifact=gene_final_artifact,
        pathways_df=pathways_df,
        children_map=children_map,
        actor_model=actor_model,
        critic_model=critic_model,
    )

    selected_outputs = {
        "pathway_moa": build_selected_outputs(pathway_branch_result),
        "gene": build_selected_outputs(gene_branch_result),
    }
    saved_result = {
        "drug_name": drug_name,
        **selected_outputs,
    }

    out_path = output_dir / f"{slugify(drug_name)}.json"
    save_json(saved_result, out_path)
    return {
        "status": "completed",
        "drug_name": drug_name,
        "selected_outputs": selected_outputs,
        "json_path": str(out_path),
    }


# =========================
# Batch runner
# =========================
def run_batch(
    input_dir: Path = DEFAULT_INPUT_JSON_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    skip_existing: bool = True,
    actor_model: str = DEFAULT_ACTOR_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))

    results: List[Dict[str, Any]] = []

    for idx, json_path in enumerate(json_files, start=1):
        drug_name = json_path.stem
        out_path = output_dir / f"{json_path.stem}.json"

        if skip_existing and out_path.exists():
            print(f"[{idx}/{len(json_files)}] Skip existing: {drug_name}")
            saved_payload = json.loads(out_path.read_text(encoding="utf-8"))
            results.append({
                "status": "completed",
                "drug_name": saved_payload.get("drug_name", drug_name),
                "selected_outputs": {
                    "pathway_moa": saved_payload.get("pathway_moa", []),
                    "gene": saved_payload.get("gene", []),
                },
                "json_path": str(out_path),
            })
            continue

        print(f"[{idx}/{len(json_files)}] Start: {drug_name}")

        try:
            result = run_dual_branch_stage2_for_file(
                source_json_path=json_path,
                output_dir=output_dir,
                actor_model=actor_model,
                critic_model=critic_model,
            )
            print(
                f"[{idx}/{len(json_files)}] Done: {drug_name} | "
                f"pathway_moa={len(result['selected_outputs'].get('pathway_moa', []))} terminal pathways | "
                f"gene={len(result['selected_outputs'].get('gene', []))} terminal pathways"
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "drug_name": drug_name,
                "source_json": str(json_path),
                "error": str(exc),
            }
            save_json(result, out_path)
            print(f"[{idx}/{len(json_files)}] Failed: {drug_name} | {exc}")

        results.append(result)

    return results


if __name__ == "__main__":
    run_batch()
