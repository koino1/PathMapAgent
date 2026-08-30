from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json
import re
import sys

from Tool.common import compact_records, normalize_text


# GeneAgent is vendored with this project, so its location does not depend on
# the machine or the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENEAGENT_ROOT = PROJECT_ROOT / "GeneAgent"

if str(GENEAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GENEAGENT_ROOT))

from config import configure_openai, get_model_name, get_openai_client, message_to_dict
from worker import AgentPhD


configure_openai()

MODEL_NAME = get_model_name()

_CLAIM_PATTERN = re.compile(r"^[a-zA-Z0-9,.;?!*()_-]+$")
_TEXT_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

SYSTEM = "You are an efficient and insightful assistant to a molecular biologist."
SYSTEM_VERIFY = "You are a helpful and objective fact-checker to verify the summary of gene set."

TOPIC_INSTRUCTION = """
Only generate claims with affirmative sentence for the entire gene set.
The gene set should only be separated by comma, e.g., "a,b,c".
Don't generate claims for the single gene or incomplete gene set.
Don't generate hypotheis claims over the previous analysis.
Please replace the statement like 'these genes', 'this system' with the core genes in the given gene set.
"""

ANALYSIS_INSTRUCTION = """
Generate claims for genes and their biological functions around the updated process name.
Don't generate claims for the entire gene set or 'this system'.
Don't generate unworthy claims such as the summarization and reasoning over the previous analysis.
Claims must contain the gene names and their biological process functions.
"""

MODIFICATION_INSTRUCTION = """
Put the updated process name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens. All modified sentence should encoded into utf-8.
Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
Be factual, do not editorialize.
You must retain the gene names of each updated biological functions in the new summary.
"""

SUMMARIZATION_INSTRUCTION = """
If the analytical narratives of genes can't directly support or related to the updated process name, you must propose a new brief biological process name from the analytical texts.
Otherwise, you must retain the updated process name and only can make a grammar revision.
IF the claim is supported, you must complement the narratives by using the standard evidence of gene set functions (or gene summaries) in the verification report but don't change the updated process name.
IF the claim is not supported, do not mention any statement like "... was not directly confirmed by..."
Be concise, do not use unnecessary format like **, only return the concise texts.
"""

REPOSITORIES = [
    "get_complex_for_gene_set",
    "get_disease_for_single_gene",
    "get_domain_for_single_gene",
    "get_enrichment_for_gene_set",
    "get_pathway_for_gene_set",
    "get_interactions_for_gene_set",
    "get_gene_summary_for_single_gene",
    "get_pubmed_articles",
]


def _baseline_prompt(genes: str) -> str:
    return f"""
Write a critical analysis of the biological processes performed by this system of interacting proteins.
Propose a brief name for the most prominent biological process performed by the system.
Put the name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens.
Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
Be factual, do not editorialize.
For each important point, describe your reasoning and supporting information.
For each biological function name, show the corresponding gene names.
Here is the gene set: {genes}
"""


def _topic_prompt(genes: str, process: str) -> str:
    return f"""
Here is the original process name for the gene set {genes}:\n{process}
However, the process name might be false. Please generate decontextualized claims for the process name that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
"""


def _analysis_prompt(summary: str) -> str:
    return f"""
Here is the summary of the given gene set: \n{summary}
However, the gene analysis in the summary might not support the updated process name.
Please generate several decontextualized claims for the analytical narratives that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
"""


def _modification_prompt(verification_topic: str) -> str:
    return f"""
I have finished the verification for process name. Here is the verification report:\n{verification_topic}
You should only consider the successfully verified claims.
If claims are supported, you should retain the original process name and only can make a minor grammar revision.
if claims are partially supported, you should discard the unsupported part.
If claims are refuted, you must replace the original process name with the most significant (i.e., top-1) biological function term summarized from the verification report.
Meanwhile, revise the original summaries using the verified (or updated) process name. Do not use sentence like "There are no direct evidence to..."
"""


def _summarization_prompt(verification_analysis: str) -> str:
    return f"""
I have finished the verification for the revised summary. Here is the verification report:\n{verification_analysis}
Please modify the summary according to the verification report again.
"""


def _normalize_gene_set(gene_set: Any) -> str:
    if isinstance(gene_set, str):
        raw = gene_set.strip()
        if not raw:
            return ""
        raw = raw.replace(";", ",").replace("/", ",").replace("|", ",")
        if "," in raw:
            parts = [item.strip() for item in raw.split(",") if item.strip()]
            return ",".join(parts)
        return ",".join(part for part in raw.split() if part.strip())

    if isinstance(gene_set, Sequence):
        parts = [str(item).strip() for item in gene_set if str(item).strip()]
        return ",".join(parts)

    return ""


def _safe_claim_text(text: str) -> str:
    if _CLAIM_PATTERN.match(text):
        return text
    return re.sub(r"[^a-zA-Z0-9,.;?!*()_-]+$", "_", text)


def _safe_summary_text(text: str) -> str:
    if _TEXT_PATTERN.match(text):
        return text
    return re.sub(r"[^a-zA-Z0-9-_]+", "_", text)


def _extract_process_name(summary: str) -> str:
    first_line = (summary or "").split("\n")[0]
    if first_line.startswith("Process: "):
        return first_line.split("Process: ", 1)[1].strip()
    return ""


@lru_cache(maxsize=1)
def _get_agent() -> AgentPhD:
    return AgentPhD(function_names=REPOSITORIES)


def _verify_claims(claims: List[str]) -> Dict[str, Any]:
    agent = _get_agent()
    records: List[Dict[str, Any]] = []
    report_parts: List[str] = []

    for claim in claims:
        cleaned_claim = _safe_claim_text(claim)
        verification = agent.inference(cleaned_claim)
        records.append(
            {
                "claim": cleaned_claim,
                "verification": verification,
            }
        )
        report_parts.append(f"Original_claim:{cleaned_claim}Verified_claim:{verification}")

    return {
        "records": compact_records(records),
        "report_text": "".join(report_parts),
    }


def _build_match_record(
    normalized_gene_set: str,
    initial_summary: str,
    topic_claims: List[str],
    topic_verification: List[Dict[str, Any]],
    updated_summary: str,
    analysis_claims: List[str],
    analysis_verification: List[Dict[str, Any]],
    final_description: str,
) -> Dict[str, Any]:
    return compact_records(
        [
            {
                "normalized_gene_set": normalized_gene_set,
                "initial_process_name": _extract_process_name(initial_summary),
                "initial_summary": initial_summary,
                "topic_claims": topic_claims,
                "topic_verification": topic_verification,
                "updated_summary": updated_summary,
                "analysis_claims": analysis_claims,
                "analysis_verification": analysis_verification,
                "final_description": final_description,
            }
        ]
    )[0]


def get_geneagent_geneset_info(gene_set: Any) -> Dict[str, Any]:
    """
    Given a gene set, return a GeneAgent-based biological process description.
    """
    normalized_gene_set = _normalize_gene_set(gene_set)
    if not normalized_gene_set:
        return {
            "gene_set": gene_set,
            "final_description": "",
            "message": "gene_set is empty or invalid.",
        }

    try:
        client = get_openai_client()
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _baseline_prompt(normalized_gene_set)},
        ]

        initial_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        initial_message = initial_completion.choices[0].message
        initial_summary = initial_message.content or ""
        messages.append(message_to_dict(initial_message))

        topic_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFY},
                {
                    "role": "user",
                    "content": _topic_prompt(normalized_gene_set, _extract_process_name(initial_summary)) + TOPIC_INSTRUCTION,
                },
            ],
            temperature=0,
        )
        topic_claims = json.loads(topic_completion.choices[0].message.content or "[]")
        topic_verification = _verify_claims(topic_claims)

        messages.append(
            {
                "role": "user",
                "content": _modification_prompt(topic_verification["report_text"]) + MODIFICATION_INSTRUCTION,
            }
        )
        updated_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        updated_message = updated_completion.choices[0].message
        updated_summary = updated_message.content or ""
        messages.append(message_to_dict(updated_message))

        analysis_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFY},
                {
                    "role": "user",
                    "content": _analysis_prompt(_safe_summary_text(updated_summary)) + ANALYSIS_INSTRUCTION,
                },
            ],
            temperature=0,
        )
        analysis_claims = json.loads(analysis_completion.choices[0].message.content or "[]")
        analysis_verification = _verify_claims(analysis_claims)

        messages.append(
            {
                "role": "assistant",
                "content": _summarization_prompt(analysis_verification["report_text"]) + SUMMARIZATION_INSTRUCTION,
            }
        )
        final_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        final_description = final_completion.choices[0].message.content or ""

        return {
            "gene_set": gene_set,
            "final_description": final_description,
            "message": "Matched successfully." if final_description.strip() else "GeneAgent returned an empty description.",
        }
    except Exception as e:
        return {
            "gene_set": gene_set,
            "final_description": "",
            "message": f"GeneAgent processing failed: {e}",
        }


get_geneagent_geneset_info_doc = {
    "type": "function",
    "name": "get_geneagent_geneset_info",
    "description": (
        "Given a gene set, return a GeneAgent-generated final description of the shared biological "
        "process. The result includes gene_set, final_description, and message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": (
                    "A gene set to analyze. Genes can be separated by spaces, commas, semicolons, pipes, or slashes."
                ),
            },
        },
        "required": ["gene_set"],
    },
}
