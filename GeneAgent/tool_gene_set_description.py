from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence
import json
import re

from config import configure_openai, get_model_name, get_openai_client, message_to_dict
from worker import AgentPhD


configure_openai()

MODEL_NAME = get_model_name()

_CLAIM_PATTERN = re.compile(r"^[a-zA-Z0-9,.;?!*()_-]+$")
_TEXT_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


SYSTEM = "You are an efficient and insightful assistant to a molecular biologist."
SYSTEM_VERIFY = "You are a helpful and objective fact-checker to verify the summary of gene set."


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


TOPIC_INSTRUCTION = """
Only generate claims with affirmative sentence for the entire gene set.
The gene set should only be separated by comma, e.g., "a,b,c".
Don't generate claims for the single gene or incomplete gene set.
Don't generate hypotheis claims over the previous analysis.
Please replace the statement like 'these genes', 'this system' with the core genes in the given gene set.
"""


def _analysis_prompt(summary: str) -> str:
    return f"""
Here is the summary of the given gene set: \n{summary}
However, the gene analysis in the summary might not support the updated process name.
Please generate several decontextualized claims for the analytical narratives that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
"""


ANALYSIS_INSTRUCTION = """
Generate claims for genes and their biological functions around the updated process name.
Don't generate claims for the entire gene set or 'this system'.
Don't generate unworthy claims such as the summarization and reasoning over the previous analysis.
Claims must contain the gene names and their biological process functions.
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


MODIFICATION_INSTRUCTION = """
Put the updated process name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens. All modified sentence should encoded into utf-8.
Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
Be factual, do not editorialize.
You must retain the gene names of each updated biological functions in the new summary.
"""


def _summarization_prompt(verification_analysis: str) -> str:
    return f"""
I have finished the verification for the revised summary. Here is the verification report:\n{verification_analysis}
Please modify the summary according to the verification report again.
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


def _normalize_gene_set(gene_set: Any) -> str:
    if isinstance(gene_set, str):
        genes = gene_set.strip()
        if not genes:
            return ""
        return genes.replace("/", ",").replace(" ", ",")

    if isinstance(gene_set, Sequence):
        genes = [str(gene).strip() for gene in gene_set if str(gene).strip()]
        return ",".join(genes)

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


def _run_claim_verification(claims: List[str]) -> Dict[str, Any]:
    agent = _get_agent()
    verification_text_parts: List[str] = []
    verification_records: List[Dict[str, str]] = []

    for claim in claims:
        safe_claim = _safe_claim_text(claim)
        verification = agent.inference(safe_claim)
        verification_text_parts.append(f"Original_claim:{safe_claim}Verified_claim:{verification}")
        verification_records.append(
            {
                "claim": safe_claim,
                "verification": verification,
            }
        )

    return {
        "records": verification_records,
        "verification_text": "".join(verification_text_parts),
    }


def get_gene_set_description(gene_set: Any) -> Dict[str, Any]:
    """
    Given a gene set, return a GeneAgent-style description of the shared
    biological process, including intermediate verification artifacts.
    """
    normalized_gene_set = _normalize_gene_set(gene_set)
    if not normalized_gene_set:
        return {
            "gene_set": gene_set,
            "source_of_evidence": "GeneAgent",
            "model": MODEL_NAME,
            "status": "error",
            "message": "gene_set is empty or invalid.",
        }

    client = get_openai_client()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _baseline_prompt(normalized_gene_set)},
    ]

    try:
        initial_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        initial_message = initial_completion.choices[0].message
        initial_summary = initial_message.content or ""
        messages.append(message_to_dict(initial_message))

        initial_process = _extract_process_name(initial_summary)

        topic_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFY},
                {
                    "role": "user",
                    "content": _topic_prompt(normalized_gene_set, initial_process) + TOPIC_INSTRUCTION,
                },
            ],
            temperature=0,
        )
        topic_claims = json.loads(topic_completion.choices[0].message.content or "[]")
        topic_verification = _run_claim_verification(topic_claims)

        messages.append(
            {
                "role": "user",
                "content": _modification_prompt(topic_verification["verification_text"]) + MODIFICATION_INSTRUCTION,
            }
        )
        updated_topic_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        updated_topic_message = updated_topic_completion.choices[0].message
        updated_summary = updated_topic_message.content or ""
        messages.append(message_to_dict(updated_topic_message))

        safe_updated_summary = _safe_summary_text(updated_summary)
        analysis_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFY},
                {
                    "role": "user",
                    "content": _analysis_prompt(safe_updated_summary) + ANALYSIS_INSTRUCTION,
                },
            ],
            temperature=0,
        )
        analysis_claims = json.loads(analysis_completion.choices[0].message.content or "[]")
        analysis_verification = _run_claim_verification(analysis_claims)

        messages.append(
            {
                "role": "assistant",
                "content": _summarization_prompt(analysis_verification["verification_text"]) + SUMMARIZATION_INSTRUCTION,
            }
        )
        final_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        final_description = final_completion.choices[0].message.content or ""

        return {
            "gene_set": normalized_gene_set,
            "source_of_evidence": "GeneAgent",
            "model": MODEL_NAME,
            "status": "success",
            "initial_process_name": initial_process,
            "initial_summary": initial_summary,
            "topic_claims": topic_claims,
            "topic_verification": topic_verification["records"],
            "updated_summary": updated_summary,
            "analysis_claims": analysis_claims,
            "analysis_verification": analysis_verification["records"],
            "final_description": final_description,
            "message": "Gene set description generated successfully.",
        }
    except Exception as e:
        return {
            "gene_set": normalized_gene_set,
            "source_of_evidence": "GeneAgent",
            "model": MODEL_NAME,
            "status": "error",
            "message": f"Failed to generate gene set description: {e}",
        }


get_gene_set_description_doc = {
    "type": "function",
    "name": "get_gene_set_description",
    "description": (
        "Given a gene set, return a GeneAgent-style description of the shared biological process "
        "with intermediate verification results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": (
                    "A gene set to analyze. Genes can be separated by spaces, commas, or slashes, "
                    'for example, "TP53 BRCA1 BRCA2 ATM".'
                ),
            },
        },
        "required": ["gene_set"],
    },
}
