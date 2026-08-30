from __future__ import annotations

import requests
from xml.etree import ElementTree

def _safe_text(element):
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def get_pubmed_articles(term, max_results=5, timeout=20):
    if not isinstance(term, str) or not term.strip():
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": "term is empty or invalid.",
        }

    base_url_pubmed = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = f"{base_url_pubmed}/esearch.fcgi"
    fetch_url = f"{base_url_pubmed}/efetch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "xml",
        "retmax": str(max_results),
        "sort": "relevance"
    }
    try:
        search_response = requests.get(search_url, params=search_params, timeout=timeout)
        search_response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": f"PubMed search failed: {exc}",
        }

    try:
        search_results = ElementTree.fromstring(search_response.content)
        id_list = [id_tag.text for id_tag in search_results.findall('.//Id')]
    except ElementTree.ParseError as e:
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": f"Error parsing PubMed search results: {e}",
        }
    
    if not id_list:
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": "No articles found for the query.",
        }
    
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    try:
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=timeout)
        fetch_response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": f"PubMed fetch failed: {exc}",
        }
    
    try:
        articles = ElementTree.fromstring(fetch_response.content)
    except ElementTree.ParseError as e:
        return {
            "term": term,
            "num_articles": 0,
            "articles": [],
            "message": f"Error parsing PubMed fetch results: {e}",
        }

    results = []
    for article in articles.findall('.//PubmedArticle'):
        pmid = _safe_text(article.find('.//PMID'))
        title = _safe_text(article.find('.//ArticleTitle'))
        abstract_parts = article.findall('.//Abstract/AbstractText')
        abstract_text = " ".join(_safe_text(part) for part in abstract_parts if _safe_text(part))
        results.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text,
            }
        )

    return {
        "term": term,
        "num_articles": len(results),
        "articles": results,
        "message": "Matched successfully." if results else "No abstract available for fetched articles.",
    }


get_pubmed_articles_doc = {
    "type": "function",
    "name": "get_pubmed_articles",
    "description": "Given a query, return related PubMed articles with PMID, title, and abstract.",
    "parameters": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "a query to search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of PubMed articles to fetch. Default is 5.",
            },
        },
        "required": ["term"],
    },
}  
