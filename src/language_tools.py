"""
Azure AI Language SDK tool implementations.

Used when LANGUAGE_TOOL_MODE=sdk (the default).
The agent declares these as function tools; the app intercepts required_action
runs, executes the appropriate function here, and submits the output back.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential

# Pending file-uploaded text documents (legacy flow)
_pending_documents: list[str] | None = None

# Pending structured survey rows (new flow)
_pending_structured_rows: list[dict[str, Any]] | None = None


def set_pending_documents(documents: list[str]) -> None:
    """Store uploaded text documents for the next legacy tool call."""
    global _pending_documents
    _pending_documents = [str(d) for d in documents]


def _take_pending_documents() -> list[str] | None:
    """Return and clear pending uploaded text documents."""
    global _pending_documents
    docs = _pending_documents
    _pending_documents = None
    return docs


def set_pending_structured_rows(rows: list[dict[str, Any]]) -> None:
    """Store uploaded structured survey rows for the next tool call."""
    global _pending_structured_rows
    _pending_structured_rows = [dict(r) for r in rows]


def _take_pending_structured_rows() -> list[dict[str, Any]] | None:
    """Return and clear pending uploaded structured survey rows."""
    global _pending_structured_rows
    rows = _pending_structured_rows
    _pending_structured_rows = None
    return rows


@lru_cache(maxsize=1)
def _get_client() -> TextAnalyticsClient:
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"].rstrip("/")
    return TextAnalyticsClient(endpoint=endpoint, credential=DefaultAzureCredential())


def _docs(documents: Any) -> list[str]:
    """Accept either a JSON string or a Python list of documents."""
    if isinstance(documents, str):
        try:
            documents = json.loads(documents)
        except json.JSONDecodeError:
            documents = [documents]
    return [str(d) for d in documents]


def _rows(rows: Any) -> list[dict[str, Any]]:
    """Accept either a JSON string or a Python list of row objects."""
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except json.JSONDecodeError:
            rows = []
    if not isinstance(rows, list):
        return []

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(dict(row))
    return normalized


def _chunked(lst: list[Any], size: int = 10):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"na", "n/a", "none", "null", "nan"}


def _normalize_text(value: Any) -> str:
    return "" if _is_blank(value) else str(value).strip()


# ─── Structured survey helpers ───────────────────────────────────────────────

_RATING_SENTIMENT_MAP: dict[str, str] = {
    "totally satisfied": "Positive",
    "strongly agree": "Positive",
    "agree": "Positive",
    "very satisfied": "Positive",
    "somewhat satisfied": "Positive",
    "neither satisfied nor dissatisfied": "Neutral",
    "neither agree nor disagree": "Neutral",
    "somewhat dissatisfied": "Negative",
    "disagree": "Negative",
    "very dissatisfied": "Negative",
    "totally dissatisfied": "Negative",
    "strongly disagree": "Negative",
}

_MAIN_CLUSTERS = [
    "Product Satisfaction",
    "Tools",
    "Overall Service Delivery",
    "Technical Support Quality",
    "Communication",
    "Documentation & Knowledge base",
]


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _detect_response_type(response_value: str) -> str:
    if _normalize_for_match(response_value) in _RATING_SENTIMENT_MAP:
        return "structured_rating"
    return "free_text"


def _map_main_cluster(question_name: str, fields_name: str = "", response_value: str = "") -> str:
    source = " ".join([question_name, fields_name]).strip().lower()

    if any(k in source for k in ["overall satisfaction", "satisfied with product", "product satisfaction"]):
        return "Product Satisfaction"

    if any(k in source for k in [
        "tool",
        "portal",
        "navigation",
        "report",
        "download",
        "usability",
        "feature",
    ]):
        return "Tools"

    if any(k in source for k in [
        "technical support",
        "engineer",
        "troubleshooting",
        "escalation",
        "expertise",
        "resolution handling",
    ]):
        return "Technical Support Quality"

    if any(k in source for k in [
        "communication",
        "update",
        "status",
        "time to resolve",
        "timeliness",
        "follow-up",
        "follow up",
    ]):
        return "Communication"

    if any(k in source for k in [
        "documentation",
        "knowledge base",
        "kb",
        "guide",
        "article",
        "self-service",
        "manual",
    ]):
        return "Documentation & Knowledge base"

    if any(k in source for k in ["service received", "case experience", "ease of resolving", "service delivery"]):
        return "Overall Service Delivery"

    # Fallback signal from response text when question text is empty/weak.
    rv = response_value.lower()
    if any(k in rv for k in ["documentation", "knowledge base", "article"]):
        return "Documentation & Knowledge base"
    if any(k in rv for k in ["tool", "portal", "dashboard", "report"]):
        return "Tools"
    if any(k in rv for k in ["engineer", "support", "troubleshoot"]):
        return "Technical Support Quality"
    if any(k in rv for k in ["communication", "update", "response time"]):
        return "Communication"

    return "Overall Service Delivery"


def _normalize_sentiment_label(value: str) -> str:
    low = value.strip().lower()
    if low == "positive":
        return "Positive"
    if low == "neutral":
        return "Neutral"
    if low == "negative":
        return "Negative"
    if low == "mixed":
        return "Neutral"
    return "Neutral"


def _determine_survey_sentiment(row_sentiments: list[str]) -> str:
    score = 0
    for s in row_sentiments:
        if s == "Positive":
            score += 1
        elif s == "Negative":
            score -= 1

    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100, 1) if total else 0.0


# ─── New structured survey tool ──────────────────────────────────────────────

def analyze_structured_survey(
    rows: Any = None,
    include_key_phrases: bool = False,
    include_entities: bool = False,
) -> str:
    """Analyze structured survey rows and return row-level enrichment + lightweight aggregates.

    Input rows must contain mapped fields:
      - survey_response_id
      - question_name
      - response_value
      - fields_name (optional)

    If rows is omitted, pending uploaded rows are used.
    """
    client = _get_client()

    source_rows = _rows(rows) if rows is not None else (_take_pending_structured_rows() or [])
    total_rows = len(source_rows)
    if not source_rows:
        return json.dumps({"error": "No structured survey rows provided."}, ensure_ascii=False)

    eligible_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        response_value = _normalize_text(row.get("response_value", ""))
        if _is_blank(response_value):
            continue

        eligible_rows.append(
            {
                "row_index": idx,
                "survey_response_id": _normalize_text(row.get("survey_response_id", "")),
                "question_name": _normalize_text(row.get("question_name", "")),
                "response_value": response_value,
                "fields_name": _normalize_text(row.get("fields_name", "")),
            }
        )

    if not eligible_rows:
        return json.dumps(
            {
                "coverage": {
                    "total_rows": total_rows,
                    "eligible_rows": 0,
                    "processed_rows": 0,
                    "failed_rows": 0,
                    "coverage_pct": 0.0,
                },
                "enriched_rows": [],
                "survey_level_sentiment": {
                    "total_surveys": 0,
                    "positive_count": 0,
                    "neutral_count": 0,
                    "negative_count": 0,
                    "positive_pct": 0.0,
                    "neutral_pct": 0.0,
                    "negative_pct": 0.0,
                },
                "response_level_sentiment": {
                    "total_responses": 0,
                    "positive_count": 0,
                    "neutral_count": 0,
                    "negative_count": 0,
                    "positive_pct": 0.0,
                    "neutral_pct": 0.0,
                    "negative_pct": 0.0,
                },
                "main_cluster_breakdown": [],
            },
            ensure_ascii=False,
        )

    texts_for_sentiment: list[str] = []
    sentiment_target_indexes: list[int] = []

    for i, row in enumerate(eligible_rows):
        normalized_value = _normalize_for_match(row["response_value"])
        mapped = _RATING_SENTIMENT_MAP.get(normalized_value)
        if mapped:
            row["response_sentiment"] = mapped
            row["response_type"] = "structured_rating"
        else:
            row["response_type"] = "free_text"
            texts_for_sentiment.append(row["response_value"])
            sentiment_target_indexes.append(i)

    failed_rows = 0
    if texts_for_sentiment:
        cursor = 0
        for chunk in _chunked(texts_for_sentiment, 10):
            results = client.analyze_sentiment(chunk, show_opinion_mining=False)
            for doc in results:
                target_idx = sentiment_target_indexes[cursor]
                if doc.is_error:
                    eligible_rows[target_idx]["response_sentiment"] = "Neutral"
                    eligible_rows[target_idx]["analysis_error"] = doc.error.message
                    failed_rows += 1
                else:
                    eligible_rows[target_idx]["response_sentiment"] = _normalize_sentiment_label(doc.sentiment)
                cursor += 1

    for row in eligible_rows:
        row["main_cluster"] = _map_main_cluster(
            question_name=row.get("question_name", ""),
            fields_name=row.get("fields_name", ""),
            response_value=row.get("response_value", ""),
        )

    if include_key_phrases:
        all_texts = [r["response_value"] for r in eligible_rows]
        phrase_results: list[list[str]] = [[] for _ in all_texts]
        cursor = 0
        for chunk in _chunked(all_texts, 10):
            results = client.extract_key_phrases(chunk)
            for doc in results:
                if not doc.is_error:
                    phrase_results[cursor] = list(doc.key_phrases)
                cursor += 1
        for idx, row in enumerate(eligible_rows):
            row["key_phrases"] = phrase_results[idx]

    if include_entities:
        all_texts = [r["response_value"] for r in eligible_rows]
        entity_results: list[list[dict[str, Any]]] = [[] for _ in all_texts]
        cursor = 0
        for chunk in _chunked(all_texts, 10):
            results = client.recognize_entities(chunk)
            for doc in results:
                if not doc.is_error:
                    entity_results[cursor] = [
                        {
                            "text": e.text,
                            "category": e.category,
                            "subcategory": e.subcategory,
                            "confidence": round(e.confidence_score, 3),
                        }
                        for e in doc.entities
                    ]
                cursor += 1
        for idx, row in enumerate(eligible_rows):
            row["entities"] = entity_results[idx]

    # Survey-level sentiment
    survey_buckets: dict[str, list[str]] = defaultdict(list)
    for row in eligible_rows:
        sid = row.get("survey_response_id") or ""
        survey_key = sid if sid else f"__row_{row['row_index']}"
        survey_buckets[survey_key].append(row["response_sentiment"])

    survey_sentiment_by_id: dict[str, str] = {}
    for survey_id, sentiments in survey_buckets.items():
        survey_sentiment_by_id[survey_id] = _determine_survey_sentiment(sentiments)

    for row in eligible_rows:
        sid = row.get("survey_response_id") or ""
        survey_key = sid if sid else f"__row_{row['row_index']}"
        row["survey_level_sentiment"] = survey_sentiment_by_id[survey_key]

    # Response-level aggregate
    response_counter = Counter(row["response_sentiment"] for row in eligible_rows)
    total_responses = len(eligible_rows)
    response_level_sentiment = {
        "total_responses": total_responses,
        "positive_count": int(response_counter.get("Positive", 0)),
        "neutral_count": int(response_counter.get("Neutral", 0)),
        "negative_count": int(response_counter.get("Negative", 0)),
        "positive_pct": _pct(int(response_counter.get("Positive", 0)), total_responses),
        "neutral_pct": _pct(int(response_counter.get("Neutral", 0)), total_responses),
        "negative_pct": _pct(int(response_counter.get("Negative", 0)), total_responses),
    }

    # Survey-level aggregate
    survey_counter = Counter(survey_sentiment_by_id.values())
    total_surveys = len(survey_sentiment_by_id)
    survey_level_sentiment = {
        "total_surveys": total_surveys,
        "positive_count": int(survey_counter.get("Positive", 0)),
        "neutral_count": int(survey_counter.get("Neutral", 0)),
        "negative_count": int(survey_counter.get("Negative", 0)),
        "positive_pct": _pct(int(survey_counter.get("Positive", 0)), total_surveys),
        "neutral_pct": _pct(int(survey_counter.get("Neutral", 0)), total_surveys),
        "negative_pct": _pct(int(survey_counter.get("Negative", 0)), total_surveys),
    }

    # Main cluster breakdown
    by_cluster: dict[str, Counter[str]] = {cluster: Counter() for cluster in _MAIN_CLUSTERS}
    for row in eligible_rows:
        cluster = row.get("main_cluster", "Overall Service Delivery")
        sentiment = row.get("response_sentiment", "Neutral")
        if cluster not in by_cluster:
            by_cluster[cluster] = Counter()
        by_cluster[cluster][sentiment] += 1

    main_cluster_breakdown: list[dict[str, Any]] = []
    for cluster, counter in by_cluster.items():
        total_mentions = int(sum(counter.values()))
        if total_mentions == 0:
            continue
        pos = int(counter.get("Positive", 0))
        neu = int(counter.get("Neutral", 0))
        neg = int(counter.get("Negative", 0))
        main_cluster_breakdown.append(
            {
                "main_cluster": cluster,
                "positive_count": pos,
                "neutral_count": neu,
                "negative_count": neg,
                "total_mentions": total_mentions,
                "positive_pct": _pct(pos, total_mentions),
                "neutral_pct": _pct(neu, total_mentions),
                "negative_pct": _pct(neg, total_mentions),
            }
        )

    main_cluster_breakdown.sort(key=lambda x: (-x["total_mentions"], x["main_cluster"]))

    coverage = {
        "total_rows": total_rows,
        "eligible_rows": len(eligible_rows),
        "processed_rows": len(eligible_rows),
        "failed_rows": failed_rows,
        "coverage_pct": _pct(len(eligible_rows), len(eligible_rows)) if eligible_rows else 0.0,
    }

    payload = {
        "coverage": coverage,
        "survey_level_sentiment": survey_level_sentiment,
        "response_level_sentiment": response_level_sentiment,
        "main_cluster_breakdown": main_cluster_breakdown,
        "enriched_rows": eligible_rows,
    }
    return json.dumps(payload, ensure_ascii=False)


# ─── Legacy tools (kept for compatibility) ───────────────────────────────────

def analyze_sentiment(documents: Any = None) -> str:
    """Legacy sentiment tool: analyze free-text documents."""
    client = _get_client()
    docs = _docs(documents) if documents else (_take_pending_documents() or [])
    if not docs:
        return json.dumps({"error": "No documents provided."}, ensure_ascii=False)

    output = []
    for chunk in _chunked(docs, 10):
        results = client.analyze_sentiment(chunk, show_opinion_mining=True)
        for i, doc in enumerate(results):
            text = chunk[i]
            if doc.is_error:
                output.append({"text": text, "error": doc.error.message})
                continue
            output.append(
                {
                    "text": text,
                    "sentiment": _normalize_sentiment_label(doc.sentiment),
                    "confidence": {
                        "positive": round(doc.confidence_scores.positive, 3),
                        "neutral": round(doc.confidence_scores.neutral, 3),
                        "negative": round(doc.confidence_scores.negative, 3),
                    },
                }
            )
    return json.dumps(output, ensure_ascii=False)


def extract_key_phrases(documents: Any) -> str:
    """Extract the key phrases from each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.extract_key_phrases(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append({"index": i, "text": docs[i], "key_phrases": list(doc.key_phrases)})
    return json.dumps(output, ensure_ascii=False)


def recognize_entities(documents: Any) -> str:
    """Recognise named entities (people, places, organisations, dates, …)."""
    client = _get_client()
    docs = _docs(documents)
    results = client.recognize_entities(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append(
                {
                    "index": i,
                    "text": docs[i],
                    "entities": [
                        {
                            "text": e.text,
                            "category": e.category,
                            "subcategory": e.subcategory,
                            "confidence": round(e.confidence_score, 3),
                        }
                        for e in doc.entities
                    ],
                }
            )
    return json.dumps(output, ensure_ascii=False)


def detect_language(documents: Any) -> str:
    """Detect the language of each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.detect_language(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append(
                {
                    "index": i,
                    "text": docs[i],
                    "language": doc.primary_language.name,
                    "iso6391_name": doc.primary_language.iso6391_name,
                    "confidence": round(doc.primary_language.confidence_score, 3),
                }
            )
    return json.dumps(output, ensure_ascii=False)


def recognize_pii_entities(documents: Any) -> str:
    """Detect PII (names, emails, phone numbers, etc.) in each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.recognize_pii_entities(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append(
                {
                    "index": i,
                    "text": docs[i],
                    "redacted_text": doc.redacted_text,
                    "entities": [
                        {
                            "text": e.text,
                            "category": e.category,
                            "confidence": round(e.confidence_score, 3),
                        }
                        for e in doc.entities
                    ],
                }
            )
    return json.dumps(output, ensure_ascii=False)


# ─── Dispatch table ───────────────────────────────────────────────────────────

TOOL_DISPATCH: dict[str, Any] = {
    "analyze_structured_survey": analyze_structured_survey,
    "analyze_sentiment": analyze_sentiment,
    "extract_key_phrases": extract_key_phrases,
    "recognize_entities": recognize_entities,
    "detect_language": detect_language,
    "recognize_pii_entities": recognize_pii_entities,
}


# ─── Function tool definitions (OpenAI function-calling schema) ─────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_structured_survey",
            "description": (
                "Analyze mapped structured survey rows and return row-level enrichment "
                "with coverage, sentiment aggregates, and main-cluster breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "survey_response_id": {"type": "string"},
                                "question_name": {"type": "string"},
                                "response_value": {"type": "string"},
                                "fields_name": {"type": "string"},
                            },
                            "required": ["survey_response_id", "question_name", "response_value"],
                        },
                        "description": "Mapped structured survey rows. Optional when pending rows are already uploaded.",
                    },
                    "include_key_phrases": {
                        "type": "boolean",
                        "description": "Include key phrase extraction for each row.",
                        "default": False,
                    },
                    "include_entities": {
                        "type": "boolean",
                        "description": "Include entity extraction for each row.",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": (
                "Analyse the sentiment (positive / neutral / negative) of one or more text "
                "documents and mine fine-grained opinions about specific aspects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents to analyse. Any size is accepted; internal batching is handled automatically.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_key_phrases",
            "description": "Extract the most important key phrases from one or more documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_entities",
            "description": (
                "Recognise named entities such as people, organisations, locations, dates, "
                "events, and products in one or more documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_language",
            "description": "Detect the language of one or more documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_pii_entities",
            "description": (
                "Detect Personally Identifiable Information (PII) such as names, email "
                "addresses, phone numbers and ID numbers. Returns detected entities and "
                "a redacted version of the text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
]
