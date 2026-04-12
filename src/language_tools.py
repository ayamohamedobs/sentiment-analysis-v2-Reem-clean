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
    print(f"📤 set_pending_structured_rows: Stored {len(_pending_structured_rows)} rows in global")


def _take_pending_structured_rows() -> list[dict[str, Any]] | None:
    """Return and clear pending uploaded structured survey rows.

    First checks Streamlit session state, then falls back to global pending rows.
    """
    try:
        import streamlit as st

        rows = st.session_state.get("_pending_structured_rows")
        if rows:
            print(f"📥 _take_pending_structured_rows: Retrieved {len(rows)} from Streamlit session state")
            st.session_state.pop("_pending_structured_rows", None)
            return [dict(r) for r in rows]
        else:
            print("📥 _take_pending_structured_rows: No rows in Streamlit session state")
    except (ImportError, RuntimeError, AttributeError) as e:
        print(f"📥 _take_pending_structured_rows: Streamlit import/error: {type(e).__name__}: {e}")

    global _pending_structured_rows
    if _pending_structured_rows:
        print(f"📥 _take_pending_structured_rows: Retrieved {len(_pending_structured_rows)} from global fallback")
    else:
        print(f"📥 _take_pending_structured_rows: No rows in global fallback, rows={_pending_structured_rows}")
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
    "Unmapped",
]

_SHORT_NEGATIVE_PHRASES = {
    "couldn't find documentation",
    "couldnt find documentation",
    "hard to navigate",
    "confusing site navigation",
    "login issue",
    "authentication problem",
    "unable to download",
    "missing tutorials",
    "lack of communication on status",
    "engineer handoff delays",
    "slow case review",
    "unclear case updates",
    "difficult navigation",
    # Documentation-specific negatives
    "outdated documentation",
    "documentation is outdated",
    "documentation was outdated",
    "no documentation",
    "poor documentation",
    "incomplete documentation",
    "hard to find documentation",
    "documentation needs improvement",
    "documentation could be better",
    "lacking documentation",
    "insufficient documentation",
    "documentation was not helpful",
    "documentation not helpful",
    "not enough documentation",
    "documentation is confusing",
    "documentation was confusing",
    "unclear documentation",
    "difficult to find information",
    "hard to find information",
    "not well documented",
    "poorly documented",
    "needs better documentation",
    "knowledge base is lacking",
    "knowledge base needs improvement",
    "articles are outdated",
    "articles were outdated",
    "missing documentation",
    "no tutorials",
    "lack of documentation",
    "stale documentation",
}

_SHORT_POSITIVE_PHRASES = {
    "great support",
    "great service",
    "great experience",
    "great job",
    "great work",
    "excellent support",
    "excellent service",
    "excellent experience",
    "excellent engineer",
    "excellent job",
    "very helpful",
    "very satisfied",
    "very responsive",
    "very knowledgeable",
    "very professional",
    "really helpful",
    "really good",
    "really great",
    "highly satisfied",
    "well done",
    "good experience",
    "good support",
    "good service",
    "good job",
    "thank you",
    "thanks",
    "thankful",
    "much appreciated",
    "happy with",
    "pleased with",
    "impressed with",
    "satisfied with",
    "easy to use",
    "no issues",
    "no problems",
    "no complaints",
    "went well",
    "above and beyond",
    "quick resolution",
    "fast resolution",
    "timely resolution",
    "resolved quickly",
    "resolved fast",
    "outstanding",
    "exceptional",
    "phenomenal",
    "superb",
    "fantastic",
    "amazing",
    "awesome",
    "wonderful",
    "perfect",
    "love it",
    "kudos",
    "top notch",
    # Documentation-specific positives
    "helpful documentation",
    "good documentation",
    "great documentation",
    "excellent documentation",
    "documentation was helpful",
    "documentation is helpful",
    "documentation was great",
    "documentation was excellent",
    "documentation was clear",
    "clear documentation",
    "well documented",
    "easy to find",
    "easy to follow",
    "useful articles",
    "helpful articles",
    "articles were helpful",
    "knowledge base is helpful",
    "knowledge base was helpful",
    "good articles",
    "great articles",
    "informative documentation",
    "thorough documentation",
    "comprehensive documentation",
    "documentation is good",
    "documentation was good",
}


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _detect_response_type(question_name: str, response_value: str) -> str:
    q = _normalize_for_match(question_name)
    rv = _normalize_for_match(response_value)

    if rv in _RATING_SENTIMENT_MAP:
        return "structured_rating"

    structured_option_questions = [
        "which of the following elements",
        "what one improvement area would you suggest for rsa’s online support tools",
        "what one improvement area",
        "primary source of",
    ]

    if any(key in q for key in structured_option_questions):
        return "structured_option_signal"

    # Operational: "Did the TSE(s) include other RSA teams?" is always operational.
    # This is a generic select-option question — exclude from sentiment entirely.
    if "did the technical support engineer" in q and (
        "include other rsa teams" in q or "include other" in q
    ):
        return "structured_operational"

    return "free_text"


def _route_freetext_by_content(source: str, default: str) -> str:
    """Route a free-text follow-up response to the cluster that best matches
    what the customer is actually talking about, instead of blindly using the
    parent question's cluster.  Falls back to *default* when no strong signal."""

    # Documentation & Knowledge base (check first — most specific)
    if any(
        k in source
        for k in [
            "documentation",
            "knowledge base",
            "guide",
            "article",
            "self-service",
            "troubleshooting",
            "blogs",
            "tutorials",
            "videos",
        ]
    ):
        return "Documentation & Knowledge base"

    # Communication signals
    if any(
        k in source
        for k in [
            "communication",
            "status update",
            "follow-up",
            "follow up",
            "kept me informed",
            "timely",
            "response time",
            "waiting",
            "waited",
            "no update",
        ]
    ):
        return "Communication"

    # Technical Support Quality signals
    if any(
        k in source
        for k in [
            "technical support engineer",
            "support engineer",
            "tse",
            "engineer",
            "expertise",
            "knowledgeable",
            "professional",
            "handoff",
            "hand-off",
            "escalat",
            "ownership",
            "case owner",
        ]
    ):
        return "Technical Support Quality"

    # Tools signals
    if any(
        k in source
        for k in [
            "portal",
            "site navigation",
            "online tool",
            "online support tool",
            "download",
            "authentication",
            "login",
            "password",
            "search",
            "website",
            "training registration",
            "registration system",
            "creating and managing",
            "online service case",
            "managing this online",
        ]
    ):
        return "Tools"

    # Product Satisfaction signals
    if any(
        k in source
        for k in [
            "product",
            "upgrade",
            "version",
            "bug",
            "feature",
            "patch",
            "release",
            "software",
            "firmware",
            "installation",
        ]
    ):
        return "Product Satisfaction"

    return default


def _map_main_cluster(question_name: str, fields_name: str = "", response_value: str = "") -> str:
    q = _normalize_for_match(question_name)
    f = _normalize_for_match(fields_name)
    r = _normalize_for_match(response_value)
    source = " ".join([q, f, r]).strip()

    # Product Satisfaction
    if "overall, how satisfied are you with the rsa product this service case was opened for" in q:
        return "Product Satisfaction"
    if "why you provided the above answer regarding the rsa product" in q:
        routed = _route_freetext_by_content(source, default="Product Satisfaction")
        return routed

    # Tools
    if "how satisfied are you overall with rsa's online support tools" in q:
        return "Tools"
    if "additional comments for rsa online support tools" in q:
        routed = _route_freetext_by_content(source, default="Tools")
        return routed
    if "please provide additional detail to help us better understand the issue" in q:
        routed = _route_freetext_by_content(source, default="Tools")
        return routed
    if "what one improvement area" in q:
        routed = _route_freetext_by_content(source, default="Tools")
        return routed

    # Overall Service Delivery
    if "thinking only of the service case referenced in the email, how satisfied are you with the service you received" in q:
        return "Overall Service Delivery"
    if "why you provided the above answer regarding the service you received" in q:
        routed = _route_freetext_by_content(source, default="Overall Service Delivery")
        return routed
    if "to what extent do you agree with the following statement: rsa made it easy for me to resolve my technical issue" in q:
        return "Overall Service Delivery"

    # Communication
    # Checked before Technical Support to avoid collisions on ambiguous wording.
    if "how satisfied" in q and "time it took to resolve" in q:
        return "Communication"
    if "how satisfied" in q and "timely communication" in q:
        return "Communication"
    if "how satisfied" in q and "receiving timely communication" in q:
        return "Communication"

    # Technical Support Quality
    if "how would you rate your overall satisfaction with the technical support engineer" in q:
        return "Technical Support Quality"
    if "overall satisfaction with the technical support engineer" in q:
        return "Technical Support Quality"
    if "primary source of" in q and "dissatisfaction" in q:
        routed = _route_freetext_by_content(source, default="Technical Support Quality")
        return routed
    if "which of the following elements" in q and ("satisfaction" in q or "dissatisfaction" in q):
        routed = _route_freetext_by_content(source, default="Technical Support Quality")
        return routed
    if "did the technical support engineer" in q and ("include other rsa teams" in q or "include other" in q):
        return "Technical Support Quality"
    if "other, please specify" in q:
        routed = _route_freetext_by_content(source, default="Technical Support Quality")
        return routed

    # Documentation & Knowledge base fallback
    if any(
        k in source
        for k in [
            "documentation",
            "knowledge base",
            "guide",
            "article",
            "self-service",
            "troubleshooting",
            "product documentation",
            "blogs",
            "tutorials",
            "supportive videos",
            "completeness",
            "usability",
            "clarity",
            "findability",
        ]
    ):
        return "Documentation & Knowledge base"

    # Context-based fallback: Map generic free-text by content if no question matched
    # Communication keywords (broader catch)
    if any(
        k in source
        for k in [
            "communication",
            "status",
            "updates",
            "informed",
            "follow-up",
            "follow up",
            "feedback",
            "responsiveness",
            "responsive",
            "timely",
            "lag",
            "delay",
            "keep inform",
            "coordin",
            "update",
        ]
    ):
        return "Communication"

    # Technical Support Quality keywords (broader catch)
    if any(
        k in source
        for k in [
            "handling",
            "handled",
            "competent",
            "expertise",
            "knowledge",
            "professional",
            "quality",
            "engineer",
            "tse",
            "support",
            "technical",
            "resolution",
            "resolve",
            "issue",
            "problem",
            "quick",
            "fast",
            "slow",
            "satisf",
        ]
    ):
        return "Technical Support Quality"

    # Overall Service Delivery (generic service language)
    if any(
        k in source
        for k in [
            "service",
            "case",
            "help",
            "assist",
            "process",
            "experience",
            "satisfied",
            "dissatisfied",
            "easy",
            "difficult",
            "resolve",
        ]
    ):
        return "Overall Service Delivery"

    return "Unmapped"


def _normalize_sentiment_label(value: str, confidence_scores=None) -> str:
    low = value.strip().lower()
    if confidence_scores is not None:
        pos = getattr(confidence_scores, "positive", 0)
        neg = getattr(confidence_scores, "negative", 0)
        neu = getattr(confidence_scores, "neutral", 0)
        # For "neutral" or "mixed": if neutral confidence is weak, pick the
        # dominant non-neutral sentiment instead.
        if low in ("neutral", "mixed"):
            if neu < 0.55 and (pos > neu or neg > neu):
                return "Positive" if pos >= neg else "Negative"
            if low == "mixed":
                best = max(pos, neg, neu)
                if best == pos:
                    return "Positive"
                if best == neg:
                    return "Negative"
    if low == "positive":
        return "Positive"
    if low == "negative":
        return "Negative"
    return "Neutral"


def _determine_survey_sentiment(row_sentiments: list[str]) -> str:
    if not row_sentiments:
        return "Neutral"

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


def _detect_evidence_clusters(response_value: str) -> set[str]:
    """Identify all clusters that a free-text response's content relates to.

    Uses the same keyword families as _route_freetext_by_content but returns
    ALL matching clusters instead of picking just one winner.
    """
    rv = _normalize_for_match(response_value)
    clusters: set[str] = set()

    doc_kw = ["documentation", "knowledge base", "guide", "article", "self-service",
              "troubleshooting", "blogs", "tutorials", "videos"]
    comm_kw = ["communication", "status update", "follow-up", "follow up",
               "kept me informed", "timely", "response time", "waiting", "waited", "no update"]
    tsq_kw = ["technical support engineer", "support engineer", "tse", "engineer",
              "expertise", "knowledgeable", "professional", "handoff", "hand-off",
              "escalat", "ownership", "case owner"]
    tools_kw = ["portal", "site navigation", "online tool", "online support tool",
                "download", "authentication", "login", "password", "search", "website",
                "training registration"]
    prod_kw = ["product", "upgrade", "version", "bug", "feature", "patch", "release",
               "software", "firmware", "installation", "user experience",
               "token enrollment", "licensing", "license", "on-prem", "cloud"]

    if any(k in rv for k in doc_kw):
        clusters.add("Documentation & Knowledge base")
    if any(k in rv for k in comm_kw):
        clusters.add("Communication")
    if any(k in rv for k in tsq_kw):
        clusters.add("Technical Support Quality")
    if any(k in rv for k in tools_kw):
        clusters.add("Tools")
    if any(k in rv for k in prod_kw):
        clusters.add("Product Satisfaction")

    return clusters


def _build_evidence_summary(eligible_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build rich evidence summary for agent context without full row payload.

    Extracts top recurring items, free-text phrases, named items (tools, portals,
    features, product areas), and sample verbatims by cluster and sentiment.
    Excludes ratings and operational/factual responses (not evidence-bearing).
    """
    # Named-item keywords to scan for in free-text responses
    _NAMED_ITEMS = {
        "community portal": "Community Portal",
        "partner portal": "Partner Portal",
        "license portal": "License Portal",
        "support portal": "Support Portal",
        "help center": "Help Center",
        "knowledge base": "Knowledge Base",
        "site navigation": "Site Navigation",
        "search": "Search Functionality",
        "authentication": "Authentication / Login",
        "login": "Authentication / Login",
        "password": "Authentication / Login",
        "product documentation": "Product Documentation",
        "training registration": "Training Registration",
        "download": "Downloads / Patches",
        "patch": "Downloads / Patches",
        "service pack": "Downloads / Patches",
        "license": "Licensing",
        "upgrade": "Upgrade Experience",
        "installation": "Installation",
        "implementation": "Implementation",
        "case owner": "Case Ownership / Handoff",
        "handoff": "Case Ownership / Handoff",
        "hand-off": "Case Ownership / Handoff",
        "turnover": "Case Ownership / Handoff",
        "escalat": "Escalation Process",
        "time zone": "Time Zone Differences",
        "timezone": "Time Zone Differences",
        "engineering fix": "Engineering Fix Dependency",
        "intermittent": "Intermittent / Hard-to-Reproduce Issue",
        "difficult to reproduce": "Intermittent / Hard-to-Reproduce Issue",
        "blog": "Blogs / Tutorials / Videos",
        "tutorial": "Blogs / Tutorials / Videos",
        "video": "Blogs / Tutorials / Videos",
    }

    # Filter to evidence-bearing rows only (free-text and structured options)
    evidence_rows = [
        r for r in eligible_rows
        if r.get("response_type") in ("free_text", "structured_option_signal")
    ]

    if not evidence_rows:
        return {"negative_by_cluster": {}, "positive_by_cluster": {}}

    # Group by cluster and sentiment
    by_cluster_sentiment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        cluster = row.get("main_cluster", "Unmapped")
        sentiment = row.get("response_sentiment", "Neutral")
        by_cluster_sentiment[(cluster, sentiment)].append(row)
        # Dual-tagged: Positive/Neutral overall but has negative sub-points
        # Route into negative evidence for ALL clusters the negative content relates to
        if row.get("has_negative_mentions") and sentiment != "Negative":
            neg_clusters = _detect_evidence_clusters(row.get("response_value", ""))
            if not neg_clusters:
                neg_clusters = {cluster}  # fallback: primary cluster
            for neg_cluster in neg_clusters:
                by_cluster_sentiment[(neg_cluster, "Negative")].append(row)

    result = {"negative_by_cluster": {}, "positive_by_cluster": {}}

    # Process each cluster/sentiment combination
    for (cluster, sentiment), rows_in_group in by_cluster_sentiment.items():
        if sentiment not in ("Positive", "Negative"):
            continue

        # Separate structured vs free-text rows
        structured_rows = [r for r in rows_in_group if r.get("response_type") == "structured_option_signal"]
        free_text_rows = [r for r in rows_in_group if r.get("response_type") == "free_text"]

        # Count structured option values
        structured_values_count = Counter()
        for row in structured_rows:
            value = row.get("response_value", "").strip()
            if value:
                structured_values_count[value] += 1

        # Extract and count free-text phrases
        free_text_phrases_count = Counter()
        for row in free_text_rows:
            key_phrases = row.get("key_phrases", [])
            if key_phrases and isinstance(key_phrases, list):
                for phrase in key_phrases[:3]:
                    phrase_str = str(phrase).strip().lower()
                    if phrase_str and len(phrase_str) > 3:
                        free_text_phrases_count[phrase_str] += 1
            else:
                response = row.get("response_value", "").strip()
                if response:
                    first_phrase = response.split(".")[0][:80].strip().lower()
                    if first_phrase and len(first_phrase) > 3:
                        free_text_phrases_count[first_phrase] += 1

        # Extract named items (tools, portals, features, product areas)
        named_items_count = Counter()
        for row in rows_in_group:
            text = row.get("response_value", "").strip().lower()
            seen_labels = set()
            for keyword, label in _NAMED_ITEMS.items():
                if keyword in text and label not in seen_labels:
                    named_items_count[label] += 1
                    seen_labels.add(label)

        # Collect sample verbatims — more, longer
        sample_verbatims = []
        for row in rows_in_group:
            response = row.get("response_value", "").strip()
            if response and len(sample_verbatims) < 30:
                if len(response) <= 400:
                    sample_verbatims.append(response)
                else:
                    sample_verbatims.append(response[:400] + "...")

        top_structured = [
            {"label": label, "mentions": count}
            for label, count in structured_values_count.most_common(10)
        ]

        top_free_text = [
            {"label": label, "mentions": count}
            for label, count in free_text_phrases_count.most_common(10)
        ]

        top_named_items = [
            {"item": item, "mentions": count}
            for item, count in named_items_count.most_common(10)
        ]

        target_dict = result["negative_by_cluster"] if sentiment == "Negative" else result["positive_by_cluster"]
        target_dict[cluster] = {
            "top_structured_items": top_structured,
            "top_free_text_phrases": top_free_text,
            "named_items": top_named_items,
            "sample_verbatims": sample_verbatims,
            "total_evidence_rows": len(rows_in_group),
        }

    return result


def _rule_based_sentiment_override(response_value: str) -> str | None:
    rv = _normalize_for_match(response_value)

    # 1. Exact / substring match against known short phrases
    if rv in _SHORT_NEGATIVE_PHRASES:
        return "Negative"
    if any(phrase in rv for phrase in _SHORT_NEGATIVE_PHRASES):
        return "Negative"
    if rv in _SHORT_POSITIVE_PHRASES:
        return "Positive"
    if any(phrase in rv for phrase in _SHORT_POSITIVE_PHRASES):
        return "Positive"

    # 2. Signal-keyword scan for longer free-text (catches descriptive complaints/praise)
    _NEG_SIGNALS = [
        "impossible", "never be resolved", "never resolved", "can't be resolved",
        "sadly missing", "missing in", "limited in function", "very limited",
        "huge amount of work", "too much work", "too many steps",
        "turned off", "frustrated", "frustrating", "frustration",
        "i wish", "i don't believe", "i don't use", "don't use it",
        "disjointed", "disconnected", "broken", "unusable",
        "not intuitive", "not user friendly", "not user-friendly",
        "waste of time", "waste time", "time consuming", "time-consuming",
        "terrible", "horrible", "awful", "unacceptable", "ridiculous",
        "disappointed", "disappointing", "disappointment",
        "struggle", "struggling", "struggled",
        "cumbersome", "clunky", "painful", "tedious", "annoying",
        "doesn't work", "does not work", "didn't work", "did not work",
        "not working", "stopped working", "never works",
        "too complicated", "overly complicated", "overly complex",
        "worse", "worst", "downgrade", "regression",
        "lack of", "lacking", "insufficient", "inadequate",
        "no way to", "unable to", "cannot", "can't",
        "should be easier", "should be simpler", "needs to be easier",
        "needs improvement", "needs work", "room for improvement",
        "far behind", "falling behind", "lagging behind",
        "not satisfied", "dissatisfied", "unsatisfied", "unhappy",
        "bug", "bugs", "buggy", "glitch", "glitches",
    ]
    neg_hits = sum(1 for sig in _NEG_SIGNALS if sig in rv)
    if neg_hits >= 2:
        return "Negative"
    # Single strong indicators (unambiguous even alone)
    _STRONG_NEG = [
        "impossible", "never be resolved", "sadly missing", "unusable",
        "unacceptable", "terrible", "horrible", "awful", "waste of time",
        "ridiculous", "worst", "regression", "dissatisfied",
        "turned off", "i wish", "i don't believe", "i don't use",
        "huge amount of work", "very limited",
    ]
    if any(sig in rv for sig in _STRONG_NEG):
        return "Negative"

    _POS_SIGNALS = [
        "love", "loving", "loved",
        "impressed", "impressive",
        "smooth", "seamless", "flawless",
        "exceeded expectations", "above and beyond",
        "highly recommend", "would recommend",
        "best support", "best experience", "best service",
        "extremely helpful", "incredibly helpful",
        "very happy", "very pleased", "very impressed",
        "keep up the good work", "keep up the great work",
    ]
    if any(sig in rv for sig in _POS_SIGNALS):
        return "Positive"

    return None


def _has_negative_mentions(response_value: str) -> bool:
    """Check whether a response contains negative sub-points even if overall positive."""
    rv = _normalize_for_match(response_value)
    _NEG_MENTION_SIGNALS = [
        "however", "but ", "room for improvement", "needs improvement",
        "could be better", "could improve", "would enhance",
        "should be easier", "should be simpler", "should improve",
        "area of improvement", "areas of improvement", "improvement area",
        "wish ", "i wish", "lacking", "lack of",
        "slow", "slower", "delay", "delays",
        "frustrat", "cumbersome", "tedious", "manual steps",
        "hard to", "difficult to", "confusing",
        "not easy", "not intuitive", "not user friendly",
        "faster resolution", "quicker resolution",
        "simplif", "streamlin",
        "reduce", "reducing dependency",
    ]
    return sum(1 for sig in _NEG_MENTION_SIGNALS if sig in rv) >= 2


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
    print("\n🔍 ANALYZE: analyze_structured_survey called")
    print(f"   rows param: {rows is not None}")
    print(f"   source_rows retrieved: {total_rows} total")
    if source_rows:
        print(f"   First row: {source_rows[0]}")
    if not source_rows:
        return json.dumps({"error": "No structured survey rows provided."}, ensure_ascii=False)

    eligible_rows: list[dict[str, Any]] = []
    skipped_blank = 0
    for idx, row in enumerate(source_rows):
        response_value = _normalize_text(row.get("response_value", ""))
        if _is_blank(response_value):
            skipped_blank += 1
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
    print(f"   Eligible rows after blank filter: {len(eligible_rows)} (skipped {skipped_blank})")

    if not eligible_rows:
        return json.dumps(
            {
                "coverage": {
                    "total_rows": total_rows,
                    "eligible_rows": 0,
                    "rating_rows": 0,
                    "free_text_rows": 0,
                    "structured_option_rows": 0,
                    "non_sentiment_rows": 0,
                    "processed_rows": 0,
                    "failed_rows": 0,
                    "coverage_pct": 0.0,
                },
                "enriched_rows": [],
                "enriched_rows_sample": [],
                "enriched_rows_count": 0,
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
                "evidence_summary": {
                    "negative_by_cluster": {},
                    "positive_by_cluster": {},
                },
            },
            ensure_ascii=False,
        )

    texts_for_sentiment: list[str] = []
    sentiment_target_indexes: list[int] = []

    for i, row in enumerate(eligible_rows):
        response_type = _detect_response_type(
            question_name=row.get("question_name", ""),
            response_value=row["response_value"],
        )
        row["response_type"] = response_type

        normalized_value = _normalize_for_match(row["response_value"])

        if response_type == "structured_rating":
            mapped = _RATING_SENTIMENT_MAP.get(normalized_value, None)
            if mapped is None:
                mapped = "Neutral"
                row["_neutral_source"] = "unrecognized_rating"
            row["response_sentiment"] = mapped
            row["sentiment_eligible"] = True

        elif response_type == "structured_option_signal":
            row["response_sentiment"] = "Negative"
            row["structured_option_value"] = row["response_value"]
            row["sentiment_eligible"] = True

        elif response_type == "structured_operational":
            row["response_sentiment"] = None
            row["sentiment_eligible"] = False

        else:
            rule_based = _rule_based_sentiment_override(row["response_value"])
            if rule_based is not None:
                row["response_sentiment"] = rule_based
                row["sentiment_eligible"] = True
            else:
                row["sentiment_eligible"] = True
                texts_for_sentiment.append(row["response_value"])
                sentiment_target_indexes.append(i)

    failed_rows = 0
    neutral_sources: dict[str, int] = defaultdict(int)  # track where Neutrals come from
    if texts_for_sentiment:
        print(f"\n🤖 SENTIMENT: Sending {len(texts_for_sentiment)} texts to Azure API...")
        cursor = 0
        chunk_count = 0
        for chunk in _chunked(texts_for_sentiment, 10):
            chunk_count += 1
            print(f"   Chunk {chunk_count}: {len(chunk)} texts")
            results = client.analyze_sentiment(chunk, show_opinion_mining=False)
            results_list = list(results)
            print(f"   Azure returned: {len(results_list)} results")
            for doc in results_list:
                target_idx = sentiment_target_indexes[cursor]
                if doc.is_error:
                    eligible_rows[target_idx]["response_sentiment"] = "Neutral"
                    eligible_rows[target_idx]["analysis_error"] = doc.error.message
                    neutral_sources["azure_error"] += 1
                    failed_rows += 1
                else:
                    raw_label = doc.sentiment.strip().lower()
                    resolved = _normalize_sentiment_label(doc.sentiment, confidence_scores=doc.confidence_scores)
                    eligible_rows[target_idx]["response_sentiment"] = resolved
                    if raw_label == "mixed":
                        neutral_sources[f"mixed_resolved_{resolved}"] += 1
                    elif raw_label == "neutral" and resolved != "Neutral":
                        neutral_sources[f"neutral_rescued_{resolved}"] += 1
                    elif resolved == "Neutral":
                        neutral_sources["azure_neutral"] += 1
                cursor += 1

    # Count Neutrals from other sources
    for row in eligible_rows:
        if row.get("_neutral_source") == "unrecognized_rating":
            neutral_sources["unrecognized_rating"] += 1
        elif row.get("response_type") == "structured_rating" and row.get("response_sentiment") == "Neutral":
            neutral_sources["known_rating_neutral"] += 1

    print(f"\n📊 NEUTRAL SOURCE BREAKDOWN: {dict(neutral_sources)}")

    # Dump sample texts that Azure still classified as Neutral
    sample_neutrals = [
        eligible_rows[sentiment_target_indexes[i]]["response_value"]
        for i in range(len(sentiment_target_indexes))
        if eligible_rows[sentiment_target_indexes[i]].get("response_sentiment") == "Neutral"
    ]
    if sample_neutrals:
        print(f"\n📝 SAMPLE AZURE NEUTRAL TEXTS ({len(sample_neutrals)} total, showing first 15):")
        for txt in sample_neutrals[:15]:
            print(f"   • {txt[:120]}")

    # Tag Positive/Neutral free-text that contains negative sub-points
    dual_tagged = 0
    for row in eligible_rows:
        if (
            row.get("response_type") == "free_text"
            and row.get("response_sentiment") in ("Positive", "Neutral")
            and _has_negative_mentions(row.get("response_value", ""))
        ):
            row["has_negative_mentions"] = True
            dual_tagged += 1
    if dual_tagged:
        print(f"\n🔀 DUAL-TAGGED: {dual_tagged} rows are overall Positive/Neutral but contain negative mentions")

    for row in eligible_rows:
        row["main_cluster"] = _map_main_cluster(
            question_name=row.get("question_name", ""),
            fields_name=row.get("fields_name", ""),
            response_value=row.get("response_value", ""),
        )

    # ──── DIAGNOSTIC: Classification Breakdown ────────────────────────────────
    diagnostic_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for row in eligible_rows:
        q_normalized = _normalize_for_match(row.get("question_name", ""))
        cluster = row.get("main_cluster", "Unmapped")
        resp_type = row.get("response_type", "unknown")
        diagnostic_counts[q_normalized][cluster][resp_type] += 1

    print("\n" + "=" * 100)
    print("DIAGNOSTIC: Classification Breakdown (question_name → main_cluster → response_type)")
    print("=" * 100)
    unmapped_count = 0
    for q_name in sorted(diagnostic_counts.keys()):
        print(f"\nQuestion: {q_name[:80]}")
        cluster_map = diagnostic_counts[q_name]
        for cluster in sorted(cluster_map.keys()):
            type_map = cluster_map[cluster]
            type_str = ", ".join(f"{rt}:{count}" for rt, count in sorted(type_map.items()))
            total = sum(type_map.values())
            print(f"  → {cluster}: {total} rows ({type_str})")
            if cluster == "Unmapped":
                unmapped_count += total
    print(f"\n⚠️ UNMAPPED TOTAL: {unmapped_count} rows")
    print("=" * 100 + "\n")
    # ──── END DIAGNOSTIC ──────────────────────────────────────────────────────

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
        if not row.get("sentiment_eligible", True):
            continue
        sid = row.get("survey_response_id") or ""
        survey_key = sid if sid else f"__row_{row['row_index']}"
        survey_buckets[survey_key].append(row["response_sentiment"])

    survey_sentiment_by_id: dict[str, str] = {}
    for survey_id, sentiments in survey_buckets.items():
        survey_sentiment_by_id[survey_id] = _determine_survey_sentiment(sentiments)

    for row in eligible_rows:
        sid = row.get("survey_response_id") or ""
        survey_key = sid if sid else f"__row_{row['row_index']}"
        row["survey_level_sentiment"] = survey_sentiment_by_id.get(survey_key, "Neutral")

    # Response-level aggregate
    sentiment_rows = [
        row
        for row in eligible_rows
        if row.get("sentiment_eligible", True) and row.get("response_sentiment") is not None
    ]
    response_counter = Counter(row["response_sentiment"] for row in sentiment_rows)
    total_responses = len(sentiment_rows)
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
        if not row.get("sentiment_eligible", True):
            continue
        cluster = row.get("main_cluster", "Unmapped")
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

    # Ensure key phrases are available for free-text evidence extraction
    free_text_rows_without_phrases = [
        (idx, r) for idx, r in enumerate(eligible_rows)
        if r.get("response_type") == "free_text" and not r.get("key_phrases")
    ]
    if free_text_rows_without_phrases:
        texts_to_extract = [r["response_value"] for _, r in free_text_rows_without_phrases]
        cursor = 0
        for chunk in _chunked(texts_to_extract, 10):
            results = client.extract_key_phrases(chunk)
            for doc in results:
                _, row = free_text_rows_without_phrases[cursor]
                if not doc.is_error:
                    row["key_phrases"] = list(doc.key_phrases)
                else:
                    row["key_phrases"] = []
                cursor += 1

    # Build compact evidence summary for agent context
    evidence_summary = _build_evidence_summary(eligible_rows)

    coverage = {
        "total_rows": total_rows,
        "eligible_rows": len(eligible_rows),
        "rating_rows": sum(1 for r in eligible_rows if r.get("response_type") == "structured_rating"),
        "free_text_rows": sum(1 for r in eligible_rows if r.get("response_type") == "free_text"),
        "structured_option_rows": sum(
            1 for r in eligible_rows if r.get("response_type") == "structured_option_signal"
        ),
        "non_sentiment_rows": sum(
            1 for r in eligible_rows if r.get("response_type") == "structured_operational"
        ),
        "processed_rows": len(eligible_rows),
        "failed_rows": failed_rows,
        "coverage_pct": _pct(len(eligible_rows) - failed_rows, total_rows) if total_rows else 0.0,
    }
    print("\n📊 COVERAGE:")
    print(f"   Total: {coverage['total_rows']}, Eligible: {coverage['eligible_rows']}")
    print(f"   Processed: {coverage['processed_rows']}, Failed: {coverage['failed_rows']}")
    print(
        "   By type - "
        f"Rating: {coverage['rating_rows']}, "
        f"FreeText: {coverage['free_text_rows']}, "
        f"Options: {coverage['structured_option_rows']}, "
        f"NonSentiment: {coverage['non_sentiment_rows']}"
    )

    payload = {
        "coverage": coverage,
        "survey_level_sentiment": survey_level_sentiment,
        "response_level_sentiment": response_level_sentiment,
        "main_cluster_breakdown": main_cluster_breakdown,
        "evidence_summary": evidence_summary,
        "enriched_rows_sample": eligible_rows[:100],
        "enriched_rows_count": len(eligible_rows),
    }
    print("\n📦 PAYLOAD:")
    print(f"   enriched_rows_sample: {len(payload['enriched_rows_sample'])} rows")
    print(f"   enriched_rows_count: {payload['enriched_rows_count']}")
    if payload["enriched_rows_sample"]:
        print(f"   Sample row 1: {payload['enriched_rows_sample'][0]}")
    print(f"   Main clusters in breakdown: {[c['main_cluster'] for c in payload['main_cluster_breakdown']]}")

    result_json = json.dumps(payload, ensure_ascii=False)
    print(f"   JSON size: {len(result_json)} bytes\n")
    return result_json


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
                    "sentiment": _normalize_sentiment_label(doc.sentiment, confidence_scores=doc.confidence_scores),
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