"""
Streamlit chat UI for the Sentiment Analysis Foundry Agent.

Run:
  streamlit run src/app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

# .env values OVERRIDE existing env vars to avoid stale terminal sessions
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8-sig") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"')

import streamlit as st
from azure.ai.agents.models import ToolOutput
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Ensure src/ is on path so language_tools is importable
sys.path.insert(0, os.path.dirname(__file__))

# ─── Application Insights Configuration ──────────────────────────────────────

_appinsights_connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if _appinsights_connection_string:
    try:
        from azure.ai.agents.telemetry import AIAgentsInstrumentor
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=_appinsights_connection_string)
        AIAgentsInstrumentor().instrument()
        print("✅ Application Insights monitoring enabled (with agents tracing)")
    except ImportError:
        print("⚠️  Application Insights packages not installed. Run: pip install -r requirements.txt")
else:
    print("ℹ️  Application Insights not configured (APPLICATIONINSIGHTS_CONNECTION_STRING not set)")


# ─── File reader ─────────────────────────────────────────────────────────────

_REQUIRED_FIELD_KEYS = ["survey_response_id", "question_name", "response_value"]
_OPTIONAL_FIELD_KEYS = ["fields_name"]
_NULL_LIKE_VALUES = {"", "na", "n/a", "none", "null", "nan"}


def _load_dataframe(file_bytes: bytes) -> "pd.DataFrame":
    """Try every supported format and return a DataFrame."""
    import pandas as pd

    errors = []
    for engine in ["openpyxl", "xlrd"]:
        try:
            return pd.read_excel(io.BytesIO(file_bytes), engine=engine)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception as exc:
            errors.append(f"csv/{enc}: {exc}")
    try:
        tables = __import__("pandas").read_html(io.BytesIO(file_bytes))
        if tables:
            return tables[0]
    except Exception as exc:
        errors.append(f"html: {exc}")

    drm_hint = ""
    if file_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        try:
            import olefile

            ole = olefile.OleFileIO(io.BytesIO(file_bytes))
            streams = [s for entry in ole.listdir() for s in entry]
            if "EncryptedPackage" in streams or "DRMEncryptedDataSpace" in streams:
                drm_hint = (
                    "\n\n**This file is protected by Microsoft Information Protection (DRM).**\n"
                    "To use it:\n"
                    "1. Open it in Excel\n"
                    "2. Go to **File → Save As** and save as a new `.xlsx` or CSV file\n"
                    "3. Upload the new unprotected file"
                )
        except Exception:
            pass
    raise ValueError("Could not open file." + drm_hint)


def _normalize_col_name(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _suggest_mapping(columns: list[str], field_key: str) -> str | None:
    field_aliases = {
        "survey_response_id": [
            "survey response id",
            "response id",
            "survey id",
            "submission id",
            "case id",
            "id",
        ],
        "question_name": [
            "question name",
            "question",
            "question text",
            "question label",
            "item",
        ],
        "response_value": [
            "response value",
            "response text",
            "response",
            "answer",
            "comment",
            "feedback",
            "value",
            "text",
        ],
        "fields_name": ["fields name", "field name", "field", "metadata", "attribute"],
    }

    normalized = {_normalize_col_name(c): c for c in columns}
    for alias in field_aliases.get(field_key, []):
        alias_norm = _normalize_col_name(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]

    for col in columns:
        ncol = _normalize_col_name(col)
        if any(alias in ncol for alias in field_aliases.get(field_key, [])):
            return col
    return None


def _is_blank_or_null_like(value: Any) -> bool:
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:
        pass

    s = str(value).strip()
    if not s:
        return True
    return s.lower() in _NULL_LIKE_VALUES


def _to_text(value: Any) -> str:
    if _is_blank_or_null_like(value):
        return ""
    return str(value).strip()


def _build_structured_rows(df: "pd.DataFrame", mapping: dict[str, str]) -> list[dict[str, str]]:
    """Build structured row records, skipping only blank/null response values."""
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        response_raw = row.get(mapping["response_value"])
        if _is_blank_or_null_like(response_raw):
            continue

        record = {
            "survey_response_id": _to_text(row.get(mapping["survey_response_id"])),
            "question_name": _to_text(row.get(mapping["question_name"])),
            "response_value": _to_text(response_raw),
        }

        optional_col = mapping.get("fields_name")
        if optional_col:
            record["fields_name"] = _to_text(row.get(optional_col))

        rows.append(record)

    return rows


# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Survey Sentiment Agent",
    page_icon="📊",
    layout="wide",
)


# ─── Load agent config / prompts ─────────────────────────────────────────────

@st.cache_resource
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "agent_config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_client(endpoint: str) -> AIProjectClient:
    return AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())


@st.cache_data
def load_prompt_text() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompt.txt")
    with open(prompt_path, encoding="utf-8") as f:
        return f.read().strip()


# ─── Session state ────────────────────────────────────────────────────────────

def init_session(client: AIProjectClient) -> None:
    if "thread_id" not in st.session_state:
        thread = client.agents.threads.create()
        st.session_state.thread_id = thread.id
        st.session_state.messages = []  # [{role, content}]


def reset_thread(client: AIProjectClient) -> None:
    thread = client.agents.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []
    st.session_state.pop("_pdf_cache", None)
    st.session_state.pop("_last_agent_reply", None)
    st.session_state.pop("_last_structured_survey_full_payload", None)
    st.success("New conversation started.")


# ─── SDK tool-call loop ──────────────────────────────────────────────────────

def _count_docs_from_args(fn_args: dict) -> int:
    """Count documents from tool arguments."""
    docs = fn_args.get("documents", [])
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except (json.JSONDecodeError, TypeError):
            return 1
    return len(docs) if isinstance(docs, list) else 0


def _count_rows_from_args(fn_args: dict) -> int:
    """Count structured rows from tool arguments when explicitly provided."""
    rows = fn_args.get("rows", [])
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except (json.JSONDecodeError, TypeError):
            return 1
    return len(rows) if isinstance(rows, list) else 0


def _extract_processed_count(fn_name: str, fn_args: dict, result: str) -> int:
    """Estimate rows/documents processed by analysis tool calls."""
    if fn_name == "analyze_sentiment":
        explicit = _count_docs_from_args(fn_args)
        if explicit:
            return explicit

        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return 0

        if isinstance(payload, dict):
            total = payload.get("total_documents")
            return int(total) if isinstance(total, int) else 0
        if isinstance(payload, list):
            return len(payload)
        return 0

    if fn_name == "analyze_structured_survey":
        explicit = _count_rows_from_args(fn_args)
        if explicit:
            return explicit

        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return 0

        if isinstance(payload, dict):
            coverage = payload.get("coverage", {})
            if isinstance(coverage, dict):
                processed = coverage.get("processed_rows")
                if isinstance(processed, int):
                    return processed
            enriched_rows = payload.get("enriched_rows")
            if isinstance(enriched_rows, list):
                return len(enriched_rows)
        return 0

    return 0


def _compute_rows_checksum(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        part = "|".join(
            [
                str(row.get("survey_response_id", "")),
                str(row.get("question_name", "")),
                str(row.get("response_value", "")),
            ]
        )
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _build_hard_proof(payload: dict[str, Any]) -> dict[str, Any]:
    coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {}
    
    # Handle both old format (full enriched_rows) and new format (enriched_rows_sample + enriched_rows_count)
    enriched_rows = payload.get("enriched_rows", []) if isinstance(payload.get("enriched_rows"), list) else []
    enriched_rows_sample = payload.get("enriched_rows_sample", []) if isinstance(payload.get("enriched_rows_sample"), list) else []
    enriched_rows_count = payload.get("enriched_rows_count", 0) if isinstance(payload.get("enriched_rows_count"), int) else 0
    
    # Use full rows if available (old format), else use count and sample (new format)
    rows_in_payload = enriched_rows if enriched_rows else enriched_rows_sample
    total_rows_processed = len(enriched_rows) if enriched_rows else enriched_rows_count

    total_rows = int(coverage.get("total_rows", 0) or 0)
    eligible_rows = int(coverage.get("eligible_rows", 0) or 0)
    processed_rows = int(coverage.get("processed_rows", 0) or 0)
    failed_rows = int(coverage.get("failed_rows", 0) or 0)

    return {
        "rows_total_input": total_rows,
        "rows_eligible_for_analysis": eligible_rows,
        "rows_processed": processed_rows,
        "rows_failed": failed_rows,
        "rows_kept_in_full_payload": len(rows_in_payload),
        "rows_lost_check": max(processed_rows - total_rows_processed, 0),
        "full_rows_sha256": _compute_rows_checksum(rows_in_payload),
    }


def _prepare_tool_output_for_submit(fn_name: str, result: str) -> str:
    """
    Keep full row-level payload locally, but submit a TPM-safe aggregate payload to the agent.
    This does NOT truncate analysis itself; it only reduces transport volume.
    """
    print(f"\n🔧 _prepare_tool_output_for_submit(fn_name={fn_name})")
    print(f"   result type: {type(result)}")
    print(f"   result size: {len(result) if isinstance(result, str) else 'N/A'} bytes")
    
    if fn_name != "analyze_structured_survey":
        print(f"   ⏭️ Not analyze_structured_survey, returning as-is")
        return result

    try:
        payload = json.loads(result)
        print(f"✅ JSON parsed successfully")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ JSON parse failed: {e}")
        return result

    print(f"   payload type: {type(payload)}")
    if isinstance(payload, dict):
        print(f"   Top-level keys: {sorted(payload.keys())}")
    else:
        print(f"   ⚠️ payload is not dict, is {type(payload)}")
        return result

    if payload.get("error"):
        print(f"   ⚠️ payload contains error: {payload.get('error')}")
        return result

    st.session_state["_last_structured_survey_full_payload"] = payload

    # Back up full enriched rows from language_tools for follow-up queries
    from language_tools import _last_enriched_rows
    if _last_enriched_rows:
        st.session_state["_all_enriched_rows"] = _last_enriched_rows
        print(f"💾 Backed up {len(_last_enriched_rows)} enriched rows to session state")

    # Debug: Inspect each candidate key path
    print(f"\n   Inspecting row candidates:")
    
    candidate_paths = [
        ("enriched_rows_sample", lambda p: p.get("enriched_rows_sample", [])),
        ("enriched_rows", lambda p: p.get("enriched_rows", [])),
        ("rows", lambda p: p.get("rows", [])),
        ("data", lambda p: p.get("data", [])),
    ]
    
    for key_name, extractor in candidate_paths:
        try:
            extracted = extractor(payload)
            is_list = isinstance(extracted, list)
            count = len(extracted) if is_list else "N/A (not list)"
            print(f"     {key_name}: {count} rows (type: {type(extracted).__name__})")
            if is_list and extracted and len(extracted) > 0:
                print(f"       First row: {extracted[0]}")
        except Exception as e:
            print(f"     {key_name}: ERROR - {e}")
    
    # The actual extraction
    enriched_rows_sample = payload.get("enriched_rows_sample", [])
    enriched_rows_count = payload.get("enriched_rows_count", 0)
    
    print(f"\n   Extracted for transport:")
    print(f"     enriched_rows_sample: {type(enriched_rows_sample).__name__}, {len(enriched_rows_sample) if isinstance(enriched_rows_sample, list) else '?'} items")
    print(f"     enriched_rows_count: {enriched_rows_count} (type: {type(enriched_rows_count).__name__})")
    
    if isinstance(enriched_rows_sample, list) and enriched_rows_sample:
        print(f"     First sample row keys: {sorted(enriched_rows_sample[0].keys())}")
        print(f"     First sample row: {enriched_rows_sample[0]}")

    proof = _build_hard_proof(payload)
    transport_payload = {
        "coverage": payload.get("coverage", {}),
        "survey_level_sentiment": payload.get("survey_level_sentiment", {}),
        "response_level_sentiment": payload.get("response_level_sentiment", {}),
        "main_cluster_breakdown": payload.get("main_cluster_breakdown", []),
        "evidence_summary": payload.get("evidence_summary", {}),
        "enriched_rows_sample": enriched_rows_sample,
        "enriched_rows_count": enriched_rows_count,
        "hard_proof": proof,
        "analysis_completed_on_full_dataset": True,
        "total_rows_analyzed": enriched_rows_count,
        "reference_sample_only": True,
        "notes": {
            "description": "All rows were analyzed; sample provided for context",
            "sample_size": len(enriched_rows_sample),
            "total_rows_analyzed": enriched_rows_count,
        },
    }
    
    print(f"\n🚀 FINAL TRANSPORT PAYLOAD:")
    print(f"   enriched_rows_sample in transport: {len(transport_payload.get('enriched_rows_sample', []))} rows")
    print(f"   enriched_rows_count in transport: {transport_payload.get('enriched_rows_count', 0)}")
    if transport_payload.get('enriched_rows_sample'):
        row1 = transport_payload['enriched_rows_sample'][0]
        print(f"   Sample row 1 in transport: {row1}")
    else:
        print(f"   ⚠️ NO ROWS IN TRANSPORT SAMPLE")
    
    result_json = json.dumps(transport_payload, ensure_ascii=False)
    print(f"   Transport JSON size: {len(result_json)} bytes\n")
    return result_json


def _append_hard_proof_block(reply: str) -> str:
    payload = st.session_state.get("_last_structured_survey_full_payload")
    if not isinstance(payload, dict):
        return reply

    proof = _build_hard_proof(payload)
    proof_md = (
        "\n\n---\n"
        "### Hard Proof Block\n"
        f"- Rows total input: **{proof['rows_total_input']}**\n"
        f"- Rows eligible for analysis: **{proof['rows_eligible_for_analysis']}**\n"
        f"- Rows processed: **{proof['rows_processed']}**\n"
        f"- Rows failed: **{proof['rows_failed']}**\n"
        f"- Rows kept in full payload: **{proof['rows_kept_in_full_payload']}**\n"
        f"- Rows lost check: **{proof['rows_lost_check']}**\n"
        f"- Full rows SHA-256: `{proof['full_rows_sha256']}`\n"
    )
    return reply + proof_md


def _build_tool_outputs(run, client: AIProjectClient, thread_id: str, status_widget=None):
    """Execute Language SDK function tools and submit outputs."""
    from language_tools import TOOL_DISPATCH

    tool_outputs = []
    calls = run.required_action.submit_tool_outputs.tool_calls
    for i, call in enumerate(calls, 1):
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments or "{}")
        if status_widget:
            status_widget.update(
                label=f"Language Tools: {fn_name} ({i}/{len(calls)})",
                state="running",
            )
        t0 = time.time()
        try:
            fn = TOOL_DISPATCH.get(fn_name)
            if fn is None:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                result = fn(**fn_args)
        except Exception as exc:  # noqa: BLE001
            result = json.dumps({"error": str(exc)})

        # Debug: Show raw tool result before processing
        if fn_name == "analyze_structured_survey":
            print(f"\n🔨 RAW TOOL RESULT from {fn_name}:")
            print(f"   Type: {type(result)}")
            print(f"   Size: {len(result) if isinstance(result, str) else 'N/A'} bytes")
            if isinstance(result, str) and len(result) < 5000:
                try:
                    parsed = json.loads(result)
                    print(f"   Parsed keys: {sorted(parsed.keys()) if isinstance(parsed, dict) else 'not-a-dict'}")
                    if isinstance(parsed, dict):
                        print(f"   enriched_rows_sample: {len(parsed.get('enriched_rows_sample', []))} rows")
                        print(f"   enriched_rows_count: {parsed.get('enriched_rows_count', 0)}")
                except:
                    print(f"   Could not parse as JSON")
            else:
                print(f"   (Result too large to show, size={len(result)})")


        processed = _extract_processed_count(fn_name, fn_args, result)
        if processed:
            st.session_state.setdefault("_rows_processed", 0)
            st.session_state["_rows_processed"] += processed
            if status_widget:
                status_widget.update(
                    label=(
                        f"Language Tools: {fn_name} ({i}/{len(calls)}) "
                        f"- {st.session_state['_rows_processed']} rows processed"
                    ),
                    state="running",
                )

        output_for_submit = _prepare_tool_output_for_submit(fn_name, result)
        print(f"  ⚙️ {fn_name} took {time.time() - t0:.1f}s")
        tool_outputs.append(ToolOutput(tool_call_id=call.id, output=output_for_submit))

    # Debug: print tool output sizes before submitting
    for t in tool_outputs:
        output_text = t.output if hasattr(t, 'output') else t.get("output", "")
        print("TOOL OUTPUT SIZE:", len(output_text))
    print("TOTAL TOOL OUTPUT BYTES:", sum(len(t.output if hasattr(t, 'output') else t.get("output", "")) for t in tool_outputs))

    # Submit tool outputs with diagnostic error handling
    try:
        return client.agents.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs,
        )
    except Exception:
        print("submit_tool_outputs failed")
        print("tool_outputs count:", len(tool_outputs))
        for i, t in enumerate(tool_outputs):
            output_size = len(t.output if hasattr(t, 'output') else t.get("output", ""))
            print(f"tool_output[{i}] size:", output_size)
        print("TOTAL TOOL OUTPUT BYTES:", sum(len(t.output if hasattr(t, 'output') else t.get("output", "")) for t in tool_outputs))
        raise


def _wait_for_run(client: AIProjectClient, thread_id: str, run, status_widget=None, task: str = "chat") -> object:
    """Poll run to completion, handling SDK function tool calls."""
    terminal = {"completed", "failed", "cancelled", "expired"}
    deadline = time.time() + 600
    t_start = time.time()
    t_phase = t_start

    phase1_labels = {
        "fabric": "Foundry Agent → Fabric Agent: querying data",
        "file": "Foundry Agent: processing file",
        "chat": "Foundry Agent: thinking",
    }
    current_label = phase1_labels.get(task, phase1_labels["chat"])
    tools_ran = False

    while run.status not in terminal:
        if time.time() > deadline:
            try:
                client.agents.runs.cancel(thread_id=thread_id, run_id=run.id)
            except Exception:
                pass
            run._data["status"] = "failed"
            run._data["last_error"] = {"code": "timeout", "message": "Run exceeded 10-minute timeout."}
            rows_processed = int(st.session_state.pop("_rows_processed", 0))
            return run, rows_processed

        elapsed = int(time.time() - t_phase)
        if status_widget:
            status_widget.update(label=f"{current_label}... ({elapsed}s)", state="running")
        time.sleep(0.5)
        run = client.agents.runs.get(thread_id=thread_id, run_id=run.id)
        if run.status == "requires_action":
            print(f"⏱️ requires_action at {time.time() - t_start:.1f}s")
            t_phase = time.time()
            tools_ran = True
            run = _build_tool_outputs(run, client, thread_id, status_widget=status_widget)
            t_phase = time.time()
            current_label = "Foundry Agent: processing results"
        elif tools_ran and current_label != "Foundry Agent: generating response":
            current_label = "Foundry Agent: generating response"
            t_phase = time.time()

    total = int(time.time() - t_start)
    rows_processed = int(st.session_state.pop("_rows_processed", 0))
    print(f"⏱️ Run completed in {total}s (status: {run.status}, rows: {rows_processed})")
    if status_widget:
        done_label = f"Done ({total}s total)"
        if rows_processed:
            done_label += f" - {rows_processed} rows processed"
        status_widget.update(label=done_label, state="complete")
    return run, rows_processed


def _cancel_active_runs(client: AIProjectClient, thread_id: str) -> None:
    """Cancel any runs that are still in a non-terminal state on this thread."""
    terminal = {"completed", "failed", "cancelled", "expired"}
    try:
        runs = client.agents.runs.list(thread_id=thread_id)
        for run in runs:
            if run.status not in terminal:
                try:
                    client.agents.runs.cancel(thread_id=thread_id, run_id=run.id)
                    for _ in range(10):
                        time.sleep(0.5)
                        r = client.agents.runs.get(thread_id=thread_id, run_id=run.id)
                        if r.status in terminal:
                            break
                except Exception:
                    pass
    except Exception:
        pass


def send_message(
    client: AIProjectClient,
    agent_id: str,
    thread_id: str,
    content: str,
    status_widget=None,
    task: str = "chat",
) -> str:
    if task == "file":
        st.session_state.pop("_last_structured_survey_full_payload", None)
        st.session_state.pop("_pdf_cache", None)

    _cancel_active_runs(client, thread_id)
    client.agents.messages.create(thread_id=thread_id, role="user", content=content)

    if status_widget:
        status_widget.update(label="Starting Foundry Agent...", state="running")
    run = client.agents.runs.create(thread_id=thread_id, agent_id=agent_id)
    run, rows_processed = _wait_for_run(client, thread_id, run, status_widget=status_widget, task=task)

    if run.status == "failed":
        return f"❌ Run failed: {run.last_error}"

    last = client.agents.messages.get_last_message_text_by_role(thread_id=thread_id, role="assistant")
    reply = last.text.value if last else "(no response)"
    if rows_processed:
        reply += f"\n\n---\n*Language service processed: {rows_processed} rows*"

    if task == "file":
        reply = _append_hard_proof_block(reply)

    return reply


# ─── PDF export ───────────────────────────────────────────────────────────────

def _build_pdf_cached(filename: str, agent_reply: str) -> tuple[bytes | None, str]:
    """
    Build PDF bytes once and cache them in session state.
    Returns (pdf_bytes, output_filename) or (None, "") on failure.
    Re-builds only when the agent reply changes.
    """
    payload = st.session_state.get("_last_structured_survey_full_payload")
    if not isinstance(payload, dict):
        return None, ""

    # Use reply length + filename as a cheap cache key
    cache_key = (len(agent_reply), filename)
    cached = st.session_state.get("_pdf_cache")
    if cached and cached.get("key") == cache_key:
        return cached["bytes"], cached["filename"]

    try:
        from report_pdf import build_pdf

        clean_reply = re.sub(r"\n---\n### Hard Proof Block.*$", "", agent_reply, flags=re.DOTALL)
        clean_reply = re.sub(r"\n---\n\*Language service processed:.*$", "", clean_reply, flags=re.DOTALL)

        pdf_bytes = build_pdf(
            payload=payload,
            agent_reply=clean_reply.strip(),
            filename=filename,
        )
        stem = re.sub(r"\.[^.]+$", "", filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_name = f"{stem}_sentiment_report_{ts}.pdf"

        st.session_state["_pdf_cache"] = {"key": cache_key, "bytes": pdf_bytes, "filename": out_name}
        return pdf_bytes, out_name

    except Exception as exc:
        st.warning(f"PDF generation failed: {exc}")
        return None, ""


def _offer_pdf_download(filename: str, agent_reply: str, label: str = "📄 Download PDF Report") -> None:
    """Render a persistent PDF download button in the sidebar using cached bytes."""
    pdf_bytes, out_name = _build_pdf_cached(filename, agent_reply)
    if not pdf_bytes:
        return
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=out_name,
        # octet-stream forces the browser to save the file instead of opening it
        mime="application/octet-stream",
        use_container_width=True,
        key="sidebar_pdf_download",
    )


# ─── Rich reply renderer ─────────────────────────────────────────────────────

_CHAT_STYLES = """
<style>
/* ── Tables ──────────────────────────────────────────── */
div[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.855rem;
    margin: 0.6rem 0 1.1rem 0;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 5px rgba(0,0,0,0.08);
}
div[data-testid="stChatMessage"] th {
    background: #1B2A4A;
    color: #e8edf5;
    padding: 0.5rem 0.8rem;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
div[data-testid="stChatMessage"] td {
    padding: 0.42rem 0.8rem;
    border-bottom: 1px solid #e8edf0;
    vertical-align: top;
    line-height: 1.45;
}
div[data-testid="stChatMessage"] tr:nth-child(even) td { background: #f6f9fc; }
div[data-testid="stChatMessage"] tr:last-child td { border-bottom: none; }
div[data-testid="stChatMessage"] tr:hover td { background: #edf2fa; }
/* ── Headings ─────────────────────────────────────────── */
div[data-testid="stChatMessage"] h2 {
    color: #1B2A4A;
    font-size: 1.1rem;
    font-weight: 700;
    border-bottom: 2px solid #3498DB;
    padding-bottom: 0.3rem;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
}
div[data-testid="stChatMessage"] h3 {
    color: #2C3E50;
    font-size: 0.97rem;
    font-weight: 600;
    margin-top: 0.9rem;
    margin-bottom: 0.35rem;
}
/* ── Blockquotes / verbatim quotes ───────────────────── */
div[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid #3498DB;
    padding: 0.45rem 0.9rem;
    margin: 0.4rem 0;
    background: #f0f5fc;
    border-radius: 0 4px 4px 0;
    color: #2C3E50;
    font-style: italic;
}
/* ── Code ────────────────────────────────────────────── */
div[data-testid="stChatMessage"] code {
    background: #eef2f7;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-size: 0.84rem;
}
/* ── HR ──────────────────────────────────────────────── */
div[data-testid="stChatMessage"] hr {
    border: none;
    border-top: 1px solid #dde3ec;
    margin: 0.75rem 0;
}
/* ── Lists ───────────────────────────────────────────── */
div[data-testid="stChatMessage"] ol,
div[data-testid="stChatMessage"] ul {
    padding-left: 1.4rem;
    margin: 0.25rem 0 0.65rem 0;
}
div[data-testid="stChatMessage"] li { margin-bottom: 0.22rem; line-height: 1.5; }
</style>
"""


def _inject_chat_styles() -> None:
    """Inject CSS once per page load to improve chat rendering."""
    st.markdown(_CHAT_STYLES, unsafe_allow_html=True)


def _render_agent_reply(reply: str) -> None:
    """Render an agent reply: split into ## sections with visual dividers."""
    if not reply or not reply.strip():
        return
    sections = re.split(r"\n(?=## )", reply.strip())
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        st.markdown(section)
        if i < len(sections) - 1:
            st.divider()


# ─── UI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    client = get_client(config["endpoint"])
    init_session(client)
    _inject_chat_styles()

    with st.sidebar:
        st.title("📊 Survey Analysis")
        st.caption(f"Agent: `{config['agent_name']}`")
        st.caption(f"Model: `{config['model']}`")
        st.caption(f"Tools: `{config.get('tool_mode', 'sdk').upper()}`")
        st.divider()

        fabric_enabled = bool(
            os.environ.get("FABRIC_SQL_ENDPOINT")
            and os.environ.get("FABRIC_DATABASE")
        )
        if fabric_enabled:
            data_source = st.radio(
                "Data Source",
                ["Local File", "Fabric Warehouse"],
                help="Choose between uploading a local file or querying Fabric warehouse",
            )
        else:
            data_source = "Local File"

        st.subheader("📁 " + data_source)

        if data_source == "Local File":
            uploaded_file = st.file_uploader(
                "Excel / CSV file",
                type=["xlsx", "xls", "csv"],
                help="Upload a survey file to analyse",
            )
        else:
            uploaded_file = None

        file_bytes = None
        if uploaded_file:
            file_bytes = uploaded_file.read()
            try:
                df_preview = _load_dataframe(file_bytes)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            all_cols = [str(c) for c in df_preview.columns]
            df_preview.columns = all_cols

            # Auto-map columns; only show manual selectors if auto-mapping fails
            mapping: dict[str, str | None] = {}
            for field_key in _REQUIRED_FIELD_KEYS:
                mapping[field_key] = _suggest_mapping(all_cols, field_key)

            optional_key = _OPTIONAL_FIELD_KEYS[0]
            mapping[optional_key] = _suggest_mapping(all_cols, optional_key)

            auto_mapped = all(mapping.get(k) for k in _REQUIRED_FIELD_KEYS)
            if not auto_mapped:
                st.markdown("**Map source columns to required survey fields**")
                for field_key in _REQUIRED_FIELD_KEYS:
                    suggestion = mapping[field_key]
                    options = ["— Select column —", *all_cols]
                    default_idx = options.index(suggestion) if suggestion and suggestion in options else 0
                    selected = st.selectbox(
                        f"{field_key} *",
                        options=options,
                        index=default_idx,
                        key=f"map_{field_key}",
                    )
                    mapping[field_key] = None if selected == options[0] else selected

                optional_options = ["— Not provided —", *all_cols]
                optional_idx = optional_options.index(mapping[optional_key]) if mapping.get(optional_key) and mapping[optional_key] in optional_options else 0
                optional_selected = st.selectbox(
                    f"{optional_key} (optional)",
                    options=optional_options,
                    index=optional_idx,
                    key=f"map_{optional_key}",
                )
                mapping[optional_key] = None if optional_selected == optional_options[0] else optional_selected

            if all(mapping.get(k) for k in _REQUIRED_FIELD_KEYS):
                preview_cols = [mapping[k] for k in _REQUIRED_FIELD_KEYS if mapping[k]]
                if mapping.get(optional_key):
                    preview_cols.append(mapping[optional_key])
                st.caption("Preview of mapped columns")
                st.dataframe(df_preview[preview_cols].head(5), use_container_width=True)

            if st.button("Analyse File", type="primary", use_container_width=True):
                missing_required = [k for k in _REQUIRED_FIELD_KEYS if not mapping.get(k)]
                if missing_required:
                    st.error(f"Missing required mappings: {', '.join(missing_required)}")
                    st.stop()

                required_mapped_values = [mapping[k] for k in _REQUIRED_FIELD_KEYS]
                if len(set(required_mapped_values)) != len(required_mapped_values):
                    st.error("Each required field must map to a different source column.")
                    st.stop()

                structured_rows = _build_structured_rows(df_preview, mapping)  # type: ignore[arg-type]
                if not structured_rows:
                    st.warning("No eligible rows found. Only rows with non-empty response_value are processed.")
                    st.stop()

                # Store rows in session state AND global backup for tool access
                print(f"\n📤 UPLOAD: Storing {len(structured_rows)} rows")
                if structured_rows:
                    print(f"   Row 1: {structured_rows[0]}")
                    print(f"   Columns: {list(structured_rows[0].keys())}")
                st.session_state["_pending_structured_rows"] = structured_rows
                from language_tools import set_pending_structured_rows
                set_pending_structured_rows(structured_rows)
                print(f"✅ Rows stored in session + global backup\n")

                prompt_text = load_prompt_text()
                header = (
                    f"File: **{uploaded_file.name}**\n"
                    f"Eligible structured rows: **{len(structured_rows)}**"
                )
                user_msg = (
                    f"{header}\n\n"
                    f"{prompt_text}\n\n"
                    "Call analyze_structured_survey with NO arguments so it uses the full uploaded dataset."
                )

                st.session_state.messages.append({"role": "user", "content": user_msg})
                st.session_state["_pending_file_msg"] = user_msg
                st.session_state["_last_filename"] = uploaded_file.name
                st.rerun()

        # ── Fabric Warehouse flow ──
        if data_source == "Fabric Warehouse":
            from fabric_warehouse import list_views, list_columns, load_view, load_distinct_values, count_rows

            default_view = os.environ.get("FABRIC_VIEW", "")
            try:
                available_views = list_views()
            except Exception as exc:
                st.error(f"Could not connect to Fabric Warehouse: {exc}")
                available_views = []

            if available_views:
                default_idx = 0
                if default_view and default_view in available_views:
                    default_idx = available_views.index(default_view)
                selected_view = st.selectbox(
                    "Warehouse View",
                    options=available_views,
                    index=default_idx,
                    help="Select the database view containing survey responses",
                )

                if selected_view:
                    try:
                        df_fabric_preview = load_view(selected_view, limit=5)
                        fabric_cols = [str(c) for c in df_fabric_preview.columns]
                    except Exception as exc:
                        st.error(f"Could not load view: {exc}")
                        fabric_cols = []
                        df_fabric_preview = None

                    if fabric_cols:
                        # ── Pre-analysis filters ──
                        st.markdown("---")
                        st.markdown("**🔍 Filter data before analysis**")

                        # Detect date-like columns
                        _date_hints = {"date", "time", "opened", "closed", "created", "completion"}
                        date_cols = [c for c in fabric_cols if any(h in c.lower() for h in _date_hints)]
                        # Detect categorical filter columns
                        _cat_hints = {
                            "product": "RSA Product Set",
                            "version": "RSA Version/Condition",
                            "region": "Account Global Region",
                            "severity": "Severity",
                        }
                        cat_cols_found: dict[str, str] = {}
                        for hint, expected in _cat_hints.items():
                            if expected in fabric_cols:
                                cat_cols_found[hint] = expected
                            else:
                                match = next((c for c in fabric_cols if hint in c.lower()), None)
                                if match:
                                    cat_cols_found[hint] = match

                        fabric_filters: dict = {}

                        # Date range filter
                        if date_cols:
                            date_col = st.selectbox(
                                "Date column",
                                options=date_cols,
                                index=0,
                                key="fab_date_col",
                            )
                            col_start, col_end = st.columns(2)
                            from datetime import date as _date, timedelta
                            with col_start:
                                start_date = st.date_input(
                                    "From",
                                    value=_date.today() - timedelta(days=365),
                                    key="fab_start_date",
                                )
                            with col_end:
                                end_date = st.date_input(
                                    "To",
                                    value=_date.today(),
                                    key="fab_end_date",
                                )
                            if start_date and end_date:
                                fabric_filters[date_col] = (str(start_date), str(end_date))

                        # Categorical filters (product, version, region, etc.)
                        for hint, col_name in cat_cols_found.items():
                            cache_key = f"_fab_distinct_{col_name}"
                            if cache_key not in st.session_state:
                                try:
                                    st.session_state[cache_key] = load_distinct_values(selected_view, col_name)
                                except Exception:
                                    st.session_state[cache_key] = []
                            distinct_vals = st.session_state[cache_key]
                            if distinct_vals:
                                chosen = st.multiselect(
                                    col_name,
                                    options=distinct_vals,
                                    default=[],
                                    key=f"fab_filter_{col_name}",
                                )
                                if chosen:
                                    fabric_filters[col_name] = chosen

                        # Show filtered preview
                        if fabric_filters:
                            try:
                                df_fabric_preview = load_view(selected_view, limit=5, filters=fabric_filters)
                                fabric_cols = [str(c) for c in df_fabric_preview.columns]
                                n_filtered = count_rows(selected_view, filters=fabric_filters)
                                st.info(f"**{n_filtered:,}** rows match the current filters")
                            except Exception as exc:
                                st.warning(f"Filter preview error: {exc}")

                        st.markdown("---")

                        # Auto-map known Fabric columns — no manual mapping needed
                        fabric_mapping: dict[str, str | None] = {}
                        for field_key in _REQUIRED_FIELD_KEYS:
                            suggestion = _suggest_mapping(fabric_cols, field_key)
                            fabric_mapping[field_key] = suggestion
                        optional_key = _OPTIONAL_FIELD_KEYS[0]
                        fabric_mapping[optional_key] = _suggest_mapping(fabric_cols, optional_key)

                        # If auto-mapping fails for any required field, fall back to manual selectors
                        auto_mapped = all(fabric_mapping.get(k) for k in _REQUIRED_FIELD_KEYS)
                        if not auto_mapped:
                            st.markdown("**Map warehouse columns to required survey fields**")
                            for field_key in _REQUIRED_FIELD_KEYS:
                                options = ["— Select column —", *fabric_cols]
                                default_fld = options.index(fabric_mapping[field_key]) if fabric_mapping.get(field_key) and fabric_mapping[field_key] in options else 0
                                selected = st.selectbox(
                                    f"{field_key} *",
                                    options=options,
                                    index=default_fld,
                                    key=f"fab_map_{field_key}",
                                )
                                fabric_mapping[field_key] = None if selected == options[0] else selected

                            fab_opt_options = ["— Not provided —", *fabric_cols]
                            fab_opt_idx = fab_opt_options.index(fabric_mapping[optional_key]) if fabric_mapping.get(optional_key) and fabric_mapping[optional_key] in fab_opt_options else 0
                            fab_opt_selected = st.selectbox(
                                f"{optional_key} (optional)",
                                options=fab_opt_options,
                                index=fab_opt_idx,
                                key=f"fab_map_{optional_key}",
                            )
                            fabric_mapping[optional_key] = None if fab_opt_selected == fab_opt_options[0] else fab_opt_selected

                        if df_fabric_preview is not None and all(fabric_mapping.get(k) for k in _REQUIRED_FIELD_KEYS):
                            preview_c = [fabric_mapping[k] for k in _REQUIRED_FIELD_KEYS if fabric_mapping[k]]
                            if fabric_mapping.get(optional_key):
                                preview_c.append(fabric_mapping[optional_key])
                            st.caption("Preview (first 5 rows)")
                            st.dataframe(df_fabric_preview[preview_c].head(5), use_container_width=True)

                        if st.button("Analyse Warehouse Data", type="primary", use_container_width=True):
                            missing_req = [k for k in _REQUIRED_FIELD_KEYS if not fabric_mapping.get(k)]
                            if missing_req:
                                st.error(f"Missing required mappings: {', '.join(missing_req)}")
                                st.stop()

                            req_vals = [fabric_mapping[k] for k in _REQUIRED_FIELD_KEYS]
                            if len(set(req_vals)) != len(req_vals):
                                st.error("Each required field must map to a different column.")
                                st.stop()

                            with st.spinner("Loading full dataset from Fabric Warehouse..."):
                                df_fabric_full = load_view(selected_view, filters=fabric_filters if fabric_filters else None)

                            structured_rows = _build_structured_rows(df_fabric_full, fabric_mapping)  # type: ignore[arg-type]
                            if not structured_rows:
                                st.warning("No eligible rows found in the warehouse view.")
                                st.stop()

                            print(f"\n📤 FABRIC: Storing {len(structured_rows)} rows from {selected_view}")
                            st.session_state["_pending_structured_rows"] = structured_rows
                            from language_tools import set_pending_structured_rows
                            set_pending_structured_rows(structured_rows)

                            prompt_text = load_prompt_text()
                            filter_summary = ""
                            if fabric_filters:
                                parts = []
                                for fc, fv in fabric_filters.items():
                                    if isinstance(fv, tuple):
                                        parts.append(f"{fc}: {fv[0]} → {fv[1]}")
                                    elif isinstance(fv, list):
                                        parts.append(f"{fc}: {', '.join(fv)}")
                                    else:
                                        parts.append(f"{fc}: {fv}")
                                filter_summary = " · Filters: " + " | ".join(parts)
                            header = (
                                f"Source: **Fabric Warehouse** · View: **{selected_view}**{filter_summary}\n"
                                f"Eligible structured rows: **{len(structured_rows)}**"
                            )
                            user_msg = (
                                f"{header}\n\n"
                                f"{prompt_text}\n\n"
                                "Call analyze_structured_survey with NO arguments so it uses the full uploaded dataset."
                            )

                            st.session_state.messages.append({"role": "user", "content": user_msg})
                            st.session_state["_pending_file_msg"] = user_msg
                            st.session_state["_last_filename"] = selected_view.replace(".", "_")
                            st.rerun()

        st.divider()
        if st.button("🗑️ New Conversation", use_container_width=True):
            reset_thread(client)
            st.rerun()

        # PDF download — shown whenever a completed analysis exists
        if st.session_state.get("_last_structured_survey_full_payload") and st.session_state.get("_last_agent_reply"):
            st.divider()
            _offer_pdf_download(
                st.session_state.get("_last_filename", "survey"),
                st.session_state["_last_agent_reply"],
                label="📄 Download PDF Report",
            )

        st.divider()
        st.caption(f"Thread: `{st.session_state.get('thread_id', '...')}`")

    st.title("Survey Sentiment Agent")
    st.caption("Powered by Azure AI Foundry · Azure Language MCP · GPT-4o")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_agent_reply(msg["content"])
            else:
                st.markdown(msg["content"])

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            if fabric_enabled:
                st.markdown(
                    "👋 Hi! I'm your Survey Analysis Agent. You can:\n\n"
                    "- **Upload an Excel file** and map required survey fields\n"
                    "- **Connect to Fabric Warehouse** and analyse live data\n"
                    "- **Type a message** below — paste responses directly or ask questions"
                )
            else:
                st.markdown(
                    "👋 Hi! I'm your Survey Analysis Agent. You can:\n\n"
                    "- **Upload an Excel file** and map required survey fields\n"
                    "- **Type a message** below — paste responses directly or ask questions"
                )

    if st.session_state.get("_pending_file_msg"):
        user_msg = st.session_state.pop("_pending_file_msg")
        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, user_msg, status_widget=status, task="file")
            _render_agent_reply(reply)
            # Store the pure agent reply (no prompt) for PDF export
            st.session_state["_last_agent_reply"] = reply
            # Pre-build and cache the PDF now so the sidebar button is instant
            _build_pdf_cached(st.session_state.get("_last_filename", "survey"), reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("_pending_fabric_msg"):
        user_msg = st.session_state.pop("_pending_fabric_msg")
        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, user_msg, status_widget=status, task="fabric")
            _render_agent_reply(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if prompt := st.chat_input("Ask the agent or paste survey responses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, prompt, status_widget=status, task="chat")
            _render_agent_reply(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
