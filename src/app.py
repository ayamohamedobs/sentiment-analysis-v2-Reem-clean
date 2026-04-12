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
import sys
import time
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
    deadline = time.time() + 300
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
            run._data["last_error"] = {"code": "timeout", "message": "Run exceeded 5-minute timeout."}
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


# ─── UI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    client = get_client(config["endpoint"])
    init_session(client)

    with st.sidebar:
        st.title("📊 Survey Analysis")
        st.caption(f"Agent: `{config['agent_name']}`")
        st.caption(f"Model: `{config['model']}`")
        st.caption(f"Tools: `{config.get('tool_mode', 'sdk').upper()}`")
        st.divider()

        fabric_enabled = bool(os.environ.get("FABRIC_CONNECTION_NAME"))
        if fabric_enabled:
            data_source = st.radio(
                "Data Source",
                ["Local File", "Fabric Semantic Model"],
                help="Choose between uploading a local file or querying Fabric data",
            )
        else:
            data_source = "Local File"

        st.subheader("📁 " + data_source)

        fabric_query = None

        if data_source == "Local File":
            uploaded_file = st.file_uploader(
                "Excel / CSV file",
                type=["xlsx", "xls", "csv"],
                help="Upload a survey file to analyse",
            )
        else:
            uploaded_file = None
            fabric_query = st.text_area(
                "Natural Language Query",
                placeholder="e.g., Get all survey responses from last quarter",
                help="Describe what data you want to analyze from your Fabric semantic model",
                height=100,
            )

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

            st.markdown("**Map source columns to required survey fields**")
            mapping: dict[str, str | None] = {}
            for field_key in _REQUIRED_FIELD_KEYS:
                suggestion = _suggest_mapping(all_cols, field_key)
                options = ["— Select column —", *all_cols]
                default_idx = options.index(suggestion) if suggestion in options else 0
                selected = st.selectbox(
                    f"{field_key} *",
                    options=options,
                    index=default_idx,
                    key=f"map_{field_key}",
                )
                mapping[field_key] = None if selected == options[0] else selected

            optional_key = _OPTIONAL_FIELD_KEYS[0]
            optional_suggestion = _suggest_mapping(all_cols, optional_key)
            optional_options = ["— Not provided —", *all_cols]
            optional_idx = optional_options.index(optional_suggestion) if optional_suggestion in optional_options else 0
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
                st.rerun()

        if data_source == "Fabric Semantic Model" and fabric_query and st.button("Query & Analyze", type="primary", use_container_width=True):
            user_msg = (
                f"Use the fabric_dataagent tool now to query the semantic model: {fabric_query}\n\n"
                "After retrieving the data, analyze it using the Language tools "
                "(analyze_sentiment, extract_key_phrases, recognize_entities — batch up to 10 per call).\n\n"
                "Present results in this structure:\n"
                "1. Customer Sentiment Overview (executive summary)\n"
                "2. Where Sentiment Breaks Down (table with themes and sentiment percentages)\n"
                "3. Key Drivers of Negative Sentiment (table with top 5 issue clusters)\n"
                "4. Key Drivers of Positive Sentiment (table with top strengths)\n"
                "5. Insight-Driven Recommendations (numbered, with Why/Recommendation format)"
            )

            st.session_state.messages.append({"role": "user", "content": f"Query: {fabric_query}"})
            st.session_state["_pending_fabric_msg"] = user_msg
            st.rerun()

        st.divider()
        if st.button("🗑️ New Conversation", use_container_width=True):
            reset_thread(client)
            st.rerun()

        st.divider()
        st.caption(f"Thread: `{st.session_state.get('thread_id', '...')}`")

    st.title("Survey Sentiment Agent")
    st.caption("Powered by Azure AI Foundry · Azure Language MCP · GPT-4o")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            if fabric_enabled:
                st.markdown(
                    "👋 Hi! I'm your Survey Analysis Agent. You can:\n\n"
                    "- **Upload an Excel file** and map required survey fields\n"
                    "- **Query Fabric data** using natural language\n"
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
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("_pending_fabric_msg"):
        user_msg = st.session_state.pop("_pending_fabric_msg")
        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, user_msg, status_widget=status, task="fabric")
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if prompt := st.chat_input("Ask the agent or paste survey responses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, prompt, status_widget=status, task="chat")
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
