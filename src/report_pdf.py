"""
PDF report generator for Survey Sentiment Analysis.

Produces a clean, executive-ready PDF from the agent's markdown output.
All data flows from the agent reply -- no payload reconstruction.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

# --- Brand palette ---
_BRAND_DARK   = colors.HexColor("#1B2A4A")
_BRAND_MID    = colors.HexColor("#2C3E50")
_BRAND_ACCENT = colors.HexColor("#3498DB")
_GREEN        = colors.HexColor("#27AE60")
_GREEN_LIGHT  = colors.HexColor("#EAFAF1")
_YELLOW       = colors.HexColor("#F39C12")
_YELLOW_LIGHT = colors.HexColor("#FEF9E7")
_RED          = colors.HexColor("#E74C3C")
_RED_LIGHT    = colors.HexColor("#FDEDEC")
_LIGHT_BG     = colors.HexColor("#F8F9FB")
_BORDER       = colors.HexColor("#DFE6E9")
_TEXT         = colors.HexColor("#2D3436")
_MUTED        = colors.HexColor("#95A5A6")
_WHITE        = colors.white

_SENTIMENT_COLOUR = {
    "Positive": _GREEN, "Neutral": _YELLOW, "Negative": _RED,
    "positive": _GREEN, "neutral": _YELLOW, "negative": _RED,
}


# --- Custom flowables ---

class _CoverBar(Flowable):
    def __init__(self, width: float, height: float = 6):
        super().__init__()
        self.bar_width = width
        self.height = height
        self.width = width

    def draw(self):
        self.canv.setFillColor(_BRAND_DARK)
        self.canv.rect(0, 0, self.bar_width, self.height, fill=1, stroke=0)


class _AccentLine(Flowable):
    def __init__(self, width: float, colour=None):
        super().__init__()
        self.bar_width = width
        self.colour = colour or _BRAND_ACCENT
        self.width = width
        self.height = 2

    def draw(self):
        self.canv.setFillColor(self.colour)
        self.canv.rect(0, 0, self.bar_width, 1.5, fill=1, stroke=0)


# --- Styles ---

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "rpt_title", parent=base["Title"],
            fontSize=26, textColor=_BRAND_DARK, spaceAfter=2,
            fontName="Helvetica-Bold", leading=30,
        ),
        "subtitle": ParagraphStyle(
            "rpt_subtitle", parent=base["Normal"],
            fontSize=12, textColor=_MUTED, spaceAfter=2, fontName="Helvetica",
        ),
        "h1": ParagraphStyle(
            "rpt_h1", parent=base["Heading1"],
            fontSize=14, textColor=_BRAND_DARK, spaceBefore=16, spaceAfter=2,
            fontName="Helvetica-Bold", borderPadding=0,
        ),
        "h2": ParagraphStyle(
            "rpt_h2", parent=base["Heading2"],
            fontSize=11, textColor=_BRAND_MID, spaceBefore=10, spaceAfter=3,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "rpt_h3", parent=base["Heading3"],
            fontSize=10, textColor=_BRAND_ACCENT, spaceBefore=8, spaceAfter=2,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "rpt_body", parent=base["Normal"],
            fontSize=9, textColor=_TEXT, spaceAfter=4, fontName="Helvetica",
            leading=14, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "rpt_bullet", parent=base["Normal"],
            fontSize=9, textColor=_TEXT, spaceAfter=3, fontName="Helvetica",
            leading=13, leftIndent=14, bulletIndent=4,
        ),
        "small": ParagraphStyle(
            "rpt_small", parent=base["Normal"],
            fontSize=7.5, textColor=_MUTED, spaceAfter=2, fontName="Helvetica",
        ),
        "verbatim": ParagraphStyle(
            "rpt_verbatim", parent=base["Normal"],
            fontSize=8.5, textColor=colors.HexColor("#555555"),
            spaceAfter=4, fontName="Helvetica-Oblique",
            leftIndent=12, leading=12, borderPadding=(2, 4, 2, 4),
            borderColor=_BORDER, borderWidth=0, backColor=colors.HexColor("#F9F9F9"),
        ),
        "th": ParagraphStyle(
            "rpt_th", parent=base["Normal"],
            fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=11,
        ),
        "th_center": ParagraphStyle(
            "rpt_th_center", parent=base["Normal"],
            fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER, leading=11,
        ),
        "td": ParagraphStyle(
            "rpt_td", parent=base["Normal"],
            fontSize=8, textColor=_TEXT, fontName="Helvetica",
            alignment=TA_LEFT, leading=12,
        ),
        "td_center": ParagraphStyle(
            "rpt_td_center", parent=base["Normal"],
            fontSize=8, textColor=_TEXT, fontName="Helvetica",
            alignment=TA_CENTER, leading=12,
        ),
        "td_bold": ParagraphStyle(
            "rpt_td_bold", parent=base["Normal"],
            fontSize=8, textColor=_TEXT, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=12,
        ),
    }


def _section_heading(story: list, s: dict, title: str, colour=None) -> None:
    story.append(Paragraph(title, s["h1"]))
    story.append(_AccentLine(45 * mm, colour=colour))
    story.append(Spacer(1, 3 * mm))


# --- Intelligent table formatting ---

_COUNT_WORDS = {"mentions", "count", "total", "responses", "#", "mention"}
_PCT_WORDS = {"%", "pos%", "neg%", "neu%", "positive%", "negative%", "neutral%"}


def _detect_column_types(headers: list[str]) -> list[str]:
    types = []
    for h in headers:
        hl = h.strip().lower()
        if hl.rstrip("%") in ("positive", "negative", "neutral") or hl in _PCT_WORDS:
            types.append("sentiment")
        elif hl in _COUNT_WORDS:
            types.append("count")
        else:
            types.append("text")
    return types


def _colour_cell_text(val: str, col_type: str, header_hint: str = "") -> str:
    """Add coloured ● dots and bold to sentiment values / percentages."""
    v = val.strip().lower()
    hh = header_hint.strip().lower()

    # Sentiment-typed column: colour percentages by column header
    if col_type == "sentiment":
        # If the header tells us which sentiment this column is, use that colour
        header_colour = ""
        for kw, chex in [("pos", "#27AE60"), ("neu", "#F39C12"), ("neg", "#E74C3C")]:
            if kw in hh:
                header_colour = chex
                break
        # Percentage values (e.g. "45.2%") — colour by header
        if re.match(r"^[\d\.]+%?$", v) and header_colour:
            return f'<font color="{header_colour}"><b>{val.strip()}</b></font>'
        # Text values containing sentiment words
        for kw, chex, dot in [("positive", "#27AE60", "●"), ("neutral", "#F39C12", "●"),
                               ("negative", "#E74C3C", "●")]:
            if kw in v:
                return f'<font color="{chex}">{dot}</font> <font color="{chex}"><b>{val.strip()}</b></font>'
        if re.match(r"^[\d\.]+%?$", v):
            return f"<b>{val.strip()}</b>"
        return val.strip()

    # Count columns — bold the number
    if col_type == "count" and re.match(r"^\d+$", v):
        return f"<b>{val.strip()}</b>"

    return val.strip()


def _smart_col_widths(headers: list[str], col_types: list[str], avail: float) -> list[float]:
    n = len(headers)
    if n == 0:
        return []
    weights = []
    for ct in col_types:
        if ct == "text":
            weights.append(3.0)
        elif ct == "count":
            weights.append(1.0)
        else:
            weights.append(1.2)
    total_w = sum(weights)
    widths = [(w / total_w) * avail for w in weights]
    for i, ct in enumerate(col_types):
        if ct in ("count", "sentiment") and widths[i] < 18 * mm:
            widths[i] = 18 * mm
    return widths


def _group_rows_by_first_col(rows: list[list[str]]) -> list[list[str]]:
    """Sort rows so that rows sharing the same first-column value are grouped together.
    Within each group the original order is preserved."""
    from collections import OrderedDict
    groups: OrderedDict[str, list[list[str]]] = OrderedDict()
    for row in rows:
        key = row[0].strip() if row else ""
        groups.setdefault(key, []).append(row)
    out: list[list[str]] = []
    for group_rows in groups.values():
        out.extend(group_rows)
    return out


def _build_table(headers: list[str], rows: list[list[str]], s: dict,
                 avail: float = 170 * mm, accent_colour=None) -> Table:
    col_types = _detect_column_types(headers)
    n = len(headers)
    col_widths = _smart_col_widths(headers, col_types, avail)

    # Group rows by first column (e.g. cluster name) so same cluster stays together
    rows = _group_rows_by_first_col(rows)

    header_cells = []
    for j, h in enumerate(headers):
        # Add coloured dots to sentiment column headers
        label = h
        for kw, chex in [("pos", "#27AE60"), ("neu", "#F39C12"), ("neg", "#E74C3C")]:
            if kw in h.strip().lower():
                label = f'<font color="{chex}">●</font> {h}'
                break
        style = s["th_center"] if col_types[j] in ("count", "sentiment") else s["th"]
        header_cells.append(Paragraph(label, style))

    # Build data rows — bold the first-column value on first occurrence in a group,
    # blank it on subsequent rows in the same group for a clean merged look
    data_rows = []
    prev_first_col = None
    for row in rows:
        cells = []
        first_val = row[0].strip() if row else ""
        for j in range(n):
            raw = row[j].strip() if j < len(row) else ""
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", raw)
            clean = _colour_cell_text(clean, col_types[j], headers[j] if j < len(headers) else "")

            # First column: bold on first occurrence, blank on repeats
            if j == 0 and col_types[0] == "text":
                if first_val == prev_first_col:
                    clean = ""  # visual merge — blank the repeated cluster name
                else:
                    clean = f"<b>{clean}</b>"

            style = s["td_center"] if col_types[j] in ("count", "sentiment") else s["td"]
            cells.append(Paragraph(clean, style))
        while len(cells) < n:
            cells.append(Paragraph("", s["td"]))
        data_rows.append(cells[:n])
        prev_first_col = first_val

    header_bg = accent_colour or _BRAND_DARK
    t = Table([header_cells] + data_rows, colWidths=col_widths, repeatRows=1)

    style_cmds: list = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, header_bg),
    ]

    # Add group separator lines between cluster groups
    prev_first = None
    for i, row in enumerate(rows):
        first_val = row[0].strip() if row else ""
        if prev_first is not None and first_val != prev_first:
            # Thicker line between groups (i+1 because row 0 is header)
            style_cmds.append(("LINEABOVE", (0, i + 1), (-1, i + 1), 0.8, _BRAND_MID))
        prev_first = first_val

    for j, ct in enumerate(col_types):
        if ct in ("count", "sentiment"):
            style_cmds.append(("ALIGN", (j, 0), (j, -1), "CENTER"))

    t.setStyle(TableStyle(style_cmds))
    return t


# --- Parsing helpers ---

def _parse_agent_sections(reply: str) -> dict[str, str]:
    text = re.sub(r"\n---\n### Hard Proof Block.*$", "", reply, flags=re.DOTALL)
    text = re.sub(r"\n---\n\*Language service processed.*$", "", text, flags=re.DOTALL)

    parts = re.split(r"(?m)^(#{1,3}\s+.+)$", text.strip())

    sections: dict[str, str] = {}
    if parts:
        sections["intro"] = parts[0].strip()

    for i in range(1, len(parts), 2):
        heading_raw = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        key = re.sub(r"^#+\s*", "", heading_raw)
        key = re.sub(r"^\d+[\.\)\.\s]+", "", key).strip().lower()
        key = re.sub(r"[^\w\s]", "", key).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        sections[key] = body

    return sections


def _parse_markdown_table(table_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r"^[\|\-\s:]+$", stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not headers:
            headers = cells
        else:
            rows.append(cells)
    return headers, rows


def _sort_rows_by_mentions(headers: list[str], rows: list[list[str]]) -> list[list[str]]:
    mention_idx: int | None = None
    for i, h in enumerate(headers):
        if h.strip().lower() in ("mentions", "count", "mention", "total"):
            mention_idx = i
            break
    if mention_idx is None:
        return rows

    def _num(val: str) -> int:
        m = re.search(r"\d+", str(val))
        return int(m.group()) if m else 0

    return sorted(
        rows,
        key=lambda r: _num(r[mention_idx]) if mention_idx < len(r) else 0,
        reverse=True,
    )


# --- Section renderers ---

def _render_section_body(story: list, s: dict, body: str,
                         accent_colour=None) -> None:
    table_lines: list[str] = []
    prose_lines: list[str] = []

    def _flush_prose() -> None:
        if prose_lines:
            _render_prose(story, s, "\n".join(prose_lines))
            prose_lines.clear()

    def _flush_table() -> None:
        if not table_lines:
            return
        headers, rows = _parse_markdown_table(table_lines)
        table_lines.clear()
        if not (headers and rows):
            return
        rows = _sort_rows_by_mentions(headers, rows)
        t = _build_table(headers, rows, s, accent_colour=accent_colour)
        story.append(t)
        story.append(Spacer(1, 4 * mm))

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") or re.match(r"^[\|\-\s:]+$", stripped):
            _flush_prose()
            table_lines.append(line)
        else:
            _flush_table()
            prose_lines.append(line)

    _flush_table()
    _flush_prose()


def _render_prose(story: list, s: dict, text: str) -> None:
    for line in text.split("\n"):
        line_s = line.strip()
        if not line_s:
            story.append(Spacer(1, 1.5 * mm))
            continue
        if re.match(r"^[\|\-\s:]+$", line_s) or line_s.startswith("|"):
            continue

        # Markdown heading
        hm = re.match(r"^#{1,4}\s+(.+)$", line_s)
        if hm:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(hm.group(1), s["h2"]))
            continue

        # Bold-only line -> sub-heading
        bm = re.match(r"^\*\*(.+)\*\*:?$", line_s)
        if bm:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(bm.group(1), s["h3"]))
            continue

        # Verbatim quote lines
        if line_s.startswith("'") or line_s.startswith('"') or line_s.startswith("\u201c"):
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line_s)
            story.append(Paragraph(clean, s["verbatim"]))
            continue

        # Inline markdown cleanup
        clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line_s)
        clean = re.sub(r"\*(.+?)\*", r"<i>\1</i>", clean)
        clean = re.sub(r"`(.+?)`", r"\1", clean)

        # Existing bullet
        bul = re.match(r"^[-\*\u2022]\s+(.*)$", clean)
        if bul:
            story.append(Paragraph(f"\u2022  {bul.group(1)}", s["bullet"]))
            continue

        # Numbered list item
        num = re.match(r"^(\d+[\.\)]\s+)(.*)$", clean)
        if num:
            story.append(Paragraph(f"{num.group(1)}{num.group(2)}", s["bullet"]))
            continue

        # Long paragraph (4+ sentences): break into bullet points per sentence
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', clean)
        if len(sentences) >= 4:
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    story.append(Paragraph(f"\u2022  {sent}", s["bullet"]))
            continue

        story.append(Paragraph(clean, s["body"]))


def _evidence_section(
    story: list, s: dict,
    evidence_summary: dict,
    sentiment: str,
    context_text: str = "",
) -> None:
    if not context_text.strip():
        return

    colour = _RED if sentiment == "Negative" else _GREEN
    _section_heading(story, s, f"Key Drivers of {sentiment} Sentiment", colour=colour)
    _render_section_body(story, s, context_text, accent_colour=colour)
    story.append(Spacer(1, 4 * mm))


def _render_recommendations(story: list, s: dict, text: str) -> None:
    """Render recommendations with thin separators between each numbered item."""
    lines = text.split("\n")
    buffer: list[str] = []
    first_item = True

    def _flush_buffer() -> None:
        nonlocal first_item
        if not buffer:
            return
        if not first_item:
            story.append(Spacer(1, 2 * mm))
            story.append(
                HRFlowable(width=170 * mm, thickness=0.4, color=_BORDER,
                           spaceAfter=3, spaceBefore=1)
            )
        _render_prose(story, s, "\n".join(buffer))
        buffer.clear()
        first_item = False

    for line in lines:
        stripped = line.strip()
        # Detect numbered item start ("1. ", "2) ", etc.) or bold heading
        is_new_item = bool(re.match(r"^\d+[\.\)]\s+", stripped))
        is_bold_heading = bool(re.match(r"^\*\*\d+", stripped) or re.match(r"^#{1,4}\s+", stripped))
        if (is_new_item or is_bold_heading) and buffer:
            _flush_buffer()
        buffer.append(line)

    _flush_buffer()


# --- Cover page ---

def _cover(story: list, s: dict, payload: dict, filename: str) -> None:
    story.append(Spacer(1, 30 * mm))
    story.append(_CoverBar(170 * mm, 4))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Survey Sentiment Analysis", s["title"]))
    story.append(Paragraph("Executive Report", s["subtitle"]))
    story.append(Spacer(1, 4 * mm))

    ts = datetime.now().strftime("%B %d, %Y")
    meta = f"Generated: {ts}&nbsp;&nbsp;|&nbsp;&nbsp;Source: {filename}"
    story.append(Paragraph(meta, s["small"]))
    story.append(Spacer(1, 6 * mm))
    story.append(_CoverBar(170 * mm, 4))
    story.append(Spacer(1, 10 * mm))


# --- Page footer ---

def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#DFE6E9"))
    canvas.line(15 * mm, 14 * mm, 200 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#95A5A6"))
    canvas.drawRightString(200 * mm, 10 * mm, f"Page {doc.page}")
    canvas.drawString(15 * mm, 10 * mm, "Survey Sentiment Analysis \u2014 Confidential")
    canvas.restoreState()


# --- Public entry point ---

def build_pdf(
    payload: dict[str, Any],
    agent_reply: str,
    filename: str = "survey_analysis",
) -> bytes:
    buf = io.BytesIO()
    W, H = A4
    margin = 15 * mm

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=20 * mm,
    )
    frame = Frame(margin, 20 * mm, W - 2 * margin, H - margin - 20 * mm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_page_footer)])

    s = _styles()
    story: list = []

    evidence = payload.get("evidence_summary", {})

    parsed = _parse_agent_sections(agent_reply)

    def _find(keywords: list[str]) -> str:
        for kw in keywords:
            for k, v in parsed.items():
                if kw in k:
                    return v
        return ""

    overview_text  = _find(["customer sentiment overview", "executive", "intro"])
    coverage_text  = _find(["coverage summary", "coverage"])
    sentiment_text = _find(["where sentiment breaks", "sentiment overview", "sentiment breakdown"])
    breakdown_text = _find(["cluster breakdown", "main cluster"])
    neg_context    = _find(["key drivers of negative", "negative sentiment"])
    pos_context    = _find(["key drivers of positive", "positive sentiment"])
    reco_text      = _find(["insight-driven", "recommendation"])

    _cover(story, s, payload, filename)

    if overview_text:
        _section_heading(story, s, "Executive Overview")
        _render_section_body(story, s, overview_text)
        story.append(Spacer(1, 5 * mm))

    if coverage_text:
        _section_heading(story, s, "Coverage Summary")
        _render_section_body(story, s, coverage_text)
        story.append(Spacer(1, 4 * mm))

    if sentiment_text:
        _section_heading(story, s, "Where Sentiment Breaks Down")
        _render_section_body(story, s, sentiment_text)
        story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    if breakdown_text:
        _section_heading(story, s, "Main Cluster Breakdown")
        _render_section_body(story, s, breakdown_text)
        story.append(Spacer(1, 4 * mm))

    _evidence_section(story, s, evidence, "Negative", context_text=neg_context)
    _evidence_section(story, s, evidence, "Positive", context_text=pos_context)

    if reco_text:
        story.append(PageBreak())
        _section_heading(story, s, "Insight-Driven Recommendations", colour=_BRAND_ACCENT)
        _render_recommendations(story, s, reco_text)

    doc.build(story)
    return buf.getvalue()
