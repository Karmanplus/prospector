"""Rendering a declarative report into a PDF.

A :class:`ReportSection` is data: a title, some text, key/value rows, tables, figures, cards and
footnotes. This module turns a list of them into paper, and knows nothing about what any particular
section means; :mod:`prospector.reporting.sections` builds those.

Two things worth knowing. Figures arrive styled for the app's dark theme, and :func:`print_figure`
inverts them for paper as the page is built, so there is no second set of colours to keep in step.
And a figure that fails to render becomes a bordered placeholder naming the error rather than
taking the whole document down. A report missing one chart is still useful, a report that raised is
not.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fpdf import FPDF
from fpdf.fonts import FontFace

from prospector import figures as _screen

# A key-value table is a list of (label, value) rows; a row whose value is None renders as a bold
# group header (used to split one table into Mission / Vehicle / ... blocks).
KeyValueRows = Sequence[tuple[str, str | None]]
TableLike = KeyValueRows | pd.DataFrame


@dataclass
class ReportSection:
    """One titled block of a report, rendered in order: cards, caption, body, table, figures,
    notes, footnotes.

    ``cards`` is a row of headline numbers above the text, each a ``(value, label, sub)``
    triple: a big number, a small label, and an optional third line in smaller print.
    ``caption`` is a short grey line under the heading. ``body`` is the section's text, a
    sequence of paragraphs. ``table`` is either key-value rows (see :data:`KeyValueRows`, where
    a ``None`` value turns the row into a heading) or a DataFrame drawn as a grid. ``figures``
    are ``(title, figure)`` pairs, and an empty title leaves the caption line off. ``notes`` is
    a closing block in grey italics for the section's assumptions. ``footnotes`` are short
    numbered source lines, referred to as ``[n]`` in the text and printed small and grey at the
    foot of the section. ``page_break_before`` starts the section on a new page.
    """

    title: str
    caption: str = ""
    cards: Sequence[tuple[str, str, str]] = field(default_factory=tuple)
    body: Sequence[str] = field(default_factory=tuple)
    figures: Sequence[tuple[str, go.Figure]] = field(default_factory=list)
    table: TableLike | None = None
    notes: str = ""
    footnotes: Sequence[str] = field(default_factory=tuple)
    page_break_before: bool = False


# The app's figures hardcode the dark palette (prospector.figures.theme) on every axis, marker, and
# background, so swapping the layout template alone cannot lighten them. Instead the figure is
# serialized and each dark-theme color is swapped for its paper equivalent in one pass (a combined
# alternation, so a freshly substituted white never gets re-swapped into ink). Data colorscales
# (Viridis, thermal, tier colors) read fine on white and are left alone.
_PRINT_COLORS: dict[str, str] = {
    _screen.BG.lower(): "#ffffff",        # page background -> paper
    _screen.SURFACE.lower(): "#ffffff",   # panel surface -> paper
    "rgba(22,28,51,0.7)": "rgba(255,255,255,0.9)",   # legend backdrop
    _screen.TEXT.lower(): "#1f2430",      # light text -> ink (also the departure marker)
    _screen.MUTED.lower(): "#5a6275",     # muted labels -> readable grey
    _screen.GRID.lower(): "#d9dde8",      # gridlines -> light grey
    "#ffffff": "#1f2430",                 # white markers are invisible on paper -> ink
}


_PRINT_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_PRINT_COLORS, key=len, reverse=True)),
    re.IGNORECASE,
)


def print_figure(fig: go.Figure) -> go.Figure:
    """A copy of ``fig`` restyled for white paper; the original is untouched."""
    raw = pio.to_json(fig)
    return pio.from_json(_PRINT_PATTERN.sub(lambda m: _PRINT_COLORS[m.group(0).lower()], raw))


def _figure_png(fig: go.Figure, width_px: int, height_px: int) -> bytes:
    """Rasterize a print-restyled copy at 2x for crisp embedding (raises on failure)."""
    return print_figure(fig).to_image(format="png", scale=2, width=width_px, height=height_px)


# Ink palette (RGB) for the page itself, kept restrained: near-black ink, grey support text, light
# rules, one teal accent echoing the app's reachable-region color.
_INK = (31, 36, 48)


_GREY = (90, 98, 117)


_RULE = (217, 221, 232)


_ACCENT = (23, 116, 118)


_MARGIN_MM = 18.0


# fpdf2's built-in fonts only cover Latin-1; a Unicode font unlocks the symbols the report uses
# (delta-v, em dashes). DejaVu ships with every mainstream Linux distro and many CI images; when
# absent, text is transliterated to Latin-1 instead (see _latin1).
_FONT_DIRS = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu"),
    Path("/usr/share/fonts/TTF"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("C:/Windows/Fonts"),
)


_GLYPH_FALLBACKS = {
    "Δ": "d",     # Greek capital delta -> "dV"
    "-": "-", "–": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "→": "->", "∞": "inf", "×": "x", "✕": "x",
    "≤": "<=", "≥": ">=",
    # Maths symbols used in the methodology text (the escape and cruise equations).
    "√": "sqrt", "²": "2", "³": "3", "·": "*", "≈": "~", "±": "+/-",
    "α": "alpha", "β": "beta", "μ": "mu", "∝": "prop to", "°": "deg",
}


def _find_font(stem: str) -> Path | None:
    for d in _FONT_DIRS:
        p = d / f"{stem}.ttf"
        if p.is_file():
            return p
    return None


def _latin1(text: str) -> str:
    """Convert to Latin-1 for the built-in-font fallback; anything it cannot show becomes '?'."""
    for src, dst in _GLYPH_FALLBACKS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


class _ReportPDF(FPDF):
    """A4 portrait with the report title + page numbers in every footer."""

    def __init__(self, report_title: str):
        super().__init__(orientation="portrait", unit="mm", format="A4")
        self.set_margins(_MARGIN_MM, _MARGIN_MM)
        self.set_auto_page_break(True, margin=_MARGIN_MM)
        self.alias_nb_pages()
        regular = _find_font("DejaVuSans")
        if regular is not None:
            self.add_font("report", "", regular)
            self.add_font("report", "B", _find_font("DejaVuSans-Bold") or regular)
            self.add_font("report", "I", _find_font("DejaVuSans-Oblique") or regular)
            self.family_name, self._sanitize = "report", False
        else:
            self.family_name, self._sanitize = "helvetica", True
        self._report_title = self.txt(report_title)

    def txt(self, text: str) -> str:
        return _latin1(text) if self._sanitize else text

    def font(self, style: str = "", size: float = 9.0, color: tuple = _INK) -> None:
        self.set_font(self.family_name, style, size)
        self.set_text_color(*color)

    def footer(self) -> None:
        self.set_y(-13.0)
        self.font("", 8, _GREY)
        self.cell(self.epw / 2, 6, self._report_title, align="L")
        self.cell(self.epw / 2, 6, f"Page {self.page_no()} of {{nb}}", align="R")


def _title_block(pdf: _ReportPDF, title: str, subtitle: str, generated: datetime) -> None:
    pdf.font("B", 19)
    pdf.multi_cell(pdf.epw, 9, pdf.txt(title), align="L", new_x="LMARGIN", new_y="NEXT")
    if subtitle:
        pdf.font("", 11.5, _GREY)
        pdf.multi_cell(pdf.epw, 6.5, pdf.txt(subtitle), align="L",
                       new_x="LMARGIN", new_y="NEXT")
    pdf.font("", 8.5, _GREY)
    pdf.cell(pdf.epw, 6, generated.strftime("Generated %Y-%m-%d %H:%M"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y() + 1.5, pdf.l_margin + pdf.epw, pdf.get_y() + 1.5)
    pdf.ln(7)


def _section_heading(pdf: _ReportPDF, title: str) -> None:
    # Never strand a heading at the foot of a page with no room for any content.
    if pdf.get_y() + 40 > pdf.page_break_trigger:
        pdf.add_page()
    pdf.font("B", 13)
    pdf.cell(pdf.epw, 8, pdf.txt(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_ACCENT)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(3)


def _render_pairs(pdf: _ReportPDF, rows: KeyValueRows) -> None:
    """Key-value rows: grey label column, ink values, hairline row rules; a row whose
    value is None becomes a bold group header splitting the table into blocks."""
    label_w = 0.38 * pdf.epw
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.2)
    for label, value in rows:
        if value is None:
            # Keep a group header with at least its first row, so it is never stranded at a foot.
            if pdf.get_y() + 14 > pdf.page_break_trigger:
                pdf.add_page()
            pdf.ln(2)
            pdf.font("B", 9.5)
            pdf.cell(pdf.epw, 6, pdf.txt(label), border="B", new_x="LMARGIN", new_y="NEXT")
            continue
        pdf.font("", 9, _GREY)
        pdf.cell(label_w, 5.6, pdf.txt(label), border="B")
        pdf.font("", 9)
        pdf.cell(pdf.epw - label_w, 5.6, pdf.txt(value), border="B",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _render_dataframe(pdf: _ReportPDF, df: pd.DataFrame) -> None:
    """A DataFrame as a grid: bold header row, hairline rules, everything stringified."""
    pdf.font("", 8)
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.2)
    headings = FontFace(emphasis="BOLD", color=_INK, fill_color=(240, 242, 247))
    with pdf.table(borders_layout="HORIZONTAL_LINES", text_align="LEFT",
                   line_height=5.0, headings_style=headings, padding=0.8) as table:
        header = table.row()
        for col in df.columns:
            header.cell(pdf.txt(str(col)))
        for _, row in df.iterrows():
            cells = table.row()
            for value in row:
                cells.cell(pdf.txt("" if pd.isna(value) else str(value)))
    pdf.ln(2)


def _figure_placeholder(pdf: _ReportPDF, error: Exception) -> None:
    """A bordered stand-in where a figure failed to export; the report still ships."""
    box_h = 24.0
    if pdf.get_y() + box_h > pdf.page_break_trigger:
        pdf.add_page()
    y0 = pdf.get_y()
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin, y0, pdf.epw, box_h)
    pdf.set_xy(pdf.l_margin + 4, y0 + 4)
    pdf.font("B", 9, _GREY)
    pdf.cell(pdf.epw - 8, 5, "Figure could not be exported", new_x="LEFT", new_y="NEXT")
    pdf.font("", 8, _GREY)
    message = f"{type(error).__name__}: {error}"
    if len(message) > 400:
        message = message[:400] + "..."
    pdf.multi_cell(pdf.epw - 8, 4.2, pdf.txt(message), max_line_height=4.2)
    pdf.set_y(y0 + box_h + 3)


def _render_figure(pdf: _ReportPDF, fig_title: str, fig: go.Figure) -> None:
    # Export at the figure's own aspect (the app sizes heights per chart).
    width_px = int(fig.layout.width or 1000)
    height_px = int(fig.layout.height or 520)
    title_h = 8.0 if fig_title else 0.0
    # Size the placed image FIRST, reserving room for the title, and break once for the pair -- so
    # a figure title is never left stranded at the foot of a page above its image.
    img_w = pdf.epw
    img_h = img_w * height_px / width_px
    max_h = pdf.h - pdf.t_margin - pdf.b_margin - 8 - title_h   # tallest image one page holds
    if img_h > max_h:
        img_w *= max_h / img_h
        img_h = max_h
    if pdf.get_y() + title_h + img_h > pdf.page_break_trigger:
        pdf.add_page()
    if fig_title:
        pdf.font("B", 9.5)
        pdf.cell(pdf.epw, 6, pdf.txt(fig_title), new_x="LMARGIN", new_y="NEXT")
    try:
        png = _figure_png(fig, width_px, height_px)
    except Exception as exc:  # a broken figure must never sink the whole report
        _figure_placeholder(pdf, exc)
        return
    x = pdf.l_margin + (pdf.epw - img_w) / 2          # centered when scaled below full width
    pdf.image(io.BytesIO(png), x=x, y=pdf.get_y(), w=img_w, h=img_h)
    pdf.set_y(pdf.get_y() + img_h + 3)


# A soft panel fill behind summary cards (paper-native RGB, never a swapped Plotly color).
_CARD_FILL = (244, 246, 250)


def _fit_font_size(pdf: _ReportPDF, text: str, max_w: float, style: str, size: float,
                   floor: float = 6.0) -> float:
    """The largest size from ``size`` down to ``floor`` at which ``text`` fits ``max_w`` mm.

    Lets a long value, such as an engine name of several words, shrink to fit its box rather
    than running off the edge."""
    txt = pdf.txt(text)
    while size > floor:
        pdf.set_font(pdf.family_name, style, size)
        if pdf.get_string_width(txt) <= max_w:
            return size
        size -= 0.5
    return floor


def _render_cards(pdf: _ReportPDF, cards: Sequence[tuple[str, str, str]]) -> None:
    """A grid of headline stats: a soft-filled box per card with a big value, a small label,
    and an optional fine third line. Three per row (four read better two-up), wrapping and
    page-breaking as needed. This is the at-a-glance band near the top of a section."""
    if not cards:
        return
    cols = 2 if len(cards) == 4 else 3
    gap, card_h = 3.0, 19.0
    card_w = (pdf.epw - gap * (cols - 1)) / cols
    row_y = pdf.get_y()
    for i, card in enumerate(cards):
        value, label = card[0], card[1]
        sub = card[2] if len(card) > 2 else ""
        col = i % cols
        if col == 0:
            if pdf.get_y() + card_h > pdf.page_break_trigger:
                pdf.add_page()
            row_y = pdf.get_y()
        x = pdf.l_margin + col * (card_w + gap)
        inner = card_w - 6
        pdf.set_fill_color(*_CARD_FILL)
        pdf.set_draw_color(*_RULE)
        pdf.set_line_width(0.2)
        pdf.rect(x, row_y, card_w, card_h, style="DF")
        pdf.set_xy(x + 3, row_y + 2.4)
        pdf.font("B", _fit_font_size(pdf, value, inner, "B", 15.0, floor=9.0), _INK)
        pdf.cell(inner, 7, pdf.txt(value), new_x="LEFT", new_y="NEXT")
        pdf.set_x(x + 3)
        pdf.font("", _fit_font_size(pdf, label, inner, "", 8.0, floor=6.5), _GREY)
        pdf.cell(inner, 4.4, pdf.txt(label), new_x="LEFT", new_y="NEXT")
        if sub:
            pdf.set_x(x + 3)
            pdf.font("", _fit_font_size(pdf, sub, inner, "", 7.0, floor=5.5), _ACCENT)
            pdf.cell(inner, 4, pdf.txt(sub))
        if col == cols - 1 or i == len(cards) - 1:
            pdf.set_y(row_y + card_h + gap)


def _render_footnotes(pdf: _ReportPDF, footnotes: Sequence[str]) -> None:
    """Numbered source lines in fine grey print, referred to as ``[n]`` in the section's text."""
    if not footnotes:
        return
    pdf.ln(1)
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw * 0.4, pdf.get_y())
    pdf.ln(1.5)
    for i, note in enumerate(footnotes, 1):
        pdf.font("", 7.5, _GREY)
        # new_x/new_y reset the cursor to the left margin on the next line; without them multi_cell
        # leaves x at the right edge and the following footnote overflows there.
        pdf.multi_cell(pdf.epw, 3.8, pdf.txt(f"[{i}] {note}"), max_line_height=3.8,
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _estimate_table_height(table: TableLike) -> float:
    """A rough rendered height (mm) for a table, for the keep-together decision below.

    Key-value rows are exact, since a heading row is taller than a value row. DataFrame rows are
    estimated generously, because wide cells wrap onto a second line, and a DataFrame repeats
    its header if it does split, so guessing low is harmless."""
    if isinstance(table, pd.DataFrame):
        return (len(table) + 1) * 9.0 + 4.0
    h = 2.0
    for _, value in table:
        h += 8.0 if value is None else 5.6
    return h


def _keep_block_together(pdf: _ReportPDF, est_h: float) -> None:
    """Start a fresh page when a block of height ``est_h`` would otherwise be cut awkwardly.

    If it fits where it is, leave it. If not, but it would fit on a fresh page, move the whole
    thing there, so a short table never gets split. If it is taller than a page it has to split
    whatever happens, and then only break when almost nothing would fit here, so it does not
    start on a sliver."""
    avail = pdf.page_break_trigger - pdf.get_y()
    page_h = pdf.page_break_trigger - pdf.t_margin
    if est_h <= avail:
        return
    if est_h <= page_h or avail < 0.45 * page_h:
        pdf.add_page()


def _render_section(pdf: _ReportPDF, section: ReportSection) -> None:
    # A part-opening section starts on a fresh page (unless already at the top of one).
    if section.page_break_before and pdf.get_y() > pdf.t_margin + 1.0:
        pdf.add_page()
    _section_heading(pdf, section.title)
    if section.caption:
        pdf.font("", 9.5, _GREY)
        pdf.multi_cell(pdf.epw, 5.2, pdf.txt(section.caption))
        pdf.ln(1.5)
    _render_cards(pdf, section.cards)
    for paragraph in section.body:
        pdf.font("", 9.5, _INK)
        pdf.multi_cell(pdf.epw, 5.2, pdf.txt(paragraph))
        pdf.ln(2.2)
    if section.table is not None:
        _keep_block_together(pdf, _estimate_table_height(section.table))
        if isinstance(section.table, pd.DataFrame):
            _render_dataframe(pdf, section.table)
        else:
            _render_pairs(pdf, section.table)
    for fig_title, fig in section.figures:
        _render_figure(pdf, fig_title, fig)
    if section.notes:
        pdf.font("I", 8.5, _GREY)
        pdf.multi_cell(pdf.epw, 4.6, pdf.txt(section.notes))
        pdf.ln(1.5)
    _render_footnotes(pdf, section.footnotes)
    pdf.ln(3)


def build_report(path: Path | str, *, title: str, subtitle: str = "",
                 sections: Sequence[ReportSection],
                 generated: datetime | None = None) -> Path:
    """Render ``sections`` into a single PDF at ``path`` and return that path.

    The document is A4 portrait: a title block with the title, subtitle and when it was made, then
    each section in order, with page numbers in the footer. A figure that fails to export becomes a
    placeholder, so the report always builds.
    """
    path = Path(path)
    pdf = _ReportPDF(title)
    pdf.add_page()
    _title_block(pdf, title, subtitle, generated or datetime.now())
    for section in sections:
        _render_section(pdf, section)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
