"""Palette and page chrome for the NiceGUI app.

Heavy grey-blue surfaces with a single orange accent, kept for whatever has focus. Close to but not
identical to the figure colours in ``prospector/figures``, which keep a darker background so an
exported chart drops cleanly into a proposal. Both read as the same dark theme on screen.
"""
from __future__ import annotations

from nicegui import ui

# Surfaces (back-to-front) and the accents. ACCENT (orange) is reserved for focus/primary.
BG, PANEL, PANEL2 = "#161d27", "#1e2733", "#27313f"
BORDER, ACCENT, GREEN, AMBER, RED = "#3a4658", "#ff6a3c", "#40df92", "#fbaf2a", "#ff5566"
TEXT, MUTED = "#e2e8f1", "#8b96a9"
PICKAXE = "⛏"


def page_styling() -> None:
    """Set Quasar's brand colors and inject the full-viewport, no-scroll shell CSS.

    Called once per page build.
    """
    ui.colors(primary=ACCENT, secondary=ACCENT, accent=ACCENT, dark=PANEL, dark_page=BG,
              positive=GREEN, negative=RED, warning=AMBER)
    ui.add_head_html(f"""
    <style>
      html, body {{ margin:0; overflow:hidden; background:{BG}; }}
      .nicegui-content {{ padding:0 !important; gap:0 !important; height:100vh; overflow:hidden; }}
      .hover-row:hover {{ background:{PANEL2}; }}
      /* selectable tables: no checkbox column, click any row to highlight it */
      .pf-rowsel thead th:first-child, .pf-rowsel tbody td:first-child {{ display:none !important; }}
      .pf-rowsel tbody tr {{ cursor:pointer; }}
      .pf-rowsel tbody tr.selected td {{ background:{PANEL2} !important; }}
      /* fixed-layout tables: honor per-column widths so wrap-cells wraps instead of scrolling */
      .pf-fixed table {{ table-layout:fixed; }}
      /* mission timeline: hide the slider's own rail/fill so the phase gradient shows through */
      .pf-timeline .q-slider__track {{ background:transparent !important; }}
      .pf-timeline .q-slider__selection {{ background:transparent !important; }}
    </style>""")
