"""The standalone feasibility PDF.

    document   ReportSection, the PDF rendering, and turning dark figures into print ones
    sections   the section builders: the numbers, and which sentence applies
    wording    every word the report says, in one place

Content is kept apart from presentation. A section is just data: text, rows, tables, figure
captions and footnotes. ``document`` is the only thing that knows how any of it lands on a page.
Neither builds a figure: the caller passes those in, already built by
:mod:`prospector.figures.products`, which is what keeps the document and the app showing the same
charts rather than two separate drawings of the same result.

Public names are re-exported here, so a caller writes ``reporting.build_report(...)`` and
``reporting.summary_section(...)`` without tracking the split.
"""
from __future__ import annotations

from prospector.reporting.document import (  # noqa: F401
    KeyValueRows,
    ReportSection,
    TableLike,
    build_report,
    print_figure,
)
from prospector.reporting.sections import (  # noqa: F401
    Figures,
    assumptions_section,
    buildability_section,
    closure_section,
    cruise_section,
    escape_section,
    parameters_section,
    spacecraft_section,
    summary_section,
)

__all__ = [
    "Figures",
    "KeyValueRows",
    "ReportSection",
    "TableLike",
    "assumptions_section",
    "build_report",
    "buildability_section",
    "closure_section",
    "cruise_section",
    "escape_section",
    "parameters_section",
    "print_figure",
    "spacecraft_section",
    "summary_section",
]
