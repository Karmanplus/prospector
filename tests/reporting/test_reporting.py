"""Tests for the PDF report builder (prospector.reporting).

These lock the report contract: a list of sections renders to a real PDF (figures via Kaleido,
tables via fpdf2); the parameters section is generated straight from a ResolvedConfig; summary
cards, per-section footnotes, and page breaks render; the print restyle lightens a dark-themed
figure without mutating it; and a figure that fails to export degrades to a placeholder instead of
sinking the report. They also pin the standalone voice: the text names methods, not the software,
and the misleading "propellant exhausted" gating line is gone.
"""
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit
from prospector.reporting import (
    ReportSection,
    assumptions_section,
    build_report,
    buildability_section,
    closure_section,
    cruise_section,
    escape_section,
    parameters_section,
    print_figure,
    spacecraft_section,
    summary_section,
)
from prospector.reporting import wording as W
from prospector.spacecraft import buildability
from prospector.spacecraft.buildability import mass_allocation_components
from prospector.spacecraft.propulsion import Engine

CAT = {"E": Engine(name="E", isp_s=2000, thrust_mN=100, power_W=1000, mass_kg=5)}
# A minimal launch catalog: an LV-provided escape and a 400 km SEP-spiral start.
LAUNCHES = {
    "TLI": LaunchOrbit(name="tli", perigee_alt_km=185, apogee_alt_km=400000,
                       inclination_deg=28.5, escape_provided=True),
    "LEO": LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400,
                       inclination_deg=28.5),
}


def _resolved(launch="LEO", return_trip=False, **mission_kw):
    vehicle = Vehicle(name="v", dry_mass=500, fuel_mass=500,
                      engines=[EngineMount(type="E", count=2)])
    mission = Mission(launch_orbit=launch, return_trip=return_trip, **mission_kw)
    return ResolvedConfig.build(mission, vehicle, Screening(reference="2008 EV5"), CAT,
                                launches=LAUNCHES)


def _build_dict():
    """A real bus assessment for the resolved test vehicle (so the mass identity holds)."""
    return buildability.assess(dry_kg=500.0, prop_kg=500.0, n_engines=2,
                               engine_power_W=1000.0, engine_mass_kg=5.0,
                               throughput_kg=120.0, ignitions=800.0, propellant_name="xenon")


def _dark_fig():
    """A small figure styled like the app's dark theme (prospector.plots palette)."""
    fig = go.Figure(go.Scatter(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 4.0], mode="lines+markers",
                               marker=dict(color="#ffffff")))
    fig.update_layout(width=420, height=300, paper_bgcolor="#080a0f",
                      plot_bgcolor="#12151e", font=dict(color="#e1e6f0"),
                      title="ΔV vs tilt - unit test")
    return fig


# ---------------------------------------------------------------------------
# core rendering: figures, tables, cards, footnotes, page breaks
# ---------------------------------------------------------------------------

def test_build_report_with_figure_and_table(tmp_path):
    sections = [
        ReportSection(
            title="Reachability",
            caption="The element-based screen over the test population.",
            figures=[("ΔV vs inclination", _dark_fig())],
            notes="Coarse estimates rank only; a higher-fidelity solver is the arbiter.",
        ),
        ReportSection(
            title="Screen summary - table only",
            table=[("Totals", None),
                   ("Targets screened", "12,345"),
                   ("ΔV budget", "9.81 km/s")],
        ),
    ]
    out = build_report(tmp_path / "report.pdf", title="Study - ΔV screen",
                       subtitle="unit test artifact", sections=sections,
                       generated=datetime(2026, 6, 10, 12, 0))
    data = out.read_bytes()
    assert out.is_file()
    assert data[:5] == b"%PDF-"
    assert len(data) > 10_000        # the embedded figure PNG dominates the file size


def test_cards_footnotes_and_page_break_render(tmp_path):
    sections = [
        ReportSection(title="Summary",
                      cards=[("500 kg", "Dry mass", "CBE"), ("4 × X", "Thrusters", "300 mN"),
                             ("5.0 kW", "Array", "BOL"), ("128 d", "Escape", "78 kg")],
                      body=["Overview referencing a source [1]."],
                      footnotes=["Edelbaum, T. N. (1961). ARS Journal 31(8)."]),
        ReportSection(title="Second part", page_break_before=True,
                      body=["Begins on a fresh page."]),
    ]
    out = build_report(tmp_path / "cards.pdf", title="Cards", sections=sections)
    assert out.read_bytes()[:5] == b"%PDF-"


def test_dataframe_table_renders(tmp_path):
    df = pd.DataFrame({"target": ["2008 EV5", "1996 FG3"],
                       "ΔV (km/s)": [4.2, 5.1],
                       "tier": ["S", "A"]})
    out = build_report(tmp_path / "table.pdf", title="Table report",
                       sections=[ReportSection(title="Finalists", table=df)])
    assert out.read_bytes()[:5] == b"%PDF-"


def test_print_figure_restyles_copy_only():
    fig = _dark_fig()
    paper = print_figure(fig)
    assert paper.layout.paper_bgcolor == "#ffffff"
    assert paper.layout.plot_bgcolor == "#ffffff"
    assert paper.layout.font.color == "#1f2430"
    assert paper.data[0].marker.color == "#1f2430"   # white markers become ink on paper
    # The interactive original keeps its dark theme.
    assert fig.layout.paper_bgcolor == "#080a0f"
    assert fig.data[0].marker.color == "#ffffff"


def test_figure_export_failure_degrades_to_placeholder(tmp_path, monkeypatch):
    def _boom(self, *args, **kwargs):
        raise RuntimeError("kaleido exploded - Δ glyph in the error on purpose")

    monkeypatch.setattr(go.Figure, "to_image", _boom)
    out = build_report(
        tmp_path / "broken.pdf", title="Broken figure report",
        sections=[ReportSection(title="Reachability",
                                figures=[("ΔV vs inclination", _dark_fig())])])
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 1_000          # a complete document, placeholder box and all


def test_empty_sections_still_builds(tmp_path):
    out = build_report(tmp_path / "empty.pdf", title="Empty", sections=[])
    assert out.read_bytes()[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# parameters appendix
# ---------------------------------------------------------------------------

def test_parameters_section_contents():
    rc = _resolved()
    section = parameters_section(rc)
    assert section.title == "Mission and vehicle parameters"
    labels = [label for label, _ in section.table]
    values = {label: value for label, value in section.table}
    for header in ("Mission", "Spacecraft", "Launch and Earth escape", "Delta-v"):
        assert header in labels and values[header] is None
    assert values["Total ΔV capability"] == f"{rc.total_dv_capability:.2f} km/s"
    assert values["ΔV available for cruise"] == f"{rc.cruise_dv_limit:.2f} km/s"
    assert values["Escape ΔV budget (rated-Isp equivalent)"].endswith("(estimate)")   # LEO: estimate
    assert values["Engine assembly"] == "2 × E"
    # Screening-tool fields (magnitude cutoff, margin factor, reference) are not in a standalone
    # analysis.
    assert "Magnitude cutoff (H)" not in labels


def test_parameters_section_return_trip_and_lv_escape():
    rc = _resolved(launch="TLI", return_trip=True, return_by=date(2030, 1, 1),
                   asteroid_payload_mass=1000)
    values = dict(parameters_section(rc).table)
    assert values["Escape ΔV budget (rated-Isp equivalent)"] == "0.00 km/s (launch vehicle)"
    assert values["Return trip"] == "yes"
    assert values["Collected payload"] == "1,000 kg"
    assert "Return ΔV capability (laden)" in values


def test_parameters_section_builds_pdf(tmp_path):
    out = build_report(tmp_path / "params.pdf", title="Study",
                       sections=[parameters_section(_resolved())],
                       generated=datetime(2026, 6, 10))
    assert out.read_bytes()[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# narrative sections
# ---------------------------------------------------------------------------

def _spiral(escaped=True):
    """A minimal serialized escape-spiral dict, like the worker writes."""
    return {
        "status": "escaped" if escaped else "out_of_fuel",
        "dv_at_escape_kms": 7.6 if escaped else None,
        "dv_kms": 7.9, "tof_days": 120.0, "tof_at_escape_days": 110.0,
        "vinf_kms": 1.0, "target_vinf_kms": 1.0,
        "initial_mass_kg": 1000.0, "final_mass_kg": 820.0,
        "belt_days": 35.0, "eclipse_days": 12.0, "revolutions": 850.0,
        "power_fraction_end": 0.94, "inc_deg_end": 28.5,
        "radiation_model": "AP8-MIN worst case", "coverglass_um": 212.5,
        "coverglass_density_g_cm3": 1.640,
    }


def _sf(converged=True):
    """A minimal serialized low-thrust cruise dict."""
    return {
        "nseg": 15, "vinf_arr_kms": 0.1, "dv_kms": 4.0, "tof_days": 400.0,
        "dep_mjd2000": 10000.0, "feasible": converged,
        "mismatch": 1e-6 if converged else 1e-1,
        "initial_mass_kg": 820.0, "final_mass_kg": 760.0, "propellant_kg": 60.0,
        "vinf_dep_kms": 1.0,
    }


def test_summary_section_cards_and_overview():
    rc = _resolved()
    sec = summary_section(rc, spiral=_spiral(), sf=_sf(), build=_build_dict(),
                          target_name="2008 EV5")
    assert sec.title == "Summary"
    assert sec.cards                                   # the at-a-glance band
    labels = " ".join(label for _, label, *_ in sec.cards)
    assert "Dry mass" in labels and "Mission ΔV used" in labels
    assert "2008 EV5" in " ".join(sec.body)


def test_spacecraft_section_breakdown():
    rc = _resolved()
    sec = spacecraft_section(rc)
    assert sec.title == "The spacecraft"
    values = {label: value for label, value in sec.table}
    assert "Propulsion" in values and "Mass" in values
    assert values["Effective specific impulse"] == f"{rc.effective_isp:,.0f} s"
    body = " ".join(sec.body)
    assert "specific impulse" in body                  # explained in plain terms
    # Independent voice: the software is never named.
    assert "Prospector" not in body and "MONTE" not in body


def test_escape_section_escaped_numbers_voice_and_footnotes():
    rc = _resolved()                     # LEO start -> a real SEP spiral
    sec = escape_section(rc, _spiral())
    values = {label: value for label, value in sec.table}
    assert values["Velocity change to escape"] == "7.60 km/s"
    assert values["Days in the radiation belts"] == "35 days"
    # The report names the radiation scenario flown and quotes the spiral's PHYSICAL end-of-life
    # array fraction (0.94 -> 94%), not a separately recomputed flat-rate number.
    assert values["Radiation model"] == "AP8-MIN worst case, 212 µm cover @ 1.64 g/cm³ (34.9 mg/cm²)"
    assert values["Panel power at end of escape (belt loss)"] == "94% of start of life"
    body = " ".join(sec.body)
    # The misleading "propellant exhausted" gating line is gone.
    assert "propellant is exhausted" not in body and "propellant exhausted" not in body
    # Sources are footnoted rather than named inline as author-year in the text.
    assert sec.footnotes and any("Edelbaum" in f for f in sec.footnotes)
    assert "Prospector" not in body


def test_escape_section_lv_escape_is_short():
    rc = _resolved(launch="TLI")         # the launch vehicle provides escape
    sec = escape_section(rc, None)
    assert sec.table is None
    assert "launch vehicle" in (sec.caption + " " + " ".join(sec.body)).lower()


def test_cruise_section_converged_with_verification():
    rc = _resolved()
    verification = {"thrust_N": 0.2, "veff_kms": 19.6, "avg_throttle": 0.62,
                    "integral_prop_kg": 58.0, "reported_prop_kg": 60.0}
    sec = cruise_section(rc, _sf(), target_name="2008 EV5", verification=verification)
    body = " ".join(sec.body)
    assert "ARRM" in body or "Asteroid Redirect Robotic Mission" in body
    assert "as-flown" not in body          # ARM never flew
    assert "58.0 kg" in body and "60.0 kg" in body   # the area-under-the-curve cross-check
    values = {label: value for label, value in sec.table}
    assert values["Cruise velocity change"] == "4.00 km/s"
    assert sec.footnotes and any("Flanagan" in f for f in sec.footnotes)


def test_cruise_section_flags_unconverged():
    rc = _resolved()
    sec = cruise_section(rc, _sf(converged=False))
    assert W.CRUISE["not_converged"] in " ".join(sec.body)


def test_closure_ledger_closes():
    rc = _resolved()
    sec = closure_section(rc, spiral=_spiral(), sf=_sf())
    values = {label: value for label, value in sec.table}
    assert "Total for the mission" in values
    body = " ".join(sec.body)
    assert W.CLOSURE["verdict_closes"] in body
    # The closure section now names the industry tools a reader would confirm this against (MALTO /
    # Copernicus / MONTE), a deliberate editorial change, so the "never name software" rule is
    # narrowed here to what it is really protecting: the report must not name the tool that
    # PRODUCED it, which would undercut the independent voice. Naming external verification tools
    # does the opposite. The other sections still assert the full rule.
    assert "Prospector" not in body
    assert "recommended next step" in body


def test_buildability_section_and_allocation():
    rc = _resolved()
    build = _build_dict()
    sec = buildability_section(rc, build)
    assert sec.title == "Building the spacecraft" and sec.page_break_before
    labels = [label for label, _ in sec.table]
    assert "Dry-mass allocation" in labels and "Solar array" in labels
    # The dry-mass slices close to the dry mass (the bus identity).
    comps = mass_allocation_components(build, dry_mass_kg=rc.vehicle.dry_mass,
                                      n_engines=rc.vehicle.engine_count)
    assert abs(sum(kg for _, kg in comps) - rc.vehicle.dry_mass) < 1.0


def test_buildability_lifetime_caution_reported_faithfully():
    rc = _resolved()
    # 500 kg over 2 thrusters = 250 kg each, past a 120 kg qualified throughput.
    build = buildability.assess(dry_kg=500.0, prop_kg=500.0, n_engines=2,
                                engine_power_W=1000.0, engine_mass_kg=5.0,
                                throughput_kg=120.0, propellant_name="xenon")
    assert build["lifetime_ok"] is False
    body = " ".join(buildability_section(rc, build).body)
    assert "qualified" in body and "120" in body       # the overage is stated, not softened


def test_assumptions_is_dataframe_with_caveat():
    sec = assumptions_section()
    assert isinstance(sec.table, pd.DataFrame)
    assert "preliminary-design" in sec.notes
    # Cost is omitted entirely from the standalone document.
    assert not any("Cost" in str(row[0]) for row in sec.table.values)


def test_full_dossier_builds_pdf(tmp_path):
    rc = _resolved()
    build = _build_dict()
    sections = [
        summary_section(rc, spiral=_spiral(), sf=_sf(), build=build, target_name="2008 EV5"),
        spacecraft_section(rc),
        escape_section(rc, _spiral(), figures=[("spiral", _dark_fig())]),
        cruise_section(rc, _sf(), figures=[("trajectory", _dark_fig())], target_name="2008 EV5"),
        closure_section(rc, spiral=_spiral(), sf=_sf(), figures=[("propellant", _dark_fig())]),
        buildability_section(rc, build, figures=[("mass", _dark_fig())]),
        assumptions_section(),
        parameters_section(rc),
    ]
    out = build_report(tmp_path / "dossier.pdf",
                       title="Vehicle and trajectory feasibility - 2008 EV5",
                       subtitle="full assembly", sections=sections)
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 10_000


@pytest.mark.parametrize("bad_title", ["ΔV - résumé ✕", "plain"])
def test_unicode_in_titles_is_safe(tmp_path, bad_title):
    # Whether or not a Unicode TTF is found, exotic glyphs must never raise.
    out = build_report(tmp_path / "uni.pdf", title=bad_title,
                       sections=[ReportSection(title=bad_title,
                                               table=[(bad_title, "≥ 1 - fine")])])
    assert out.read_bytes()[:5] == b"%PDF-"
