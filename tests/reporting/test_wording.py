"""The report's wording (prospector.reporting.wording).

This exists so that ``wording.py`` is safe to edit by someone changing what the report says rather
than what it does. Every check here is about how a template is put together, never about what it
says: the words are meant to change, and a test that pinned them would have to be updated with
every edit, which teaches people to update tests without reading them.

What is checked is that an edit cannot break the report quietly. A placeholder renamed in the text
but not in the builder raises ``KeyError`` at render time, deep in a PDF export, on whichever
branch happens to use that sentence, possibly weeks later. These tests move that failure to the
moment the file is saved.
"""
from __future__ import annotations

import re
import string

import pytest

from prospector.reporting import sections, wording

# Every template dict the module publishes, by the name an editor sees at the top of its block.
CATALOGUES = {name: value for name, value in vars(wording).items()
              if name.isupper() and isinstance(value, dict)}


def _templates(catalogue: dict):
    """Every ``(key, template)`` string in a catalogue, flattening the lists of rows."""
    for key, value in catalogue.items():
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    yield f"{key}[{i}]", item
                elif isinstance(item, list):          # a table row: three cells
                    for j, cell in enumerate(item):
                        yield f"{key}[{i}][{j}]", cell


def test_every_catalogue_is_reachable():
    """A catalogue nobody reads is wording that silently does nothing when edited."""
    source = (sections.__file__ and open(sections.__file__).read()) or ""
    unused = [name for name in CATALOGUES if f"W.{name}[" not in source]
    assert not unused, f"wording no builder reads: {unused}"


@pytest.mark.parametrize("catalogue", sorted(CATALOGUES))
def test_placeholders_are_well_formed(catalogue):
    """Every ``{name}`` parses and is a name, not a position or a format spec.

    ``str.format`` accepts ``{}`` and ``{0}`` too, but a numbered placeholder breaks the moment a
    sentence is reordered, so the rule is names only.
    """
    for key, template in _templates(CATALOGUES[catalogue]):
        try:
            fields = [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]
        except ValueError as exc:
            pytest.fail(f"{catalogue}[{key!r}] is not a valid template: {exc}")
        for field in fields:
            assert field, f"{catalogue}[{key!r}] uses a positional placeholder; name it instead"
            assert field.isidentifier(), (
                f"{catalogue}[{key!r}] placeholder {field!r} is not a plain name")


@pytest.mark.parametrize("catalogue", sorted(CATALOGUES))
def test_every_template_renders(catalogue):
    """Each template fills with a stand-in for every placeholder it declares.

    Catches the unbalanced brace -- a literal ``{`` that was meant as punctuation -- which
    ``str.format`` reports only when that specific sentence is rendered.
    """
    for key, template in _templates(CATALOGUES[catalogue]):
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        rendered = template.format(**dict.fromkeys(fields, "X"))
        assert "{" not in rendered and "}" not in rendered, (
            f"{catalogue}[{key!r}] still has a brace after rendering; "
            f"double a literal brace as {{{{ or }}}}")


@pytest.mark.parametrize("catalogue", sorted(CATALOGUES))
def test_no_template_is_empty_or_stray_formatted(catalogue):
    """Guards two things an editor does by accident: emptying a sentence rather than removing
    its key, and leaving an f-string prefix behind when moving text in from ``sections.py`` --
    which yields a template whose ``{name}`` was already substituted away."""
    for key, template in _templates(CATALOGUES[catalogue]):
        assert template.strip(), f"{catalogue}[{key!r}] is empty; delete the key instead"
        assert not re.match(r"^f['\"]", template.strip()), (
            f"{catalogue}[{key!r}] looks like a leftover f-string")


def test_builders_pass_every_placeholder_a_template_declares():
    """Render the whole report on the branch combinations the builders actually take.

    This is the check that matters: a placeholder added to a sentence but not supplied by the
    builder raises here rather than in a PDF. It exercises both launch kinds and both the solved
    and unsolved states, which between them reach every conditional variant.
    """
    from tests.reporting.test_reporting import _build_dict, _resolved

    spiral = {"dv_kms": 7.1, "dv_at_escape_kms": 7.0, "tof_days": 210.0, "revolutions": 900.0,
              "tof_at_escape_days": 200.0, "initial_mass_kg": 1000.0, "final_mass_kg": 880.0,
              "status": "escaped", "power_fraction_end": 0.72, "belt_days": 90.0,
              "eclipse_days": 40.0, "vinf_kms": 0.6, "power_limited": True,
              "radiation_model": "ap8min-worstcase", "coverglass_um": 212.5,
              "coverglass_density_g_cm3": 1.64, "inc_deg_end": 5.0}
    sf = {"dv_kms": 4.2, "tof_days": 540.0, "propellant_kg": 260.0, "feasible": True,
          "mismatch": 1e-6, "dep_mjd2000": 11000.0, "initial_mass_kg": 880.0,
          "final_mass_kg": 620.0, "nseg": 15, "vinf_dep_kms": 0.6, "vinf_arr_kms": 0.1}
    verification = {"avg_throttle": 0.8, "thrust_N": 0.2,
                    "integral_prop_kg": 261.0, "reported_prop_kg": 260.0}

    for launch in ("LEO", "TLI"):
        rc = _resolved(launch=launch)
        build = _build_dict()
        for sp, solution in ((spiral, sf), (None, None)):
            sections.summary_section(rc, spiral=sp, sf=solution, build=build)
            sections.spacecraft_section(rc)
            sections.buildability_section(rc, build)
            sections.escape_section(rc, sp)
            sections.cruise_section(rc, solution, verification=verification if solution else None)
            sections.closure_section(rc, spiral=sp, sf=solution)
            sections.parameters_section(rc)
            sections.assumptions_section(departure_vinf_kms=0.6)
            sections.assumptions_section(departure_vinf_kms=0.0)
