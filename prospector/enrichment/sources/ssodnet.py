"""SsODNet: name resolution and the aggregated literature values behind it.

SsODNet is the identity service for small bodies. It knows that ``2008 EV5``, ``341843`` and every
misspelling in the literature are the same object, and it carries that object's collected published
albedos, taxonomies, spin periods and colours, each flagged with whether the compilers consider it
the best available determination.

It is queried through the ``rocks`` package, an optional dependency: the desirability axis needs
it, the reachability screen does not, and a deployment that only wants trajectories should not have
to install it.

The identifier the caller passed in is carried through untouched as the join key. SsODNet's
resolved name is kept for display only, because it differs from the input for named bodies and
joining on it would silently lose every target whose name resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from prospector.enrichment.measurements import Measurement


class SourceUnavailable(RuntimeError):
    """This source cannot be reached from here: an optional dependency is missing."""


@dataclass
class Body:
    """One resolved body: its identity, its orbit, and its published measurements."""

    input_id: str                       # what the caller asked for; the join key
    name: str | None = None             # the name SsODNet settles on, for display
    number: int | None = None
    semi_major_axis_au: float | None = None
    eccentricity: float | None = None
    inclination_deg: float | None = None
    abs_mag: list[Measurement] = field(default_factory=list)
    albedo: list[Measurement] = field(default_factory=list)
    taxonomy: list[Measurement] = field(default_factory=list)
    period_h: list[Measurement] = field(default_factory=list)
    colors: dict[str, list[Measurement]] = field(default_factory=dict)
    resolved: bool = True               # False when SsODNet knows nothing by this name


def _measurements(frame, value_column: str) -> list[Measurement]:
    """One of the datacloud tables as measurements, dropping rows with no value."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or value_column not in frame.columns:
        return []
    flags = (frame["preferred"] if "preferred" in frame.columns
             else pd.Series(False, index=frame.index))
    return [Measurement(value=value, source="ssodnet", preferred=bool(flag))
            for value, flag in zip(frame[value_column], flags) if pd.notna(value)]


def _colors(frame) -> dict[str, list[Measurement]]:
    """Colour indices keyed ``"<filter1>-<filter2>"``, e.g. ``"B-V"``.

    The filter identifiers arrive fully qualified (``Johnson.V``); only the trailing filter name is
    kept, which is what the solar-colour and wavelength tables are keyed on.
    """
    import pandas as pd

    required = ("id_filter_1", "id_filter_2", "value")
    if not isinstance(frame, pd.DataFrame) or not all(c in frame.columns for c in required):
        return {}
    flags = (frame["preferred"] if "preferred" in frame.columns
             else pd.Series(False, index=frame.index))

    out: dict[str, list[Measurement]] = {}
    for index, row in frame.iterrows():
        if not (isinstance(row["id_filter_1"], str) and isinstance(row["id_filter_2"], str)):
            continue
        if pd.isna(row["value"]):
            continue
        key = f"{row['id_filter_1'].split('.')[-1]}-{row['id_filter_2'].split('.')[-1]}"
        out.setdefault(key, []).append(
            Measurement(value=row["value"], source="ssodnet", preferred=bool(flags[index])))
    return out


def lookup(identifier: str) -> Body:
    """Resolve one body and collect its published measurements.

    A body SsODNet cannot resolve comes back with ``resolved=False`` rather than raising: an
    unrecognised designation is one target's problem, and the rest of the batch continues.
    :class:`SourceUnavailable` is different: it means nothing can be looked up at all.
    """
    try:
        import rocks
    except ImportError as exc:
        raise SourceUnavailable(
            "the 'rocks' package is required for enrichment (install the enrichment extra)"
        ) from exc

    try:
        rock = rocks.Rock(identifier, datacloud=["albedos", "taxonomies", "colors", "spins"])
    except Exception:
        return Body(input_id=identifier, resolved=False)

    # A failed identification does not raise and does not come back empty: the input string is
    # echoed straight into `name`, so a body that does not exist looks like one that does with
    # nothing measured. The resolved SsODNet id is the reliable signal, being blank only when
    # nothing matched, and without this check a typo'd designation would be characterized as a
    # real, entirely unknown target and cached as one.
    if not getattr(rock, "id_", None):
        return Body(input_id=identifier, resolved=False)

    elements = getattr(rock, "orbital_elements", None)
    body = Body(
        input_id=identifier,
        name=rock.name,
        number=rock.number if getattr(rock, "number", None) is not None else None,
        semi_major_axis_au=getattr(getattr(elements, "semi_major_axis", None), "value", None),
        eccentricity=getattr(getattr(elements, "eccentricity", None), "value", None),
        inclination_deg=getattr(getattr(elements, "inclination", None), "value", None),
        albedo=_measurements(getattr(rock, "albedos", None), "albedo"),
        taxonomy=_measurements(getattr(rock, "taxonomies", None), "class_"),
        period_h=_measurements(getattr(rock, "spins", None), "period"),
        colors=_colors(getattr(rock, "colors", None)),
    )
    # H comes from the resolved body itself rather than a datacloud table, and SsODNet gives one,
    # so it is the preferred value with nothing to compare against.
    abs_mag = getattr(getattr(rock, "H", None), "value", None)
    if abs_mag is not None:
        body.abs_mag = [Measurement(value=abs_mag, source="ssodnet", preferred=True)]
    return body
