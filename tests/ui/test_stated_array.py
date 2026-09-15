"""A vehicle whose file states its array keeps that array through edits on the project page.

The flown missions state their arrays (Dawn 10.3 kW); re-sizing them from the engine loads on
every keystroke cut Dawn to 2.75 kW and emptied its transfer grid.
"""
from prospector.config import EngineMount, Vehicle
from ui import state
from ui.state import S


def test_a_stated_array_is_kept_and_an_unsized_one_is_not(monkeypatch):
    mount = [EngineMount(type="nstar", count=1)]
    stated = Vehicle(name="Flown", dry_mass=800.0, fuel_mass=400.0, solar_power_W=10300.0,
                     area_m2=0.0, engines=mount)
    unsized = Vehicle(name="Concept", dry_mass=800.0, fuel_mass=400.0, solar_power_W=0.0,
                      engines=mount)
    library = {"flown": stated, "concept": unsized}
    monkeypatch.setattr(state, "load_vehicle", lambda key: library[key])
    monkeypatch.setattr(S, "vehicle_key", "flown")
    assert state.stated_array() == (10300.0, 0.0)
    monkeypatch.setattr(S, "vehicle_key", "concept")
    assert state.stated_array() is None
    monkeypatch.setattr(S, "vehicle_key", None)
    assert state.stated_array() is None
