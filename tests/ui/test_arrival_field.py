"""The mission form's arrival control: rendezvous stores 0, flyby or impact stores the speed field,
and a form without the control keeps the mission's value."""
from types import SimpleNamespace

from prospector.config import Mission
from ui.workspaces import project


def test_the_arrival_control_maps_onto_the_one_mission_field(monkeypatch):
    prev = Mission(arrival_vinf_kms=2.5)
    monkeypatch.setattr(project, "_w", {})
    assert project._arrival_vinf(prev) == 2.5
    monkeypatch.setattr(project, "_w", {"mis_arr": SimpleNamespace(value="rendezvous"),
                                        "mis_arr_v": SimpleNamespace(value=6.0)})
    assert project._arrival_vinf(prev) == 0.0
    monkeypatch.setattr(project, "_w", {"mis_arr": SimpleNamespace(value="flyby"),
                                        "mis_arr_v": SimpleNamespace(value=6.0)})
    assert project._arrival_vinf(prev) == 6.0
    monkeypatch.setattr(project, "_w", {"mis_arr": SimpleNamespace(value="flyby"),
                                        "mis_arr_v": SimpleNamespace(value=None)})
    assert project._arrival_vinf(prev) == 6.0
