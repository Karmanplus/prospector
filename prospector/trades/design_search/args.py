"""The arguments the app spawns this search with.

Internal plumbing rather than a command-line tool: the UI is the entry point, and it starts
``python -m prospector.trades.design_search`` as a detached subprocess, the same role ``worker.py``
plays for the solver jobs. tests/test_design_search.py pins these flags so the two cannot drift
apart.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from prospector.population import DEFAULT_H_MAX
from prospector.spacecraft import buildability
from prospector.trades.design_search.search import Search, run_search
from prospector.trades.design_search.session import DEFAULT_OUT_ROOT, write_session_status


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required: naming a default here ties this file to one library's contents, and the app always
    # passes the open study's mission anyway.
    p.add_argument("--mission", required=True,
                   help="mission-library key defining the flight profile")
    # Recorded, never read: the app files each sweep under the study that launched it, so the
    # Compare-vehicles list can show a project its own sweeps rather than every project's.
    p.add_argument("--study", default=None,
                   help="study-library key this sweep belongs to (recorded in session.json)")
    p.add_argument("--target", default="99942", help="population designation or name")
    p.add_argument("--h-max", type=float, default=DEFAULT_H_MAX, dest="h_max",
                   help="absolute-magnitude cutoff (H) of the population the target is "
                        "resolved against. Match the study's screening h_max so a faint "
                        "target (e.g. the minimoon 2006 RH120 at H~29.5) is findable")
    # No literal default: the key must exist in whatever library is active, and naming one here
    # ties this file to one library's contents.
    p.add_argument("--engine", default=None,
                   help="engine-library key to sweep (default: the first key in the library)")
    p.add_argument("--engines", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--propellants", nargs="+", default=["xenon"],
                   metavar="GAS",
                   help="working gases to evaluate every engine pairing under (propellant "
                        "library keys, e.g. xenon krypton iodine). An engine with no "
                        "authored numbers for a gas is estimated by scaling its xenon "
                        "thrust/Isp (flagged in the results); a thruster with real "
                        "multi-gas data gets its own engine entry instead")
    p.add_argument("--dry-min", type=float, default=150.0)
    p.add_argument("--dry-max", type=float, default=350.0)
    p.add_argument("--dry-step", type=float, default=10.0)
    p.add_argument("--prop-min", type=float, default=200.0,
                   help="propellant axis floor (kg) - swept directly, independent of dry")
    p.add_argument("--prop-max", type=float, default=400.0,
                   help="propellant axis ceiling (kg)")
    p.add_argument("--prop-step", type=float, default=100.0,
                   help="propellant axis resolution (kg)")
    p.add_argument("--max-wet", type=float, default=750.0)
    # The cruise runs at its own on-time limit, as it does in the app (default: none). One
    # --duty used to throttle both the spiral and the cruise to 90%, which the app never does.
    p.add_argument("--cruise-duty", type=float, default=1.0, dest="cruise_duty",
                   help="max fraction of each cruise segment the engine may fire for (0-1)")
    # The escape's radiation scenario and coverglass, as the app's escape runs set them; unset
    # means the scenario's own defaults.
    p.add_argument("--radiation-model", default=None, dest="radiation_model",
                   help="radiation scenario key (configs/radiation/) the escape spiral flies")
    p.add_argument("--coverglass-um", type=float, default=None, dest="coverglass_um",
                   help="solar-cell coverglass thickness (um) for the escape's belt degradation")
    p.add_argument("--coverglass-density", type=float, default=None, dest="coverglass_density",
                   help="coverglass density (g/cm^3) for the escape's belt degradation")
    p.add_argument("--duty", type=float, default=0.9,
                   help="duty cycle (spiral) = thrust limit (cruise)")
    p.add_argument("--sweep-param", action="append", default=None, dest="sweep_param",
                   metavar="KEY=V1,V2,...",
                   help="sweep a MODEL setting as an extra axis on top of the vehicle "
                        "grid, e.g. --sweep-param power_margin_pct=10,20,30 (repeatable "
                        "for more axes). Every value is evaluated against every vehicle "
                        "combo and recorded as a param_<key> column. Supported keys: "
                        + ", ".join(buildability.SWEEPABLE) + ". The power/mass/tank "
                        "knobs only bite with buildability on (the default)")
    p.add_argument("--vinf", type=float, nargs="+", default=[0.0],
                   help="departure speeds to try (km/s). The cautious default "
                        "(a bare 0.0 escape) banks nothing on aiming the exit "
                        "asymptote; add points (e.g. 0.0 0.6 1.2 1.74) to let the "
                        "search exploit departure speed; about 1.9 deg of heliocentric "
                        "tilt is bought per km/s")
    p.add_argument("--nseg", type=int, default=20,
                   help="Sims-Flanagan segments per cruise leg. 20 matches the interactive "
                        "solve's fidelity so the screen doesn't over-report (a coarser 12 can "
                        "'close' a leg the higher-fidelity solve can't); lower it to trade "
                        "accuracy for speed on a broad grid")
    p.add_argument("--maxeval", type=int, default=1500)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--retries", type=int, default=1,
                   help="reseeded retries when no sweep point converges")
    p.add_argument("--pairs", nargs="+", default=None, metavar="ENGINE:COUNTS",
                   help="engine/count pairings to sweep in one run, e.g. "
                        "--pairs thruster-a:1,2 thruster-b:3-6 (overrides "
                        "--engine/--engines; counts take commas and ranges)")
    p.add_argument("--parallel", type=int, default=0,
                   help="combos evaluated concurrently across worker processes "
                        "(0 = one per CPU core, minus one)")
    p.add_argument("--workers", type=int, default=0,
                   help="processes per sweep when running serially "
                        "(--parallel 1); 0 = auto")
    p.add_argument("--buildability", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="assess each combo's buildability and rough build cost "
                        "(prospector.spacecraft.buildability sizing model: arrays for the power "
                        "appetite, tank, standard bus, payload capacity left in the "
                        "dry budget)")
    p.add_argument("--anchor", action="store_true",
                   help="also evaluate the fixed reference vehicle (150/300 kg, "
                        "3 engines) for cross-session comparison; it is labeled "
                        "'reference' and never competes for the win")
    p.add_argument("--persist-winner", action="store_true",
                   help="re-submit the winner through the standard sweep job channel")
    p.add_argument("--out", default=str(DEFAULT_OUT_ROOT /
                                        datetime.now().strftime("%Y%m%d-%H%M%S")))
    return p.parse_args(argv)


def main(argv=None) -> int:
    # The app captures this process's stdout into the session's console.log, so the handler writes
    # there. Configured at the entry point only; the library modules just get a logger and never
    # impose a handler on an importing application.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(argv)
    with Search(args) as search:
        try:
            return run_search(search, args)
        except BaseException as exc:           # incl. KeyboardInterrupt: never leave
            write_session_status(search.out, state="error",  # a phantom "running"
                                 message=f"{type(exc).__name__}: {exc}",
                                 n_evaluated=len(search.evaluations))
            raise
