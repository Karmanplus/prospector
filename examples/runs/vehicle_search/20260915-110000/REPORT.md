# Vehicle design-space search - 2026-09-15 11:11

Target **4 Vesta (A807 FA)** on mission `dawn-to-vesta` (launch 2007-09-26 … 2007-10-15, arrive by 2011-07-16, ESCAPE drop-off). Engine `None`, duty/thrust limit 90%, wet-mass cap 1400 kg.

## Winner

**Dry 850 kg + prop 500 kg (1350 kg wet, prop/dry 0.59), 1x nstar on Xenon** - +235 kg propellant margin (+7.47 km/s at rated Isp).

Best split: departure v-inf 3.30 km/s - escape 0.00 + cruise 6.71 = 6.71 km/s of 14.19 km/s capability. Spiral 0 d; cruise departs 2007-10-14, arrives 2011-07-15 (1369 d).

Buildability: fits the dry budget - payload + margin 369 kg after arrays (110.0 kg for 10300 W BOL), tank, thrusters, and a standard bus. Rough build cost ~$79.3M (hardware + wraps, launch excluded).

## Frontier per line

Every dry mass at or below a line's frontier ALSO closes (margin only grows as dry shrinks at fixed propellant) - the highlighted row is each line's heaviest, i.e. its **minimum prop/dry** closing point.

| engine | gas | count | propellant (kg) | max closing dry (kg) | min prop/dry | margin (km/s) | limit |
|---|---|---|---|---|---|---|---|
| nstar | xenon | 1 | 300 | 850 | 0.35 | +2.17 | grid |
| nstar | xenon | 1 | 350 | 850 | 0.41 | +3.43 | grid |
| nstar | xenon | 1 | 400 | 850 | 0.47 | +4.83 | grid |
| nstar | xenon | 1 | 450 | 850 | 0.53 | +6.25 | grid |
| nstar | xenon | 1 | 500 | 850 | 0.59 | +7.47 | grid |
| nstar | xenon | 2 | 300 | 850 | 0.35 | +2.09 | grid |
| nstar | xenon | 2 | 350 | 850 | 0.41 | +3.70 | grid |
| nstar | xenon | 2 | 400 | 850 | 0.47 | +4.53 | grid |
| nstar | xenon | 2 | 450 | 850 | 0.53 | +5.02 | grid |
| nstar | xenon | 2 | 500 | 850 | 0.59 | +5.80 | grid |

## Near misses (closest first)

| dry | prop | prop/dry | engines | gas | short by (km/s) | short by (kg prop) | why |
|---|---|---|---|---|---|---|---|

## Assumptions & caveats

- Duty cycle 90% applied as the spiral duty cycle AND the cruise thrust limit.
- Engine system mass (thruster + PPU + feed, per the engine library) is not added on top of dry mass for the trajectory - budget it inside dry; the buildability check charges it explicitly.
- Success = SF matchpoint mismatch <= 1e-3 and escape+cruise propellant (rocket equation at the solved total dV) fits the usable load. The SF solve itself is tank-unconstrained, so shortfalls are measured, not guessed.
- A failed SF convergence is evidence, not proof (MBH is stochastic); a blanket failure is retried once with a new seed.
- Target elements are re-osculated at the arrival era via Horizons, so the two-body propagation is locally accurate there. An arrival on the far side of a planetary close approach still carries growing error - MONTE (or a real ephemeris) must arbitrate any finalist.
- The escape estimate and the element-based cruise floor in analytic_map.csv are both optimistic bounds: that map can prove infeasibility, never feasibility.
- Propellants evaluated: xenon. A gas marked `*` (and flagged `propellant_estimated` in combos.csv) is estimated - the engine has no authored numbers for it, so its thrust/Isp were scaled from the xenon values by the propellant library's baseline factors. Treat those as first-order; a thruster with measured multi-gas data should be added as its own engine entry.
- Buildability columns use the prospector/buildability.py sizing model: arrays sized for the thruster's power at EOL, PCDU, per-gas propellant tank, standard deep-space bus, structure/thermal/harness ratios, 10% system margin. Negative payload capacity = the bus alone busts the dry budget. Costs are comparison-grade only (array $/W and propellant $/kg from the sizing model; thruster unit and bus $/kg are documented placeholders; launch not included).
