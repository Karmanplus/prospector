"""Every word the report says, in one place.

The report's wording changes far more often than its arithmetic, and keeping the two in the same
lines means rewording a sentence involves reading past the calculation that feeds it and editing a
Python string without breaking a quote. So the sentences live here, and the builders in
:mod:`prospector.reporting.sections` work out the numbers and choose which sentence applies.

Editing this file

Text is grouped by section, in the order the sections appear in the document. Each entry is a
template, where ``{name}`` marks a value the builder supplies, already formatted with its unit --
``{dry_mass}`` arrives as ``"500 kg"`` rather than as a bare number. Rewrite freely around the
placeholders, but keep every ``{name}`` already in a template: a missing one raises at render time
rather than producing a quietly shorter sentence.

Where a section says one of several things depending on what was solved or how the vehicle is set
up, the alternatives sit together under names that say when each applies, so ``escape_by_lv`` is
next to ``escape_by_spiral``. The builder picks which one; this file shows what the reader gets
either way.

Two kinds of text are not here. The handover between escape and cruise (``_seam_clause``) reads
differently in four situations and its sentences cannot be separated from the arithmetic that
chooses between them, so it stays next to that. And a table's row labels stay next to the values
they name, because a label and the expression that fills it are one thing, and splitting them means
editing two files in step to keep a table honest.

Nothing here imports or computes anything, so a mistake in this file is a wording mistake and
nothing more. ``tests/reporting/test_wording.py`` renders every template against a sample set of
values, so an unknown or misspelled placeholder fails the suite rather than reaching a PDF.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------
# Mission and vehicle parameters -- the appendix listing every input the analysis used.
#
# The table's own row labels stay beside the values they name in ``sections.py``: a label and
# the expression that fills it are one thing, and separating them means editing two files in
# step to keep a table honest. Only the caption is here.
# ---------------------------------------------------------------------------------------

PARAMETERS = {
    "caption": ("The full set of inputs behind the analysis, with the delta-v figures derived "
                "from the masses, engine assembly, and launch type."),
}


# ---------------------------------------------------------------------------------------
# Summary -- the opening page: what the mission is, in plain language, before any detail.
# ---------------------------------------------------------------------------------------

SUMMARY = {
    "opening": "This report details the high-level mission design to {leg}. {escape_phrase}.",

    # How the spacecraft gets out of Earth's gravity well. One of these completes "opening".
    "escape_by_lv": "The launch vehicle provides sufficient energy to escape Earth's gravity well",
    "escape_by_spiral": ("Earth escape is achieved via a spiral-out trajectory starting from "
                         "{launch_name} over roughly {escape_days} using solar electric "
                         "propulsion (SEP)"),

    # What the journey ends in. "cruise_solved" is used once a cruise has converged.
    "leg_solved": "reach {target_name} after about {cruise_days} of powered cruise",
    "leg_unsolved": "rendezvous with {target_name}",

    "masses": ("The vehicle is allocated a dry mass of {dry_mass}, and carries {fuel_mass} of "
               "{gas} propellant."),

    # The ΔV verdict, once a cruise has converged.
    "budget": ("End to end, the journey produces {used} of velocity change and spends {used_prop} "
               "of the {usable} of usable propellant aboard - {verdict} what the vehicle carries. "
               "The sections that follow show why the escape and cruise are flyable as modeled, "
               "and how the dry mass divides across a spacecraft that can be built."),
    "verdict_comfortable": "comfortably within",
    "verdict_tight": "tight against",

    # The same closing sentence when no cruise has been solved yet, so there is no budget to report.
    "budget_unsolved": ("The sections that follow show how the escape and cruise phases are "
                        "modeled and why the result is flyable, and how the dry mass divides "
                        "across a buildable spacecraft."),
}


# ---------------------------------------------------------------------------------------
# The spacecraft -- what it weighs, what pushes it, and how much velocity change that buys.
# ---------------------------------------------------------------------------------------

SPACECRAFT = {
    # Two mass paragraphs: one for a vehicle that carries an unusable residual in the tank, one for
    # a vehicle whose whole propellant load is usable.
    "masses_with_residual": (
        "At launch the spacecraft has a wet mass of {wet_mass} with a dry mass of {dry_mass} plus "
        "{fuel_mass} of propellant. A small fixed amount, {unusable}, stays trapped in the tank "
        "and feed lines and can never be used, leaving {usable} available to the thrusters. "
        "Expending all of it brings the spacecraft down to {burnout_mass}. By the rocket "
        "equation, that mass ratio sets the total velocity change the vehicle can produce: "
        "{capability}, meaning the total delta-v of the mission has to fit inside that figure."),
    "masses_all_usable": (
        "At launch the spacecraft has a wet mass of {wet_mass} with a dry mass of {dry_mass} plus "
        "{fuel_mass} of propellant. Expending all the propellant brings it back down to its "
        "{dry_mass} dry mass. By the rocket equation, that mass ratio sets the total velocity "
        "change the vehicle can produce: {capability}, meaning the total delta-v of the mission "
        "has to fit inside that figure."),

    "thrust": ("Thrust comes from {assembly}, together producing {thrust} and drawing {power} of "
               "electrical power. "),
    # Completes "thrust". A mixed stack has no single specific impulse, so it says how the one
    # quoted was arrived at.
    "isp_single_type": "The specific impulse is {isp}.",
    "isp_mixed_types": ("With more than one thruster type mounted, the effective "
                        "mass-flow-weighted specific impulse is {isp}."),

    "propellant": ("The thrusters run on {gas}. {gas_note}Electric thrust is limited by available "
                   "power: the force the thrusters can sustain is set by the electrical power the "
                   "solar arrays deliver, and that power falls as radiation gradually degrades "
                   "the panels during the climb through Earth's radiation belts - the subject of "
                   "the next section."),

    # One line per gas on what its storage properties do to the tank. An unlisted gas says nothing.
    "gas_note_krypton": ("Krypton is inexpensive but stores at a low density, so it needs a "
                         "physically larger, heavier tank than xenon would for the same mass."),
    "gas_note_xenon": "Xenon stores densely, which keeps the propellant tank compact for its mass.",
    "gas_note_iodine": ("Iodine stores as a solid at very high density and is sublimated to feed "
                        "the thrusters."),

    "notes": "Dry mass is a current best estimate of the complete dry spacecraft.",
    "notes_residual": " The unusable propellant is a fixed residual.",
    "caption": "The mass, propulsion, and velocity-change capability the journey is flown on.",
}


# ---------------------------------------------------------------------------------------
# Building the spacecraft -- whether the dry mass can hold the hardware the design needs.
# ---------------------------------------------------------------------------------------

BUILDABILITY = {
    "opening": (
        "A velocity-change budget that closes is necessary but not sufficient to prove the "
        "mission is viable. The dry mass must be reasonable to accommodate the hardware a "
        "solar-electric spacecraft needs, such as solar arrays, power systems, propellant tanks, "
        "the thrusters themselves, vehicle primary structure, and other essential GNC hardware "
        "and payloads. The dry-mass breakdown below shows how the {dry_mass} divides. The rest of "
        "this section walks through the three items that drive a design and its mass allocations: "
        "the arrays, the tank, and the thrusters."),

    "arrays": (
        "Solar arrays: The arrays are sized for the single most demanding operating mode, not the "
        "sum of every load. They are sized to meet that load at START of life - {bol_power}, about "
        "{array_mass} of panel. Power is distributed on two buses: an unregulated high-voltage bus "
        "taken straight from the arrays, and a regulated low-voltage bus fed from it through a "
        "DC-DC conversion system at {converter_eff} efficiency. Reaching either costs the "
        "array-to-bus step first ({transmission_eff}), so about {avionics_reach} of the array's "
        "output reaches the avionics on the low-voltage bus. The thrusters run behind their "
        "power-processing units ({ppu_eff}) instead, {ppu_wiring} so about {thruster_reach} of the "
        "array's output reaches them. These factors combined with the operating modes set the "
        "sizing case for the arrays. {margin_note}The array's own output then varies through the "
        "mission rather than holding at its start-of-life figure. Irradiance falls as the inverse "
        "square of distance from the Sun, while the panel runs cooler further out and the cells "
        "convert more efficiently cold, so the two effects partly offset; and close to Earth "
        "during the escape, reflected sunlight and Earth infrared warm the rear face, costing a "
        "few percent that fades within a few Earth radii. The panel-power chart separates these "
        "from each other. Crossing the radiation belts then permanently degrades the panels by "
        "{belt_clause}, which this model treats as a flown result rather than something the array "
        "is oversized to overcome: the escape spiral and cruise run on whatever power the "
        "degrading array delivers. If the power falls below the thruster demand the vehicle "
        "switches to lower power modes with reduced thrust and Isp, a slower mission rather than "
        "an infeasible one."),

    # Which conversion stages the thrusters pay for, decided by the bus each PPU is wired to.
    "ppu_wiring_high": "and are wired to the high-voltage bus, which skips the converter entirely,",
    "ppu_wiring_low": "and are fed from the low-voltage bus, paying the converter as well,",
    "ppu_wiring_mixed": ("and this stack mixes both wirings - some units fed from the high-voltage "
                         "bus and some from the low-voltage one, blended by the power each side "
                         "draws,"),

    # How much belt degradation costs, when the climb duration is known and when it is not.
    "belt_known": ("about {belt_loss} ({belt_days} in the belts, shown in the panel-power chart "
                   "below)"),
    "belt_unknown": "a fraction set by how long the climb dwells in the belts",

    # The array margin: a reserve, no reserve, or an array undersized on purpose.
    "margin_reserve": ("That includes a {reserve} power reserve above the worst-mode load, so the "
                       "vehicle maintains thrust through the belts. "),
    "margin_undersized": ("This array is intentionally undersized ({reserve} margin), so the "
                          "vehicle flies power-limited from the outset, exchanging thrust for a "
                          "lighter, cheaper array. "),
    "margin_none": "The array is sized to the worst-mode load exactly, with no reserve. ",

    "array_mass_curve": (
        "The array mass comes from a piecewise vendor power-to-mass curve rather than a single "
        "watts-per-kilogram figure - small wings pay fixed overheads and very large ones change "
        "construction entirely - which for this design works out to about {specific_power} of "
        "start-of-life power per kilogram at AM0, representative of thin-film triple-junction "
        "cells.{illustrative} The curve should be set to match the chosen array line. Note that the radiation "
        "degradation above is computed on a GaAs/Ge single-junction response curve, so the two "
        "should be reconciled against the selected cell before the array size is relied on."),

    "tank": ("Propellant tank: The {gas} is held in a single carbon-overwrapped pressure vessel. "
             "Propellant mass, tank pressure, and temperature set the volume and so the tank "
             "diameter, at a fixed {shape_ratio} length-to-diameter shape. Burst pressure and the "
             "factor of safety size the wall thickness. {gas_note}For this load the tank works out "
             "to about {tank_mass}, at {pressure} with a factor of safety of {safety_factor} "
             "burst."),

    # Thruster life, when the mission would exceed the qualified throughput and when it would not.
    "life_exceeded": (
        "Thruster life: Split evenly across the {n_engines} thrusters, each one processes about "
        "{per_thruster} of propellant over the mission - more than the throughput it is presently "
        "qualified for ({why}). This is a caution rather than a stopper: qualified life is set "
        "conservatively, and adding thrusters lowers the load on each. The qualified throughput "
        "should be confirmed against vendor data for the flight build."),
    "life_ok": ("Thruster life. Split evenly across the {n_engines} thrusters, each processes "
                "about {per_thruster} of propellant. Within the qualified throughput and cycle "
                "life."),

    # Whether the hardware fits the dry budget.
    #
    # On the mass growth allowance: the sizing code applies no growth factor on top of a component,
    # and buildability.py says so. That is about the arithmetic, not the numbers, since the ratios
    # and the tank and array coefficients already have growth inside them. The allowance is real
    # and the report states it. Do not "correct" these sentences from the code comments alone,
    # which is how they were dropped once already.
    "closes": ("Adding the sized hardware to the fixed-fraction subsystems, everything fits inside "
               "the {dry_mass} dry mass and leaves {capacity} for payload and margin. A 10-15% "
               "mass growth allowance is included in the mass allocations of each subsystem."),
    "does_not_close": ("Adding the sized hardware to the fixed-fraction subsystems, the bus does "
                       "not fit inside the {dry_mass} dry mass: it would need to grow to at least "
                       "{min_dry} before any payload could be carried."),

    "notes": ("The sizing coefficients here, the array power-to-mass curve, subsystem mass "
              "fractions, and tank construction, are stated design assumptions. The tank mass is "
              "allocated a 10% mass growth allowance, and the arrays are allocated 15%. These "
              "estimates are the basis of the estimate; subsystem mass allocations are to be "
              "refined as the design matures and confirmed against the selected hardware for a "
              "flight build."),
    "caption": "How the {dry_mass} dry mass divides across a buildable solar-electric bus.",
}


# ---------------------------------------------------------------------------------------
# Leaving Earth -- the escape spiral, what shapes it, and what it costs.
# ---------------------------------------------------------------------------------------

ESCAPE = {
    # When the launch vehicle delivers escape there is no spiral to describe, so the whole section
    # is this one paragraph.
    "by_launch_vehicle": ("The {launch_name} launch leaves the spacecraft already on an escape "
                          "trajectory, so the electric propulsion system spends nothing to depart "
                          "Earth and its full velocity-change capability is available for the "
                          "cruise."),
    "caption_by_launch_vehicle": "The launch vehicle delivers Earth escape directly.",

    "why_a_spiral": (
        "A solar-electric spacecraft cannot leave Earth in a single burn due to its low thrust, "
        "so it raises its orbit gradually, circling outward over hundreds to thousands of laps "
        "until it has the energy to break free. A first estimate of the velocity change is the "
        "speed of the starting orbit itself [1]{estimate_tail} For this mission that is about "
        "{estimate}; the figure used below comes from propagating the actual spiral step by step."),
    # Completes the sentence above: a departure at essentially escape speed needs no qualifier, one
    # planned past escape has to say so because it adds in quadrature.
    "estimate_tail_zero_vinf": ".",
    "estimate_tail_with_vinf": ("; leaving with speed to spare adds to that in quadrature, and the "
                                "departure here is planned for {vinf} past escape."),

    "forces": (
        "The climb is integrated in an Earth-centred frame with the effects that shape a "
        "months-long spiral: Earth's gravity and equatorial bulge, the Sun and Moon, sunlight "
        "pressure, and air drag while the orbit's low point is still in the upper atmosphere. "
        "Thrust points along the direction of motion, tilted slightly out of plane to steer the "
        "orbit's tilt toward the target. The engine pauses in Earth's shadow, where the panels "
        "make no power [2], and for any operational duty cycle; the eclipsed share of each lap "
        "shrinks as the orbit grows [3]."),

    "belts": (
        "Crossing the Van Allen belts permanently degrades the panels, lowering the power and with "
        "it the thrust. The loss comes from a physical model, not a flat rate: each point of the "
        "flown path is placed on its magnetic field line (dipole L-shell), the trapped-particle "
        "dose there, mainly the damaging 1-10 MeV inner-belt protons with the gentler outer "
        "electrons, is attenuated by the cell coverglass, and the accumulated dose is converted "
        "to remaining power by the cell's measured curve (a GaAs/Ge single-junction cell, NRL "
        "displacement-damage-dose method [4]). A near-polar climb takes its dose only where it "
        "punches through the equatorial belts, so the loss tracks the trajectory's shape; the "
        "thrust and Isp are throttled to predefined modes based on available power. "
        "Proton fluxes use AP8 at solar minimum [6], the worst case for a "
        "low orbit, and the coverglass is the dominant design lever; the absolute loss should be "
        "confirmed against a dedicated tool (OMERE, or AP9 [5]) before the array size is relied on "
        "(see Building the spacecraft)."),

    # What the propagated run found. One of these four.
    "not_propagated": ("No spiral has been propagated for this configuration yet; the estimate "
                       "above stands until one is computed."),
    "did_not_escape": ("This run did not reach escape (it {status}): the spacecraft ran out of "
                       "propellant or time before breaking free. A higher starting orbit or more "
                       "propellant would resolve it."),
    "escaped": ("Propagating the spiral, the spacecraft reaches escape at {dv_at_escape} after "
                "{time_to_escape}{laps}, using {propellant} of propellant."),
    "escaped_with_vinf": ("Propagating the spiral, the spacecraft reaches escape at {dv_at_escape} "
                          "after {time_to_escape}{laps}. Continuing to the planned {vinf} "
                          "departure speed brings the total to {dv_total} and uses {propellant} of "
                          "propellant."),
    "laps": " and about {revolutions} laps of Earth",

    "power_limited": ("Note: after the belt degradation the panels can no longer power the "
                      "thrusters at full output, so this escape is flagged power-limited for the "
                      "spacecraft as currently sized."),

    "notes": ("Modeling simplifications - circular Sun/Moon orbits, a cylindrical shadow, an "
              "exponential atmosphere, and a dipole L-shell belt model with coverglass shielding - "
              "and their recommended verification are collected in Assumptions."),
    "notes_long_belt_dwell": (" This escape spends about {belt_days} in the belts; for so long a "
                              "residence the array degradation should be cross-checked against a "
                              "standard trapped-radiation tool (e.g. OMERE, or AE9/AP9 [5])."),
    "caption": ("How the spacecraft climbs out of Earth's gravity on its own thrusters, and what "
                "it costs."),

    # The bibliography, in the order the [n] markers above refer to them.
    "footnotes": [
        "Edelbaum, T. N. (1961). Propulsion requirements for controllable satellites. "
        "ARS Journal 31(8), 1079–1089.",
        "Neta, B. & Vallado, D. (1998). On satellite umbra/penumbra entry and exit positions. "
        "Journal of the Astronautical Sciences 46(1), 91–103.",
        "Kluever, C. A. (2011). Using Edelbaum's method to compute low-thrust transfers with "
        "Earth-shadow eclipses. Journal of Guidance, Control, and Dynamics 34(1), 300–303.",
        "Messenger, S. R. et al. (2001). Modeling solar cell degradation in space: a comparison "
        "of the NRL displacement damage dose and the JPL equivalent fluence approaches. "
        "Progress in Photovoltaics 9(2), 103–121.",
        "Ginet, G. P. et al. (2013). AE9, AP9 and SPM: new models for specifying the trapped "
        "energetic particle and space plasma environment. Space Science Reviews 179, 579–615.",
        "Sawyer, D. M. & Vette, J. I. (1976). AP-8 trapped proton environment for solar maximum "
        "and solar minimum. NSSDC/WDC-A-R&S 76-06, NASA Goddard Space Flight Center.",
    ],
}


# ---------------------------------------------------------------------------------------
# The cruise -- the powered heliocentric transfer, and why a converged one is flyable.
# ---------------------------------------------------------------------------------------

CRUISE = {
    "opening": ("Once clear of Earth, the spacecraft is {depart_state}. The cruise phase is the "
                "powered transfer that reshapes that orbit to meet {target_name} on or before "
                "{arrive_by}, designed to use as little propellant as possible."),
    # How the spacecraft is moving when the cruise picks it up: at Earth's own velocity, or with
    # leftover speed the escape bought it.
    "depart_state_zero_vinf": "moving around the Sun at essentially Earth's own orbital velocity",
    "depart_state_with_vinf": ("moving around the Sun with whatever leftover speed the escape gave "
                               "it"),

    "model": (
        "The transfer is modeled by dividing the flight into {nseg} short segments and "
        "representing the engine's continuous push over each one as a small velocity change [1]. "
        "Each segment's velocity change is capped at what the engine can produce in that "
        "span of time (thrust times duration divided by mass), so the design can never ask for "
        "more thrust than the spacecraft has. The path is then propagated forward from departure "
        "and backward from arrival, and the design is feasible only when the two halves meet with "
        "no gap in position, velocity, or mass."),

    "why_flyable": (
        "A solution that meets in the middle is therefore physically flyable: it is one continuous "
        "path with no jumps, the engine is never overdriven, and propellant drains segment by "
        "segment at the true engine efficiency. In reality the engine thrusts without interruption "
        "while its pointing direction sweeps slowly around the orbit; the model approximates that "
        "smooth turning as a run of short, fixed-direction pushes, and adding segments brings it "
        "closer to the continuous original. That smooth steering, together with the full set of "
        "gravitational forces, is what a later high-fidelity refinement adds, starting from this "
        "solution rather than from scratch."),

    "global_search": (
        "Because low-thrust transfers have many local optima, the design is found by a global "
        "search [2]: a local optimizer is run, then nudged and re-run many times, keeping any "
        "improvement, so the search explores many candidate trajectories rather than settling for "
        "the first one it finds. The transfer model and optimizer are the Sims-Flanagan method [1] "
        "implemented in open-source astrodynamics tools."),

    # What the optimizer is held to. The two differ only in whether a departure speed is bounded.
    "constraints_zero_vinf": (
        "The search maximizes the mass delivered (equivalently, minimizes propellant). The "
        "departure direction is held to one the launch can achieve, so the cruise never "
        "asks for a departure the escape cannot fly. Arrival is a rendezvous: the spacecraft's "
        "speed is matched to the target's to within {vinf_arrival}, with a hard deadline keeping "
        "arrival on or before {arrive_by}."),
    "constraints_with_vinf": (
        "The search maximizes the mass delivered (equivalently, minimizes propellant). The "
        "departure speed is held within what the escape can deliver, pointed only in a direction "
        "the launch can achieve, so the cruise never asks for a departure the escape cannot fly. "
        "Arrival is a rendezvous: the spacecraft's speed is matched to the target's to within "
        "{vinf_arrival}, with a hard deadline keeping arrival on or before {arrive_by}."),

    "why_low_thrust_wins": (
        "A quick instantaneous-burn calculation is used only to pick promising launch dates and to "
        "seed the optimizer; because it ignores the thrust limit it is optimistic and is never "
        "used to rule a target out. Benchmarked against the 2008 EV5 rendezvous studied for NASA's "
        "Asteroid Redirect Robotic Mission, this optimizer reproduces a comparable low-thrust "
        "cost, at or below that instantaneous-burn estimate. Low thrust is efficient for two "
        "reasons: the electric engine's exhaust travels several times faster than a chemical "
        "rocket's, so each kilogram of propellant buys far more velocity change; and because the "
        "thrust is spread across many revolutions instead of confined to one or two brief burns, "
        "the optimizer can place each increment where it does the most good (and at the orbital "
        "crossing points for any change of plane), which can lower the total velocity change as "
        "well."),

    "not_solved": "No cruise has been solved for this target yet.",
    "not_converged": ("This solution has not yet converged: the two halves of the trajectory do "
                      "not quite meet, so it is not yet a flyable design and should be re-solved "
                      "before it is relied on."),

    # The independent propellant cross-check against the thrust history.
    "propellant_check": (
        "The propellant figure can be checked independently against the thrust history. Over the "
        "cruise the engine runs at an average of {avg_throttle} of full thrust (the area under its "
        "throttle curve); multiplying that by the stack's full thrust ({thrust}) and the flight "
        "time, then dividing by the exhaust speed, gives {integral_propellant} of propellant, "
        "{agreement} - confirming the two are consistent."),
    "agreement_exact": "matching the optimizer's {reported} to within rounding",
    "agreement_within": "within {error} of the {reported} the optimizer reports",

    "notes": ("Modeling notes: the transfer is represented as {nseg} bounded velocity changes "
              "between coasting arcs, a standard preliminary-design approximation rather than the "
              "final continuously-steered force model; arrival is a close rendezvous (within "
              "{vinf_arrival}) rather than an exact velocity match. A high-fidelity refinement "
              "under full force models is the recommended next step."),
    "caption": ("How the interplanetary transfer is designed, and why the result is one the "
                "spacecraft can fly."),

    "footnotes": [
        "Sims, J. & Flanagan, S. (1999). Preliminary design of low-thrust interplanetary "
        "missions. AAS 99-338.",
        "Wales, D. & Doye, J. (1997). Global optimization by basin-hopping. J. Phys. Chem. A "
        "101(28), 5111–5116. Englander, J. & Conway, B. (2017). JGCD 40(1), 15–27.",
    ],
}


# ---------------------------------------------------------------------------------------
# Does it close? -- the end-to-end ledger, and whether the journey fits the spacecraft.
# ---------------------------------------------------------------------------------------

CLOSURE = {
    "velocity_ledger": ("The whole journey's velocity change, as flown, is the Earth escape plus "
                        "the cruise: {escape_dv} to leave Earth and {cruise_dv} to reach the "
                        "target, {total_dv} in all."),

    # Whether it closes is a question of kilograms: the escape and the cruise run at different
    # specific impulses, so the velocity changes do not share a currency, but the propellant does.
    "propellant_ledger": ("The propellant is what has to fit: {escape_prop} for the escape and "
                          "{cruise_prop} for the cruise, {used_prop} together, out of the {usable} "
                          "usable propellant aboard{reserve_clause} The chart below shows it "
                          "accumulating across the two phases of the journey."),
    "reserve_clause": " - leaving {margin} in reserve.",
    "short_clause": " - {short} more than is aboard.",
    "no_reserve_clause": ".",

    "schedule": ("The schedule holds together end to end: liftoff within the {launch_open} to "
                 "{launch_close} window, about {escape_days} spiralling out from Earth, departure "
                 "on {departure}, roughly {cruise_days} of cruise, and arrival on {arrival} - on "
                 "or before the {deadline} deadline."),

    "methods": ("Each leg is computed with established methods and, where the data exist, checked "
                "against flown or studied missions. Escape is modeled using step-by-step "
                "propagation under realistic perturbations. Cruise is modeled with a low-thrust "
                "optimization whose two halves close to tolerance, and the departure direction it "
                "needs is one the escape can reach. {verdict}"),
    "verdict_closes": ("At this preliminary-design level the journey closes with margin to spare "
                       "and is flyable by the spacecraft as specified. Confirmation against other "
                       "interplanetary trajectory design tools such as MALTO, Copernicus, or MONTE "
                       "is the recommended next step."),
    "verdict_does_not_close": ("At this stage the accounting does not yet close with margin to "
                               "spare - the design or the mission constraints would need "
                               "adjustment before it is flyable."),

    "notes": ("These are preliminary-design estimates, kept on the optimistic side "
              "where simplifications are needed (for instance, {example}). The reserve and the "
              "established methods are the basis for advancing to an independent high-fidelity "
              "verification under full force models."),
    "example_zero_vinf": "arrival is treated as a close rather than exact velocity match",
    "example_with_vinf": ("the departure speed is credited in full and arrival is a close rather "
                          "than exact velocity match"),
    "caption": "The end-to-end accounting, and whether the journey fits the spacecraft.",
}


# ---------------------------------------------------------------------------------------
# Assumptions -- every simplification, why it is acceptable, and what would confirm it.
#
# Rows are [modeling choice, simplification, why it is acceptable / next step]. Add a row by
# adding a list of three strings; the departure-speed row is inserted only when the mission
# plans one, because it has no bearing otherwise.
# ---------------------------------------------------------------------------------------

ASSUMPTIONS = {
    "rows_before_departure_speed": [
        ["Escape velocity-change estimate", "First-order spiral formula",
         "Conservative Duty cycle combined with step-by-step propagation with disturbances."],
        ["Sun & Moon positions", "Simple circular orbits",
         "Accurate for their gravitational pull and eclipse timing; not date-specific."],
        ["Earth's shadow", "Treated as a cylinder",
         "Slightly over-counts eclipse time at high altitude."],
        ["Upper atmosphere", "Single exponential model",
         "Drag fades out on its own as the orbit lifts; adequate for the escape."],
        ["Radiation environment", "AP8-MIN inner belt, worst-case core rates",
         "AP8-MIN is the conservative low-altitude proton case for a low circular start; "
         "cross-check the absolute level against AP9 / OMERE."],
        ["Solar-array degradation", "Dipole L-shell dose, coverglass shielding, measured cell curve",
         "Physics-based (not fitted): the loss falls out of the flown path. Confirm against a "
         "dedicated tool (OMERE) before committing; coverglass thickness is the margin lever."],
        ["Cruise transfer", "{nseg}",
         "Preliminary-design approximation; a continuously-steered model is the next step."],
        ["Arrival", "Close rendezvous, not exact",
         "A small approach speed remains, not charged to the budget."],
    ],
    # The segment count is a property of a solve, so with no converged cruise there is no count to
    # quote. Naming one anyway put a number on paper that described nothing that had been run.
    "cruise_segments_known": "{nseg} bounded velocity changes between coasts",
    "cruise_segments_unsolved": "Bounded velocity changes between coasts",
    "row_departure_speed": ["Departure speed", "Credited in full to the budget",
                            "Optimistic; the cruise may use less."],
    "rows_after_departure_speed": [
        ["Solar-array sizing", "Worst-mode load through each system's conversion chain × (1 + margin)",
         "Belt loss is flown, not sized against; 0% margin delivers full power at start of life. "
         "Confirm the conversion efficiencies and array margin against the power-system design."],
        ["Propellant-tank sizing", "Single pressure vessel, membrane-stress walls from burst pressure",
         "Fixed-shape composite-overwrapped vessel at a set length:diameter; refine the geometry "
         "and safety factor against the flight tank design."],
        ["Subsystem masses", "Current best estimates, growth allowance included",
         "Allocations carry a 10-15% mass growth allowance; to be refined as the design "
         "matures and confirmed against the selected hardware."],
        ["Thruster life", "Compared to present qualified throughput",
         "Confirm qualified throughput and ignition life against vendor data."],
    ],
    "columns": ["Modeling choice", "Simplification", "Why it is acceptable / next step"],
    "caption": ("Where the analysis simplifies, why that is reasonable for a preliminary design, "
                "and what would confirm it."),
    "notes": ("This is a preliminary-design feasibility analysis. Where simplifications are "
              "necessary its estimates are kept realistic without being overly conservative, so a "
              "promising design is not set aside prematurely. The recommended step before "
              "committing to the design is an independent high-fidelity verification under "
              "full force models and finite-thrust dynamics."),
}
