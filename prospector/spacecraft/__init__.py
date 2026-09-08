"""The physical spacecraft, and what its environment does to it.

    propulsion    electric thrusters: performance, throttle curves, the rocket equation
    propellants   working gases: cost, storage density, cross-gas performance scaling
    arrays        solar-array output vs sun distance, and the power-to-mass curve
    radiation     belt dose to array degradation (a selectable, file-based model)
    buildability  does the vehicle fit its dry budget, and roughly what does it cost

These sit below the solvers and know nothing about trajectories: they say what the vehicle is and
what it can produce, which the solvers then spend. ``buildability`` puts the other four together
into a mass and cost assessment for the whole bus.

Engines, propellants, and radiation scenarios are all file-based libraries under ``configs/``;
nothing is hardcoded here. See ``docs/physics.md`` for the array, radiation, and sizing models.
"""
