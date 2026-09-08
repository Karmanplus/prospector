"""Solvers, cheapest and roughest first:

    edelbaum     -- delta-v estimate from orbital elements alone (fast; no dates, no ephemeris)
    lambert      -- date-resolved two-burn transfers; picks launch windows and supplies
                    starting guesses, but is not a low-thrust cost
    simsflanagan -- PyKEP Sims-Flanagan plus pygmo; the real low-thrust solve, and the only
                    one whose answer is a trajectory you could fly

Each costs more and says more than the last, so the cheap ones run over the whole population and
the expensive ones only over what survives. Lambert treats burns as instant and ignores how little
thrust the vehicle has, which makes it optimistic: it may rank targets and give the optimizer
somewhere to start, but it may never reject one. What can really be flown is settled by the
low-thrust solve.

``lambert`` and everything above it import PyKEP, which the pixi environment always provides.
``edelbaum`` imports almost nothing, so the screen still runs without them. Import the modules
directly rather than through this package, which deliberately re-exports nothing: pulling a name
through here would load PyKEP to reach the one solver that does not need it.
"""
