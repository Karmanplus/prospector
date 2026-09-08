# Physics and modeling reference

What each solver works out, which way its errors point, and what it is allowed to decide. Module
docstrings stay short and point here.

Units throughout: speeds and delta-v in km/s, semimajor axis in AU, inclination in degrees, masses
in kg, thrust in newtons (millinewtons in the engine library), power in watts, time of flight in
days, dates as MJD2000. Earth's orbit (a = 1, e = 0.0167, i = 0) is the origin of every
heliocentric transfer.

## Contents

- [Which way the errors must point](#which-way-the-errors-must-point)
- [The solvers, cheapest first](#the-solvers-cheapest-first)
- [Screening on orbital elements (`solvers/edelbaum`)](#screening-on-orbital-elements-solversedelbaum)
- [Starting guesses (`solvers/lambert`)](#starting-guesses-solverslambert)
- [The solved date grid (`trades/pipeline/grid.py`)](#the-solved-date-grid-tradespipelinegridpy)
- [The real low-thrust solve (`solvers/simsflanagan`)](#the-real-low-thrust-solve-solverssimsflanagan)
- [The launch phase (`launch`, `solvers/spiral`)](#the-launch-phase-launch-solversspiral)
- [Where inclination is bought](#where-inclination-is-bought)
- [Solar-array radiation degradation (`radiation`)](#solar-array-radiation-degradation-radiation)
- [Solar-array power and mass (`arrays`)](#solar-array-power-and-mass-arrays)
- [Buildability and rough cost (`buildability`)](#buildability-and-rough-cost-buildability)

## Which way the errors must point

Screening and analysis are biased differently, so it matters which one a number came from.

Screening estimates undercharge delta-v on purpose. The expensive mistake when choosing targets is
discarding one that was reachable, so `edelbaum`'s element-based cost, `launch.escape_dv_estimate`
and the impulsive Lambert cost all err low. They rank and bound, they do not cull, and something
more accurate decides. Where a quantity has an optimistic and a pessimistic form, the screen uses
the optimistic one; the pessimistic one supplies diagnostics and the far end of the range.

Nothing after the screen is biased that way. The escape spiral runs against worst-case
trapped-proton fluxes, through eclipse, with drag and solar pressure, on arrays that lose output
climbing the belts. Thrust is derated by the operator's duty cycle. The date grid and the
trajectory solve are ordinary optimizations.

Treat a screen delta-v as a floor, and a converged one as an estimate.

## The solvers, cheapest first

Each one costs more and says more than the last. The cheap ones run over the whole population; the
expensive ones only over what survives.

| Module | What it accounts for | Cost |
|---|---|---|
| `edelbaum` | Orbital elements only, no dates and no positions | The whole population, vectorized |
| `lambert` | Real dates and positions, but instant burns. Where a solve starts from | One transfer, instant |
| `simsflanagan` | The real low-thrust solve, and a trajectory that could be flown | Seconds per solve |
| `trades/pipeline/grid` | That solve sampled over departure date and flight time | ~9 s for an 8x8 grid |

Only the last two produce a number anyone is shown, and the grid is what the app draws. `lambert`
runs on every solve, turning a pair of dates into the two-body transfer a solve starts from. Its
delta-v is never displayed, ranked or budgeted against: the burns are instant, so it is optimistic
in size and wrong in shape, which is measured below.

Leaving Earth has its own pair, working the same way but around Earth rather than the Sun:
`launch.escape_dv_estimate` is the quick answer and `solvers/spiral` is the real one.

## Screening on orbital elements (`solvers/edelbaum`)

A cheap estimate of the delta-v for a low-thrust vehicle to get from Earth's orbit onto a target
heliocentric orbit `(a, e, i)`. It needs only elements (no ephemeris, no launch date), so it
evaluates an entire small-body population at once in numpy.

### Two strategies, kept as a bracket

The true minimum-fuel low-thrust transfer is a numerically integrated optimal-control problem. Two
closed-form strategies bracket it from opposite sides, and `lowthrust_dv` takes the
**minimum**:

- **`spiral_dv`**: a continuous thrust arc that gradually morphs Earth's orbit into the
  target's: size and plane changed together (Edelbaum) while eccentricity is stretched
  (first-order term), the two RSS-combined. A low-thrust vehicle can essentially fly this, so
  it is a conservative **ceiling**.
- **`intercept_dv`**: a ballistic Hohmann transfer out to the single cheapest meeting point on
  the target orbit, then one burn to match the target's velocity there. The burns are
  *impulsive*, so a low-thrust vehicle cannot fly it as written; it is an idealized, optimistic
  **floor**.

The realistic optimum sits between them. Taking the minimum errs toward keeping targets, which is
the safe direction; the intercept term exists specifically to rescue eccentric, Earth-approaching
orbits that the spiral badly over-charges.

### Why the two disagree

Plane and eccentricity changes are cheap done slowly and in one place, expensive smeared around the
orbit at speed.

The continuous spiral pays to *manufacture* eccentricity (about `v_c * |de|`) and tilts by
continuous steering, which is inefficient: out-of-plane thrust is spread across the whole orbit,
including the fast parts and the points where it barely tilts the orbit at all, so it pays roughly
the average cost rather than the minimum.

The intercept meets the target at a turning point, where the target moves purely tangentially and,
at aphelion, slowly. It performs the entire plane change in one burn at that slow point, and never
builds eccentricity: it borrows the target's. So an orbit with aphelion just above
1 AU is met at that slow aphelion for a small raise plus a small tangential match, far less
than the spiral's eccentricity bill.

### The terms

**`edelbaum_dv`**: the combined size and plane change between two circular orbits:

```
dV = sqrt(v0^2 + vf^2 - 2 v0 vf cos(pi/2 * di))
```

with `v = sqrt(mu/a)` the circular speed and `di` the inclination change. Exact for circular
orbits, and the core low-thrust transfer result. The `pi/2` factor is the signature of a continuous
plane change: spreading the tilt around the orbit costs about 1.57× more per degree than a single
optimally placed burn.

**`eccentricity_dv`**: a body's peri/apoapsis speed spread scales like `v_c * e`, so reshaping
eccentricity by `|de|` costs about `v_c(af) * |de|` to leading order. A documented approximation;
the rigorous cost couples with `a` and is solved numerically.

**`intercept_dv`**: three feasible meeting points are evaluated and the cheapest kept:
perihelion `q`, aphelion `Q`, and the 1 AU crossing (only when `q <= 1 <= Q`). Each leg to a
turning point `R` is a Hohmann half-ellipse (`a_t = (1 AU + R)/2`); the plane change rides on the
match burn (a `cos(di)` on the tangential term), so it is charged at the local (often slow) speed.
The 1 AU crossing usually loses because the target carries a large radial velocity where it crosses
mid-orbit, and is kept only for completeness.

## Starting guesses (`solvers/lambert`)

Given a target's orbit and two dates, this solves the two-body Lambert problem (Izzo) and returns
the transfer between them. Over a date grid that is a porkchop. It puts real ephemeris geometry
back in, so it answers where the bodies are and how a two-burn transfer between them would look.

Every live call is a seed. A cold grid cell, a clicked cell, a return leg and each multi-start
guess all begin from a Lambert transfer at their own dates. The launch window itself comes from the
mission, not from here, and which cell to fly comes from the solved grid.

### It may seed, never reject

The burns are impulsive: an instantaneous departure kick plus an instantaneous arrival match. A
low-thrust vehicle has no such thrust, so this cost is optimistic in magnitude *and* the window
shape differs. A continuously-steered, many-revolution transfer is far less phase-sensitive than a
short ballistic one, so the true low-thrust window is **broader and shifted**. The impulsive `dv`
reported here is therefore not comparable to the vehicle's low-thrust budget and must never discard
a target. It exists to give the low-thrust solve somewhere to start; that solve owns the real
window and the real cost.

Earth comes from PyKEP's built-in low-precision series (`jpl_lp`, roughly arcminute accuracy, valid
about 1800–2050). The target is propagated two-body from its orbital elements via
`pk.udpla.keplerian`. Both are heliocentric, which is adequate for phasing and seeding; full force
models arrive only at a full-force-model solve.

### Mission-epoch elements

The population cache carries one set of orbital elements per body, at the source epoch. Two-body
propagation from there silently breaks across planetary close approaches: Apophis's April 2029
Earth flyby moves it from a 0.92 AU / 3.3° orbit to 1.10 AU / 2.2°, so a rendezvous solved after
that date on the cached elements chases an orbit that no longer exists.

Horizons integrates the real dynamics and can report orbital elements at any epoch, so the fix is
to refresh the target's orbit for the arrival date and let the two-body propagation be
*locally* accurate there (`ephemeris.refresh_target_elements`, cached per body and epoch).
Arrivals on the far side of a close approach from `arrive_by` still carry two-body error; a
full-force-model solve settles the finalists. The element screen keeps the catalogue orbit as it
is: it ranks, and it makes no claim to be right.

### A planet's position is fitted, not propagated

A major planet takes neither path. Its position comes from PyKEP's `jpl_lp` series, a fit to the
published Keplerian-elements-with-rates table, good to about an arcminute over 1800-2050, which is
the same source Earth already uses as the departure body. Two-body propagation of a planet's *mean*
elements over a transfer years long accumulates real along-track error, and a rendezvous is a
phasing problem, so that error lands directly on the arrival date. The fitted series already
accounts for the perturbations a refresh from Horizons would capture, so `refresh_target_elements`
returns a planet row unchanged rather than spending a network round trip refining numbers nothing
downstream reads. The mean elements in `population/planets.py` exist for the date-free element
screen and for display only.

### What "arrival" means at a planet

Every solver here works out a rendezvous with the target's orbit around the Sun. For an asteroid
that is the whole job, since station-keeping alongside a body with negligible gravity costs nothing
more. For a planet it is *capture at infinity*: arrival at the sphere of influence with whatever
excess speed the transfer leaves over. Orbit insertion, capture into a science orbit, and
atmospheric entry are all additional, and none of them is inside the quoted dV.

This is stated rather than assumed because the omission is invisible in the number: a Mars transfer
that closes with margin has said nothing about whether the vehicle can stop there. A mission that
needs a captured orbit declares its capture cost the way a return destination does
(`arrival_vinf_kms` / `insertion_dv_kms` in `launch`), and until it is declared the quoted figure
is the cost of *getting* there.

## The solved date grid (`trades/pipeline/grid.py`)

The low-thrust solve sampled, not a new model. Every cell of a (departure epoch x flight time) grid
is a real Sims-Flanagan solve with both variables fixed, so there is no approximation in it to be
wrong about.

Five approximations were measured against converged solves on three targets, and none was usable.
The impulsive cost and the two-arc correction over it both move the wrong way against
real cost. `pk.mima2` gets the
sign right but rejects every flyable cell on two of three targets. Q-law cannot aim at a point in
an orbit, so it produces no arrival date. Shaped trajectories cannot reproduce the long coasts an
efficient transfer uses, inventing over 100 km/s of dV on a coasting arc. An element-anchored
surface (`solvers/impulse_margin`) gets the level right to +/-15% but carries no dates, so it says
nothing about a departure axis.

### Cost

PyKEP 3 supplies exact derivatives, where finite differencing was over 98% of a solve's wall clock.
Each cell then starts from its neighbour's converged control history. The march runs along flight
time from longest to shortest, since a long transfer has propellant margin and a short one is near
the wall, so hard cells inherit a solved neighbour. Measured 8x8 in about 9 s, all cells closed,
cheapest point 2.92 km/s against a pinned reference of 2.923-2.968.

### Three rules, each measured

A warm start may not be re-sampled across segment counts. The resampled point satisfies the coarse
discretization's constraints but not the finer one's, and the optimizer settles there and reports a
plausible dV with a failed matchpoint: 15 of 15 trials regressed, worst +0.62 km/s, and none did at
matched counts.

A vector that failed to close may not seed its successors, which keeps one bad cell costing one
cell.

One restart reports the neighbour's answer rather than the cell's: 3.24 km/s against 3.02 cold on
Apophis, recovering at four and beating cold at eight. A grid that quotes dV uses four or more.

### What it may claim

Per-cell dV to about 1% typically and 15% at worst. A cell someone commits to still gets a full
refine. A cell the search did not close means **not found**, not infeasible, since only the
optimizer's own convergence supports either claim.

### The flight-time trade

`cheapest_per_flight_time` is that curve, read as a column-wise minimum over cells already solved.
`polish_best_per_flight_time` then re-solves each column with the departure epoch free,
warm-started from its best cell, which improved 7 of 8 Apophis columns for about 13% of the grid's
wall clock. The two are complementary: enumerating departures cannot miss the best of the dates it
tried, and a free-departure sweep trailed the grid's own minima by up to 0.33 km/s on Apophis (0.14
at eight times the restarts), while within a column the search beats the sampling.

## The real low-thrust solve (`solvers/simsflanagan`)

The genuine low-thrust design solve. Given a target orbit, a spacecraft (mass, thrust, Isp), and a
launch window, this finds a flyable continuous-thrust rendezvous: departure epoch, time of flight,
throttle history, and final mass, and from those the real low-thrust delta-v. It respects the limit
on thrust acceleration that Lambert ignores, so it is what decides the real window and the real
cost. The same class of model as EMTG's inner loop.

PyKEP's `sf_pl2pl` transcribes the transfer as a Sims-Flanagan leg: `nseg` thrust segments, each a
bounded mini-impulse, with a matchpoint mismatch constraint that the two half-legs must meet. The
matchpoint is held at the leg midpoint (`cut = 0.5`), so neither half propagates further than the
other. The decision vector is

```
z = [t0, mf, Vinf_dep(3), Vinf_arr(3), throttles(3*nseg), tof]
```

The objective maximizes final mass (minimizes propellant); equality constraints drive the
matchpoint mismatch to zero; inequalities keep each throttle magnitude at or below 1 and the
v-infinities within bounds. The problem is non-convex with many local minima, so it is optimized
with pygmo's Monotonic Basin Hopping wrapping an SLSQP local solver.

Exact derivatives are what make this affordable. PyKEP differentiates its own model, and the three
constraints this repository adds on top, being the arrival deadline, the per-segment thrust
ceilings and the departure cone, carry their own partials, so the combined problem never falls back
to finite differences. The deadline and the cone are algebraic in a handful of variables. The
thrust ceiling is a function of each segment's distance from the Sun (`sunpower.SmoothCap`, the
array-powered thrust against distance, smoothed so a discrete-mode staircase becomes a ramp the
optimizer can follow), averaged over two dozen points along the segment's two chords, since the
segment's kick stands for the thrust integrated over that arc. Both the smoothing and the sampling
were tuned against the true arc integral of the unsmoothed staircase on one of our study grids:
reading the ceiling at the midpoint alone, or smoothing a step symmetrically, let the optimizer park
segments just past a mode step and credit power the array would not make there (every cell overshot,
by 0.14 of full thrust on average); eroding the staircase by three sigma before smoothing removed the
overshoot but lost a third of the grid's cells to pessimism. Dense sampling with a one-sigma erosion
of 0.01 AU lands within 0.02 of the true ceiling on every cell, the tolerance a constant-thrust
segment was always judged by, and gives the same answers as the iterated scheme wherever that scheme
had itself respected the ceiling. The node positions depend on the departure epoch, the excess
velocities, the flight time and every earlier throttle, so their Jacobian is accumulated along the
same forward-and-backward chain the
trajectory is built from, using the state-transition matrices the Lagrangian propagator returns,
with the kicks and the rocket-equation mass updates differentiated by hand. The result is checked
against finite differences in the tests. Putting the ceiling inside the problem means one solve
respects the power its own path has, and can move the path sunward to buy thrust; the earlier
scheme froze the ceilings, solved, re-read them off the answer and solved again, which cost several
optimizations per leg and did not always converge to a consistent answer. The one term still
settled after the solve is the leg's single Isp, the throttle-weighted mean of the segments' Isps
along the answer's path; it moves a percent or two and one re-solve at the new value normally
settles it. The
difference is not marginal: a central-difference gradient costs `2*nx` fitness evaluations,
measured at 155x the fitness cost at `nseg = 6` and 415x at `nseg = 20`, which is essentially the
whole solve. The analytic gradient costs about 12x the fitness regardless of `nseg`, and the
reference solve runs 39.6 s -> 3.3 s.

Accuracy moves the same way. Against a 5-point stencil the analytic gradient is right to 1e-5 or
better, while a finite-difference estimate is roughly 50% wrong on the departure-epoch column --
the variable that sets the launch window. Steering `t0` on a gradient that wrong is why a
finite-difference solve wandered into whichever answer it happened to reach.

### Spread across restarts

There are many local answers, so a fixed restart budget returns one optimum, not the optimum.
Across five random seeds at three restarts the reference transfer converges anywhere in 3.54-3.83
km/s. That is a search property rather than a physics uncertainty, and more restarts narrow it: at
twelve the same target settles on 3.469 km/s in 5.3 s. A single short run is not the answer.

### Starting points

The optimizer converges far more reliably from a good start. A cold cell is seeded by a Lambert
transfer at that cell's own dates; every other cell inherits its neighbour's converged control
history. Seeding instead from one cheapest impulsive cell would pull every run toward the short
transfers an impulsive surface favours, when the low-thrust optimum is often a longer and more
patient one. Covering the departure epochs and flight times is the grid's job, and the sampling is
what provides it.

### How convergence is judged

The boolean feasibility flag sits at a 1e-4 tolerance that BLAS-threaded gradients can jitter
across. The matchpoint mismatch itself is the robust signal; batch tooling keys on it at 1e-3.

The mismatch is reported non-dimensionally: position over 1 AU, velocity over Earth's orbital
speed, mass over the leg's start mass. The 1e-3 gate is therefore loose, around 1.5e5 km of
position, and it rules out a leg that did not close at all rather than making a statement of
accuracy. Converged solves land two to three orders of magnitude inside it (the reference at
2e-5, about 3000 km), which is the number to read.

### Rebuilding the flown path

Sims-Flanagan models thrust as one bounded impulse per segment, so propagating its nodes as coasts
leaves a velocity kink at every segment and the path zig-zags. The real low-thrust arc it
approximates is smooth. Each segment carries a constant throttle vector, so re-propagating it under
constant inertial thrust reproduces that smooth bend back: PyKEP's zero-order-hold Keplerian Taylor
integrator (`ta.zoh_kep`) integrates precisely that maneuver, two-body motion plus a held thrust
with the matching mass flow. Any forward/backward matchpoint mismatch shows as a small step at the
join, which is honest, since the solve reports that mismatch.

### Per-node diagnostics

Each node's thrust is split into its radial, along-track and out-of-plane frame, and each axis maps
to the element it changes: radial shapes eccentricity, transverse changes energy (semimajor axis),
normal tilts the plane (inclination).

## The launch phase (`launch`, `solvers/spiral`)

A mission names a *launch type*: the orbit the launch vehicle drops the spacecraft into (LEO, SSO
rideshare, GTO) or an escape the launch vehicle itself provides (TLI, direct C3 ≥ 0). Everything
downstream derives from that type plus the vehicle: a launch-vehicle escape costs the propulsion
system nothing; an in-orbit injection means the vehicle must spiral out of the well, and that
spiral's delta-v comes off the cruise budget.

### The analytic estimate

For a slow (low thrust-to-weight) spiral out of a circular orbit, the escape delta-v tends to the
orbit's circular speed, the classic continuous-thrust limit. Leaving with hyperbolic excess `v_inf`
is estimated as

```
dv = sqrt(v_c(a0)^2 + v_inf^2)
```

which is exact at `v_inf = 0`, reduces to plain velocity-matching as the well vanishes, and
undercuts the field-free `v_c + v_inf`; it credits the Oberth advantage of raising energy while
still deep in the well. That makes it *optimistic*, the safe direction: an optimistic escape charge
leaves a larger cruise budget, so the screen errs toward keeping targets. Elliptical injections use
the circular speed at the orbit's semimajor axis; plane changes during the spiral are not charged.

### The propagated spiral and its dV-vs-v∞ curve

`solvers/spiral` numerically propagates the real escape (J2, Sun and Moon third bodies, SRP, drag,
thrust switching off in eclipse, arrays degrading in the belts) and settles the estimate. One run
records the whole **dV-vs-v∞ curve** past escape, which prices the departure v-infinity the cruise
solvers treat as free. That curve *is* the escape/cruise tradeoff, readable in both directions:
spiral to a chosen v∞ and hand it to the low-thrust solve as its departure bound, or let the
low-thrust solve converge on a v∞ and ask the curve what the spiral pays to deliver it.

Because the spiral's marginal cost per km/s of v∞ is well under 1, planning a faster departure can
only grow the screen budget, never silently shrink it.

### The departure speed knob

`departure_vinf_kms` is the user's design knob. The escape charge, its duration, its propellant,
the departure window shift, and the cruise-start mass all derive from it, so one number moves the
whole escape/cruise trade together. The cruise budget subtracts the escape charge; the screen
budget then credits the v-infinity back, because the escape already paid for it and a free
departure excess offsets up to that much transfer delta-v (triangle inequality, and optimistic as a
screening bound must be).

### Fingerprinting

A converged spiral may refine the derived escape budget, but only for the exact physical setup it
was propagated with. `launch.escape_fingerprint` collects those inputs, injection orbit, masses,
thrust, Isp, power, drag area, and the physical spiral options, rounded for stable comparison.
Deliberately **excluded**: the target hyperbolic excess (the spiral is identical up to the escape
crossing regardless of how far past it the run continued, so any converged run refines the
bare-escape term and its curve prices every knob setting), the runtime and numerics knobs (a longer
time cap cannot change an escaped run's cost), and the injection *orientation* (node, in-plane
phase, exit-aim target, which is launch targeting rather than a different physical setup).

### The departure cone

A spiral escapes *in* its orbital plane, and a plane of inclination `i` contains no direction with
equatorial `|declination| > min(i, 180-i)`, no matter which node the launch targets. So the
low-thrust solve's departure v-infinity carries a hard cone constraint set by the drop-off (or
steer-target) inclination. Everything inside the cone is launch-targetable: the node (time of day)
orients the plane to *contain* the solved exit, and where along the plane the spiral exits is
launch-date timing, which the mission window owns. The optimizer never gets an exit the escape
cannot fly.

The cone is measured in the **equatorial** frame, and it has to be: a cap on ecliptic latitude
would admit exits no node can reach. When quoted as ecliptic latitude, the band is the equatorial
coverage slewed by ± the obliquity (23.439°) as the node and season line up.

## Where inclination is bought

A target orbit tilted `i_t` off the ecliptic must have that tilt bought somewhere. Three places,
cheapest first when geometry allows:

1. **In the departure asymptote.** Adding an out-of-ecliptic v-infinity to Earth's velocity
   tilts the heliocentric orbit before the cruise burns a gram: a v-infinity of magnitude `v`,
   optimally pointed, tilts the plane by up to `asin(v / V_earth)`, about 1.9° per km/s. The
   cruise solver exploits this freely within its v-infinity cap, but for a propulsion-flown
   escape the spiral must *buy* that v-infinity and the geocentric plane must be able to
   *point* it. The minimum-magnitude buy for a tilt `x` leaves the ecliptic at latitude
   `90 - x` degrees, nearly polar.
2. **In the spiral, by geocentric plane steering.** Cheap late in the spiral where speeds are
   low. It does not tilt the heliocentric plane directly; it rotates which asymptote latitudes
   the escape can reach, enabling option 1. Asymptotically **free** in delta-v, in the
   slow-spiral limit the plane is turned near escape where the spacecraft barely moves, so
   Edelbaum's combined cost tends to the bare-escape cost as final speed tends to zero. It
   costs time, not propellant.
3. **In the cruise.** A pure heliocentric plane rotation costs `2 v sin(di/2)`, about
   0.52 km/s per degree at 1 AU. That is the upper bound; the optimizer blends it with the
   energy change and at low-speed arcs pays less.

These helpers are rough, in both accuracy and frame. The flown spiral and the low-thrust solve are
what decide.

## Solar-array radiation degradation (`radiation`)

The escape spiral climbs out through Earth's radiation belts, and trapped particles permanently
degrade the arrays (displacement damage in the cell lattice). This is one selectable model, so a
study picks a radiation scenario the way it picks an engine or launch type, and the spiral, the
array sizing, and the report all run against that choice.

### Three parts

1. **Belt environment.** Damage is keyed to the dipole McIlwain L-shell of each point,
   `L = (rho^2 + z^2)^1.5 / (Re rho^2)` with `rho` the equatorial cylindrical radius. This is
   naturally *toroidal*: L runs to infinity over the poles, so a near-polar SSO orbit leaves the
   belt away from the equator and takes its dose only near the equator crossings. The damaging
   1–10 MeV protons sit on a plateau in L (sigmoid shoulders, a sharp outer cliff); the outer
   electron belt is a Gaussian in L. Both concentrate toward the magnetic equator (a `B/B0`
   factor) and vanish in the dense atmosphere (an altitude floor). Each belt carries a
   worst-case core displacement-damage-dose rate from the literature (AP8 fluxes × NIEL × cell
   datasheet), an absolute scale, not a number calibrated to a mission total.
2. **Coverglass shielding, the dominant design lever.** Protons below an energy whose range
   equals the glass areal density are stopped. Because the trapped spectrum is soft and NIEL
   rises toward low energy, thicker glass removes the protons that do the most damage.
   This collapses to a closed-form scale on the proton core. Electrons out-range thin glass, so
   their attenuation is far weaker.
3. **Cell response.** Cumulative dose to remaining maximum-power fraction by the cell's measured
   degradation curve, `P/P0 = 1 - coef*log10(1 + Dd/ref)`. Default: GaAs/Ge single-junction, NRL
   displacement-damage-dose method.

### Calibration

Band shapes and rates are set from first principles plus literature, so the mission total falls
out of the flown trajectory and different spirals give different totals. The
cell degradation curve is the one piece taken directly from vendor/tool data; its coefficients
reproduce an OMERE 5.9 run to four significant figures.

### Validation

Against a dedicated OMERE run (NRL, AP8-MIN protons + AE8-MAX electrons + ESP solar protons,
GaAs/Ge cell, a 125 µm / 1.99 g·cm⁻³ cover) for a matched vehicle (675 kg, 0.24 N, 1540 s, SSO, no
plane change):

| | OMERE | this model |
|---|---|---|
| Spiral duration | 233 days | 216 days |
| Total dose | 3.34e10 MeV/g | 3.37e10 MeV/g |
| Proton : electron | 842 : 1 (electrons 0.12%) | 854 : 1 (electrons 0.12%) |
| End-of-life loss | 43.4% | 43.5% |

The proton and electron cores are anchored to that run. Displacement damage in GaAs is
overwhelmingly proton; the electron belt is a roughly 0.1% contributor, an about 800:1 core ratio,
not the roughly 10:1 of the raw belt fluxes. Shielding scales with **areal density**, so a
different cover thickness *or* density moves the dose correctly.

Two caveats. The proton core is anchored to OMERE's *total* proton dose, which includes ESP solar
protons and isotropic/back-shield geometry, so it lumps those into the trapped-belt term for this
mission profile; a very different mission (much more or less solar activity) should be re-checked
against a dedicated tool. The cell type and the cover areal density are by far the largest levers.

## Solar-array power and mass (`arrays`)

Two closed-form models replacing blanket assumptions.

### Output against distance from the Sun

Sunlight falls off as `1/r^2`, but the panel also runs cooler far from the Sun, and cell efficiency
rises as the panel cools, so available power falls more slowly than `1/r^2`. Panel temperature
comes from a radiative balance (absorbed sunlight on the front face against re-emission from both
faces), `T = (alpha_f I / (sigma (eps_f + eps_r)))^(1/4)`, and cell efficiency is a measured linear
fit in temperature. Near Earth, reflected sunlight and Earth infrared add heat to the *rear* face,
attenuated by the Earth view factor `(R_E / r_geo)^2`, a warm-panel penalty that matters only in
the early escape spiral and vanishes within a few Earth radii.

### Rated against operating power

A vehicle's `solar_power_W` is its *operating* output: what the panel makes facing the Sun at 1 AU
in deep space, at the temperature it runs there (about 60 °C for ordinary optical constants). A
datasheet quotes the same array at a standard cell temperature, `array_rating_temp_C` (28 °C),
where the cells convert several percent better. `ArrayModel.operating_over_rated` is the ratio of
the efficiency fit at the two temperatures, and sizing charges mass, area and cost at
`operating / ratio` (`buildability.array_rated_W`) so the array that must deliver a power in flight
is bought at the larger rated figure. The flight physics start from the operating figure and never
see the ratio; set the rating temperature equal to the operating temperature to make the two
coincide. The array-to-bus transmission efficiency is wiring loss only, and must not be used to
carry this derate a second time.

### What an array weighs

Real arrays do not come at one fixed W/kg: small wings carry fixed overheads and very large ones
change construction entirely. The mass model is a piecewise-linear power-to-mass curve over vendor
array families, with a single `mass_scale` knob as the sweepable design lever. Bracket anchors need
not coincide with bracket edges, since a family's fit line can extend past the power range it is
applied over, which is also what lets adjacent brackets step where construction changes.

### Everything is relative to 1 AU

A vehicle's `solar_power_W` means its start-of-life output at 1 AU. Everything here is a
multiplicative *fraction* relative to that reference: `power_fraction` is 1.0 at 1 AU with no Earth
term, about 0.74 at 1.2 AU, about 0.50 at
1.5 AU. Radiation degradation is not modeled here, since the dose-based belt model owns
that, and the two fractions compose multiplicatively wherever power is consumed.

## Buildability and rough cost (`buildability`)

The solvers answer "does it fly?". This answers the next question a closing design faces: do the
parts a real spacecraft needs (arrays sized for the thruster's appetite, a PCDU, propellant tanks,
the thruster systems, and a standard bus) fit inside the dry mass the trajectory search assumed,
with payload capacity left over? And roughly what would it cost? A power-hungry engine needing an
array too heavy for its own dry budget fails here even though its trajectory closed.

### Sizing

- **Power chain.** Required beginning-of-life power =
  `max(thrusting mode, ops mode) x (1 + margin)`, where each mode's loads are individually
  grossed up for the conversion stages between the array and that load:

      thrusting mode:  housekeeping / avionics_eff  +  thrusters / thruster_eff
      ops mode:        (housekeeping + ops_load) / avionics_eff

  Three stages compound. *Transmission* (array to the high-voltage main bus) is paid by
  everything. The *HV→LV converter* (main bus down to the low-voltage bus) is paid by the
  avionics, so `avionics_eff = transmission × hvlv`. The *PPU* is paid by the thrusters only,
  and whether they also pay the converter depends on which bus their power-processing unit is
  wired to, an electrical-design property carried per engine as `Engine.ppu_bus_side`:

      high-side PPU:  thruster_eff = transmission × ppu               (skips the converter)
      low-side PPU:   thruster_eff = transmission × hvlv × ppu        (pays it)

  At the default coefficients that is 81.0% versus 75.3%, so a 4 kW stack wired high-side sizes
  roughly 450 W less array, a real mass saving, which is why the wiring is configuration rather
  than an assumption. An engine with no recorded wiring defaults to the low side, the deeper of
  the two chains, so an unspecified engine is never credited a saving it may not have.

  A stack mixing both wirings draws through two chains at once and cannot be averaged per unit.
  It blends power-weighted-harmonically,

      thruster_eff = total_power / Σ (power_side / eff_side)

  the only blend that conserves the stack's array-referred demand: dividing total power by the
  blended figure reproduces the sum of the per-side demands. (Same reasoning as the
  effective-Isp blend: conserve the physical total, then back out the equivalent single value.)
  It reduces to the single-side chain for a uniform stack.

  Sizing is to the worst-mode load at *beginning* of life. Radiation damage is not
  sized against: it is a **flown** result. The escape spiral flies whatever power the degrading
  array delivers, so an undersized array runs power-limited, reported per design as `eol_factor`
  rather than a generic %/yr term. A power-limited escape is a throttled flight, not a rejection;
  whether it still closes is the trade an array-margin sweep exists to find.
- **Array mass** from the piecewise vendor power-to-mass curve. **PCDU mass** = 1 kg per kW BOL.
- **Thruster system mass** = engine unit mass (thruster + PPU + feed, the library convention) ×
  count. Note this means engine system mass is *not* added on top of dry mass for the
  trajectory, so budget it inside dry; this check charges it explicitly.
- **Primary tank dry mass** from a single pressure-sized spherocylinder: propellant volume
  (`propellant_mass / gas_storage_density`) sets the tank diameter at a fixed length:diameter
  ratio, burst pressure sizes the walls by membrane stress (hoop barrel, biaxial caps), and mass
  = wall area × thickness × wall density. So a less-dense gas such as krypton needs a bigger,
  heavier tank; a higher pressure a thicker wall; and a rounder tank the lightest vessel for a
  given volume.
- **Fixed bus** (GNC, C&DH, comms, batteries, RCS) as a flat ratio of dry mass (13%), alongside
  the structure, thermal, and harness ratios.

The dry-mass identity `dry = [payload + auto] / (1 - ratios)` carries no growth margin and inverts
in closed form, so the assessment reports the leftover dry mass a given budget leaves after
everything the vehicle needs, the reserve the app labels "payload + margin". Negative leftover
means the vehicle cannot be built at that dry mass at all.

### Cost is comparison-grade only

Array $/W and propellant $/kg come from the sizing tool. The per-engine thruster unit cost
(`Engine.cost_musd`) and the bus $/kg are documented placeholders, present to make designs
**comparable**, not quotable. Launch is not included.

### Thruster wear

Engines optionally carry qualified wear limits per unit: propellant `throughput_kg` (the dominant
electric-propulsion wear limit) and `lifetime_ignitions` (which matter for the eclipse-gated escape
spiral, restarting the thruster roughly once per shadowed revolution). A mission demanding more
than a unit is qualified for is **flagged, not vetoed**; the check is a caution. Both are optional;
`None` means unspecified and that limit is simply not checked, so populating them is the
conservative direction.
