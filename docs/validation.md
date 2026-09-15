# Checking the tool against flown missions

The example library's studies are four solar-electric missions that flew, with their real engines,
masses, dates and routes from public sources: Dawn (2007, Vesta via Mars), Psyche (2023, via
Mars), Hayabusa2 (2014, Ryugu via an Earth swing-by) and DART (2021, an impactor into Didymos).
This page records one run of each, the way the app runs it, against what the mission did.

## Setup

Each study was run as the app runs it: the transfer grid over the launch window (16 × 16 cells,
12 segments per leg), then the cruise from the grid's best cell and a spread of cells around it.
The departure speed comes from the mission file and is held exactly, since every one of these
launched straight to escape on a launch vehicle. Duty cycle was set in the flight view: 95% Dawn,
85% Psyche, 90% Hayabusa2 and DART. All four use the example library's `default` build-model
profile, since none publishes a bus power budget to build a profile from; the known figures are in
the vehicle files' comments.

One engine is mounted per thruster the mission fired at a time. Chemical propellant for attitude
control is folded into dry mass. Residual xenon is assumed at a few kilograms.

Three of the four flew a gravity assist, and the mission file names the planet, so every transfer
the tool solves for them is two legs joined at that planet (`docs/physics.md`, "Gravity assist").
DART is a direct transfer whose mission allows an arrival of up to 6.5 km/s relative to the target,
the tool's impactor.

## Results

| | Flyby, flown vs tool | Arrival, flown vs tool | Propellant, flown vs tool | Direct from the same cell |
|---|---|---|---|---|
| Dawn, via Mars to Vesta | 17 Feb 2009 at 542 km vs 13 Feb 2009 at 500 km (4.8 km/s, 38° turn) | 16 Jul 2011 vs 26 Jun 2011 | 247 kg vs 224 kg | 307 kg |
| Psyche, via Mars to Psyche | 15 May 2026 at ~4,600 km vs 19 May 2026 at 13,539 km (5.3 km/s, 9° turn) | Aug 2029 vs 31 Jul 2029 | 1,085 kg loaded for cruise and 20 months of orbit operations vs 783 kg | 1,240 kg |
| Hayabusa2, via Earth to Ryugu, from launch | 3 Dec 2015 at 3,090 km vs 30 Nov 2015 at 2,531 km (4.2 km/s, 91° turn) | 27 Jun 2018 vs 26 Jun 2018 | 24 kg vs 22 kg | 33 kg |
| DART, direct into Didymos | none | 26 Sep 2022 at 6.1 km/s vs 25 Sep 2022 at 6.4 km/s | 55 m/s of hydrazine vs 5 kg of xenon (0.26 km/s) | |

Departures: Dawn 15 Oct 2007 (flown 27 Sep), Psyche 24 Oct 2023 (flown 13 Oct), Hayabusa2
30 Nov 2014 (flown 3 Dec), DART 27 Nov 2021 (flown 24 Nov).

**Dawn.** The flyby lands four days and 42 km from the flown one, and the tool spends 23 kg less
than Dawn did. A free throttle history with no operational outages is a floor under a flown
mission, so the sign is the expected one. Against the same cell flown direct the Mars flyby saves
83 kg, which is the mission's own reasoning: Mars is what left Dawn enough for Ceres.

**Psyche.** No flown cruise figure exists to compare: JPL publishes the 1,085 kg load for cruise
plus 20 months of orbit operations. The tool's 783 kg leaves 302 kg for orbit, which is plausible
for the few hundred metres per second the orbit transitions take. The flyby is four days from the
flown one and higher, a gentler turn that this vehicle's power model prefers. The 5.8 km/s launch
speed is derived from the reported 794 × −36,991 km injection hyperbola, not published directly.
Direct from the same cell needs 1,240 kg, more than the tank: Psyche needs Mars.

**Hayabusa2.** The hardest geometry in the set: a one-year resonant return to the departure body,
then the hard turn to Ryugu. The tool finds it three days and 560 km from the flown swing-by and
lands 2 kg under the flown figure. Direct from launch with the same launch energy costs 33 kg,
so under this model the swing-by is worth 11 kg on a 64 kg tank.

**DART.** The launcher flew this route; DART's ion engine was a technology demonstration and its
55 m/s of trajectory corrections were chemical. The tool, given the route's derived 2.53 km/s of
launch speed and an arrival of up to 6.5 km/s relative to Didymos, arrives a day early at
6.4 km/s and has the ion engine do 0.26 km/s, 5 kg of xenon: a near-coast, which is what an
impactor on a good launch is. The example uses the public NSTAR engine, since the demonstration
engine's name is not in the example library, so only the geometry and the arrival compare.

## Cost

On a 24-core machine the flyby grids took 294, 378 and 342 s (Dawn, Psyche, Hayabusa2) against
110 to 220 s for a direct grid of the same size, and their cruises 123 to 141 s against 40 to 60 s.
DART's direct grid took 42 s and its cruise 6 s. Cells that closed: 17 of 256, 49 of 256, 40 of
144, 9 of 80, long flight times only, the same pattern for every mission.

## Reading these numbers

The tool minimises propellant against the mission's arrive-by date, so it arrives early whenever
an earlier arrival is cheaper; a tool arrival before the flown one is expected, not a miss. It lands
below a flown figure where the flown mission carried constraints the model does not (outages,
pointing, fixed thrust directions) and above it where the model carries constraints the mission did
not. The flyby geometries are the check that matters: the search is never told where the mission
flew, and finds the same encounters to within days and a few hundred to a thousand kilometres.

The Psyche result also depends on the vehicle being right. An earlier version of the example
carried 922 kg of xenon at 2,608 kg (a 2021 design figure) and the datasheet 280 mN; with it,
direct was cheaper than any Mars flyby found. The flown 1,085 kg, 2,747 kg and the 240 mN flight
limit are what the numbers above use.

## Sources

- Dawn: NASA fact sheets as summarised on the Dawn (spacecraft) Wikipedia page; post-launch
  heliocentric orbit 1.00 × 1.62 AU (Astronautix), hence 3.3 km/s; Goebel and Katz,
  *Fundamentals of Electric Propulsion*, JPL DESCANSO series, ch. 9, Table 9-2 (NSTAR throttle
  table).
- Psyche: NASA/JPL fact sheets as summarised on the Psyche (spacecraft) Wikipedia page; Oh et al.,
  "Development of the Psyche Mission for NASA's Discovery Program", IEPC-2019-192 (throttle curve
  shape, xenon capacity, duty cycle); the reported 794 × −36,991 km injection hyperbola.
- Hayabusa2: JAXA fact sheets as summarised on the Hayabusa2 Wikipedia page; Tsuda et al.,
  "Trajectory navigation and guidance operation toward Earth swing-by of asteroid sample return
  mission Hayabusa2", ISSFD 2015 (C3 = 21 km²/s², 57 m/s before the swing-by, altitude);
  Nishiyama et al., "In-flight operation of the Hayabusa2 ion engine system on its way to
  rendezvous with asteroid 162173 Ryugu", Acta Astronautica 166 (2020) (24 kg and 1015 m/s for the
  forward cruise); JAXA project briefing, June 2018 (609 kg, 66 kg of xenon).
- DART: *Final DART Technical Report*, NASA/NTRS 20230015804 (615 kg at launch, 60 kg of xenon,
  50 kg of hydrazine and 55.2 m/s, 579 kg at impact, 6.1 km/s, 26 September 2022 23:14 UTC);
  Redwire on the 22 m² roll-out array's 6.6 kW; the launch energy is derived here from the
  ballistic Earth-to-Didymos transfer on the flown dates.
