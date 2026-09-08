"""Physical constants, each named with its unit.

One definition per quantity. Two modules holding their own copy of a number is not an error
anything catches: nothing raises, and the code carries on giving slightly different answers in
different places.

Every name carries its unit, so a value cannot be picked up in the wrong one. There is no bare
``AU`` or ``G0``: ``AU_M`` and ``AU_KM`` are separate names because the mistake this exists to
prevent is a metres-for-kilometres swap that still runs.

Only universal constants belong here. Coefficients that represent a choice rather than a
measurement, such as drag and radiation-pressure coefficients, belt radii and cell response curves,
stay with the model that owns them, or go in ``configs/`` where they can be edited.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Standard gravity
#
# Only ever used to turn a specific impulse into an exhaust speed (v_e = Isp * g0), so it is
# needed in both unit systems. Matches pykep's ``pk.G0``.
# ---------------------------------------------------------------------------

G0_M_S2 = 9.80665
G0_KM_S2 = 9.80665e-3

# ---------------------------------------------------------------------------
# Gravitational parameters
# ---------------------------------------------------------------------------

MU_SUN_M3_S2 = 1.32712440018e20
MU_SUN_KM3_S2 = 1.32712440018e11
MU_EARTH_M3_S2 = 3.986004418e14
MU_EARTH_KM3_S2 = 398600.4418
MU_MOON_M3_S2 = 4.9048695e12

# ---------------------------------------------------------------------------
# The astronomical unit
#
# AU_M is the IAU 2012 definition, which pykep 3 also adopted, so the two agree to the metre.
# Code round-tripping a pykep-derived value still uses pykep's own constant rather than assuming
# the agreement holds; pykep 2.6 used a value 9 m short of this one.
# ---------------------------------------------------------------------------

AU_M = 149597870700.0
AU_KM = 1.495978707e8

# ---------------------------------------------------------------------------
# Earth
#
# The equatorial and mean radii are different quantities, not competing values for one. The
# equatorial (WGS84) radius is the reference for altitudes and for J2; the mean radius is the
# sphere used where one figure stands in for the whole body. Both are named so neither looks
# like a typo for the other.
# ---------------------------------------------------------------------------

EARTH_EQUATORIAL_RADIUS_M = 6378137.0
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_MEAN_RADIUS_KM = 6371.0

EARTH_J2 = 1.08262668e-3            # oblateness coefficient, referenced to the equatorial radius
EARTH_ROTATION_RAD_S = 7.2921150e-5  # sidereal rotation rate, for the co-rotating drag wind

# Mean obliquity of the ecliptic at J2000: the tilt between the equatorial frame the geocentric
# escape is propagated in and the ecliptic frame the heliocentric cruise uses. Every frame rotation
# between the two goes through this one value.
EARTH_OBLIQUITY_DEG = 23.4392911
EARTH_OBLIQUITY_RAD = math.radians(EARTH_OBLIQUITY_DEG)

# Earth's mean heliocentric orbital speed: the scale that converts a departure hyperbolic excess
# into a heliocentric plane tilt, and the vis-viva scale at 1 AU.
EARTH_ORBITAL_SPEED_KMS = 29.785

# Earth's heliocentric orbit, the origin of every transfer in the element-based screen.
EARTH_ORBIT_SMA_AU = 1.0
EARTH_ORBIT_ECC = 0.0167
EARTH_ORBIT_INC_DEG = 0.0

# ---------------------------------------------------------------------------
# The Sun and the Moon, as the escape propagation models them
# ---------------------------------------------------------------------------

SOLAR_CONSTANT_W_M2 = 1361.0        # total solar irradiance at 1 AU
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
KELVIN_OFFSET_C = 273.15            # add to Celsius for kelvin
SECONDS_PER_HOUR = 3600.0
GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.674e-11
SRP_PRESSURE_N_M2 = 4.56e-6         # solar radiation pressure at 1 AU

MOON_ORBIT_TILT_RAD = 0.089         # lunar orbit inclination to the equator
MOON_ORBIT_PERIOD_DAYS = 27.32

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

SECONDS_PER_DAY = 86400.0
DAYS_PER_JULIAN_YEAR = 365.25
