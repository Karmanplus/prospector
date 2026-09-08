"""The physical models behind enrichment.

Each module here turns raw measurements (an absolute magnitude, an albedo, a rotation period, a set
of orbital elements) into one of the properties the desirability screen reasons about. They are
pure and network-free: the network lives in :mod:`prospector.enrichment.sources`, and the assembly
in :mod:`prospector.enrichment.backend`.

- :mod:`sizing`      -- diameter from H and albedo, with the taxonomy-median albedo fallback
- :mod:`cohesion`    -- Holsapple (2007) rotational-disruption strength and critical spin
- :mod:`spin`        -- a fitted rotation-period distribution for bodies of similar size
- :mod:`perihelion`  -- Toliou+ (2021) expected minimum perihelion distance
- :mod:`hydration`   -- hydration class from a reflectance spectrum
- :mod:`reflectance` -- colour indices to relative reflectance, and the 0.7 um band depth
"""
