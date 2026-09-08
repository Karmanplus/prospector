"""Where enrichment's raw measurements come from.

Each module wraps one outside catalogue or survey and returns lists of
:class:`~prospector.enrichment.measurements.Measurement`, with no physics, no merging, and no
opinion about which source is right. The physics is in :mod:`prospector.enrichment.models`, and
:mod:`prospector.enrichment.backend` puts the two together.

- :mod:`ssodnet`   -- the identity resolver and the aggregated literature values behind it
- :mod:`astorb`    -- Lowell's AstorbDB: survey albedos, taxonomies and lightcurves
- :mod:`surveys`   -- targeted small-body surveys (rotation periods, reflectance spectra)
- :mod:`overrides` -- a local file of hand-entered values that outrank every catalogue

Every one of them is allowed to come back empty. A source that is unreachable, rate-limited or
simply has no record of a body must leave everything else known about that body intact, because a
target with fewer known properties is still a target and the filters keep unknowns.
"""
