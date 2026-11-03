"""Shared scoring and run logging for the GDC harmonization experiments (E0).

Every method, run and track scores with the same rules and writes the same prediction and log formats,
so results stay comparable. See hbench/README.md.
"""

__version__ = "0.1.0"

# Bump the major part when any correctness rule changes, the minor part when metrics are added,
# the patch part for fixes that leave every committed number unchanged (checked by the regression test).
SCORING_VERSION = "hbench-score/1.0.0"
