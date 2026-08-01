"""Frame-level rally lifecycle spotting.

This package deliberately keeps rally-boundary learning separate from the
legacy clip classifier.  It learns four timestamped lifecycle events on a
continuous video timeline:

``serve_contact -> reception_contact -> followup_contact -> rally_end``.

The public modules are:

* :mod:`prepare` — export normalized per-frame features and labels.
* :mod:`model` — a compact temporal convolutional network.
* :mod:`metrics` — NMS and tolerance-based event metrics.
* :mod:`train` — video-grouped training and evaluation.
"""

from .taxonomy import EVENT_TYPES, NUM_CLASSES

__all__ = ["EVENT_TYPES", "NUM_CLASSES"]
