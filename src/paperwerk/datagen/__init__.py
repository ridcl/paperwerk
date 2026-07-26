"""Data-generation toolkit: templates, rendering, value synthesis, augmentation.

Re-exports the primitives most callers need so they can be imported straight
from the package:

    from paperwerk.datagen import make_template, render, random_values, augment
"""

from .augment import augment
from .render import render, annotate
from .templates import make_template
from .values import random_values

__all__ = ["make_template", "render", "random_values", "augment", "annotate"]
