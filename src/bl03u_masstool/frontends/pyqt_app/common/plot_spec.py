from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PlotProfile(str, Enum):
    """Visual profile used by the Matplotlib renderer."""

    SCREEN = "screen"
    PUBLICATION = "publication"
    MONOCHROME = "monochrome"


class SeriesRole(str, Enum):
    """Scientific meaning of a curve, independent from renderer styling."""

    EXPERIMENTAL = "experimental"
    TOTAL_FIT = "total_fit"
    COMPONENT = "component"
    MEASUREMENT = "measurement"
    COMPARISON = "comparison"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class CurveSeries:
    """One ordered curve in a scientific plot.

    The renderer must preserve the supplied point order. Data preparation,
    sorting, smoothing, interpolation, and normalization belong upstream.
    """

    key: str
    role: SeriesRole
    x: Iterable[float]
    y: Iterable[float]
    label: str = ""
    contribution_percent: float | None = None
    color: str | None = None


@dataclass(frozen=True)
class VerticalReference:
    """A labelled x-axis reference, such as a candidate ionization energy."""

    key: str
    x: float
    label: str
    color: str | None = None
    emphasized: bool = False


@dataclass(frozen=True)
class ScientificPlotSpec:
    """Renderer-neutral description of a low-interaction scientific plot."""

    title: str
    xlabel: str
    ylabel: str
    series: tuple[CurveSeries, ...]
    show_legend: bool = False
    legend_loc: str = "best"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    show_zero_line: bool = False
    component_visibility_threshold: float = 1e-6
    vertical_references: tuple[VerticalReference, ...] = ()
