from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .curve_database import (
    CurveChannel,
    CurveChannelSummary,
    find_curve_channels,
    list_curve_channels,
    load_curve_channel,
)


class CurveRepository(Protocol):
    """Small read API shared by project SQLite and temporary in-memory data."""

    def list_channels(self) -> list[CurveChannelSummary]: ...

    def load_channel(self, channel_id: int) -> CurveChannel: ...

    def find_channels(
        self,
        exact_mz: float,
        tolerance: float = 1e-6,
    ) -> list[CurveChannelSummary]: ...

    def iter_channel_batches(
        self,
        batch_size: int = 20,
    ) -> Iterator[list[CurveChannel]]: ...


def channel_to_curve(channel: CurveChannel) -> dict[str, Any]:
    """Convert a stored channel into the existing UI curve mapping."""
    rows = channel.rows.copy()
    exact_mz = float(channel.exact_mz)
    curve: dict[str, Any] = {
        "mz": exact_mz,
        "mz_rounded": int(channel.nominal_mz),
        "mz_exact_mean": exact_mz,
        "curve_key": exact_mz,
        "species": channel.species_label,
        "rows": rows,
        "channel_id": int(channel.channel_id),
        "dataset_id": int(channel.dataset_id),
        "has_nominal_collision": False,
    }
    if channel.curve_type == "temperature":
        curve["temperatures"] = _float_values(rows, "temperature")
        curve["areas"] = _float_values(rows, "area")
        curve["temperature_peak_track"] = channel.peak_track
        for name in ("curve_class", "curve_class_label", "curve_class_reason"):
            if name in rows and not rows[name].dropna().empty:
                curve[name] = rows[name].dropna().iloc[0]
    else:
        curve["energies"] = _float_values(rows, "energy")
        curve["intensities"] = _float_values(rows, "normalized_intensity")
    return curve


def _float_values(rows: pd.DataFrame, column: str) -> list[float]:
    if column not in rows:
        return []
    return pd.to_numeric(rows[column], errors="coerce").astype(float).tolist()


class SQLiteCurveRepository:
    """Lazy SQLite repository with a bounded channel-level LRU."""

    def __init__(
        self,
        database_path: str | Path,
        dataset_id: int,
        *,
        cache_size: int = 32,
    ) -> None:
        self.database_path = Path(database_path)
        self.dataset_id = int(dataset_id)
        self.cache_size = max(1, int(cache_size))
        self._summaries: list[CurveChannelSummary] | None = None
        self._cache: OrderedDict[int, CurveChannel] = OrderedDict()

    def list_channels(self) -> list[CurveChannelSummary]:
        if self._summaries is None:
            self._summaries = list_curve_channels(
                self.database_path,
                self.dataset_id,
            )
        return list(self._summaries)

    def load_channel(self, channel_id: int) -> CurveChannel:
        channel_id = int(channel_id)
        cached = self._cache.pop(channel_id, None)
        if cached is not None:
            self._cache[channel_id] = cached
            return cached
        channel = load_curve_channel(self.database_path, channel_id)
        if channel.dataset_id != self.dataset_id:
            raise ValueError(
                f"Curve channel {channel_id} does not belong to "
                f"dataset {self.dataset_id}"
            )
        self._cache[channel_id] = channel
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return channel

    def find_channels(
        self,
        exact_mz: float,
        tolerance: float = 1e-6,
    ) -> list[CurveChannelSummary]:
        return find_curve_channels(
            self.database_path,
            self.dataset_id,
            float(exact_mz),
            float(tolerance),
        )

    def iter_channel_batches(
        self,
        batch_size: int = 20,
    ) -> Iterator[list[CurveChannel]]:
        size = max(1, int(batch_size))
        summaries = self.list_channels()
        for start in range(0, len(summaries), size):
            yield [
                self.load_channel(summary.channel_id)
                for summary in summaries[start : start + size]
            ]

    def clear_cache(self) -> None:
        self._cache.clear()


class RepositoryCurveMapping(Mapping[float, dict[str, Any]]):
    """Mapping facade that preserves the legacy UI API while loading lazily."""

    def __init__(self, repository: SQLiteCurveRepository) -> None:
        self.repository = repository
        self._summaries = repository.list_channels()
        self._by_mz = {
            float(summary.exact_mz): summary for summary in self._summaries
        }

    def __len__(self) -> int:
        return len(self._summaries)

    def __iter__(self):
        return iter(self._by_mz)

    def __getitem__(self, exact_mz: float) -> dict[str, Any]:
        summary = self._by_mz[float(exact_mz)]
        return channel_to_curve(self.repository.load_channel(summary.channel_id))

    @property
    def point_count(self) -> int:
        return sum(summary.point_count for summary in self._summaries)

    @property
    def max_point_count(self) -> int:
        return max((summary.point_count for summary in self._summaries), default=0)

    def metadata(self, exact_mz: float) -> dict[str, Any]:
        summary = self._by_mz[float(exact_mz)]
        metadata = dict(summary.metadata)
        metadata.update({
            "mz": summary.exact_mz,
            "mz_exact_mean": summary.exact_mz,
            "mz_rounded": summary.nominal_mz,
            "species": summary.species_label,
            "channel_id": summary.channel_id,
            "dataset_id": summary.dataset_id,
            "left_bound": summary.left_bound,
            "right_bound": summary.right_bound,
            "point_count": summary.point_count,
            "energy_count": summary.energy_count,
        })
        return metadata


class InMemoryCurveRepository:
    """Repository adapter used only for temporary, unsaved curve results."""

    def __init__(
        self,
        curve_type: str,
        curves: Mapping[int | float, Mapping[str, Any]],
    ) -> None:
        self.curve_type = str(curve_type)
        self._channels: dict[int, CurveChannel] = {}
        for channel_id, (key, curve) in enumerate(
            sorted(curves.items(), key=lambda item: float(item[0])),
            start=1,
        ):
            rows = curve.get("rows")
            if not isinstance(rows, pd.DataFrame):
                axis_name = (
                    "temperature" if self.curve_type == "temperature" else "energy"
                )
                value_name = (
                    "area"
                    if self.curve_type == "temperature"
                    else "normalized_intensity"
                )
                rows = pd.DataFrame(
                    {
                        axis_name: curve.get(
                            "temperatures"
                            if self.curve_type == "temperature"
                            else "energies",
                            [],
                        ),
                        value_name: curve.get(
                            "areas"
                            if self.curve_type == "temperature"
                            else "intensities",
                            [],
                        ),
                    }
                )
            exact_mz = float(curve.get("mz_exact_mean", curve.get("mz", key)))
            self._channels[channel_id] = CurveChannel(
                channel_id=channel_id,
                dataset_id=0,
                curve_type=self.curve_type,
                exact_mz=exact_mz,
                nominal_mz=int(round(float(curve.get("mz_rounded", exact_mz)))),
                peak_track=_optional_int(
                    curve.get(
                        "temperature_peak_track",
                        curve.get("temperature_peak_cluster"),
                    )
                ),
                left_bound=_first_finite(rows, "left_bound"),
                right_bound=_first_finite(rows, "right_bound"),
                species_label=str(curve.get("species", "") or ""),
                rows=rows.reset_index(drop=True).copy(),
            )

    def list_channels(self) -> list[CurveChannelSummary]:
        return [
            CurveChannelSummary(
                channel_id=channel.channel_id,
                dataset_id=0,
                exact_mz=channel.exact_mz,
                nominal_mz=channel.nominal_mz,
                peak_track=channel.peak_track,
                left_bound=channel.left_bound,
                right_bound=channel.right_bound,
                species_label=channel.species_label,
                point_count=len(channel.rows),
                energy_count=(
                    int(
                        channel.rows["photon_energy"].nunique(dropna=True)
                    )
                    if "photon_energy" in channel.rows
                    else 0
                ),
                metadata={
                    key: channel.rows[key].iloc[0]
                    for key in (
                        "curve_class",
                        "curve_class_label",
                        "curve_class_reason",
                    )
                    if key in channel.rows and not channel.rows.empty
                },
            )
            for channel in self._channels.values()
        ]

    def load_channel(self, channel_id: int) -> CurveChannel:
        try:
            return self._channels[int(channel_id)]
        except KeyError as exc:
            raise ValueError(f"Curve channel {int(channel_id)} does not exist") from exc

    def find_channels(
        self,
        exact_mz: float,
        tolerance: float = 1e-6,
    ) -> list[CurveChannelSummary]:
        target = float(exact_mz)
        return sorted(
            (
                summary
                for summary in self.list_channels()
                if abs(summary.exact_mz - target) <= float(tolerance)
            ),
            key=lambda summary: (abs(summary.exact_mz - target), summary.channel_id),
        )

    def iter_channel_batches(
        self,
        batch_size: int = 20,
    ) -> Iterator[list[CurveChannel]]:
        size = max(1, int(batch_size))
        ids = list(self._channels)
        for start in range(0, len(ids), size):
            yield [self._channels[channel_id] for channel_id in ids[start : start + size]]


@dataclass(frozen=True)
class CurveParityReport:
    channel_count: int
    point_count: int


def validate_repository_round_trip(
    curves: Mapping[int | float, Mapping[str, Any]],
    repository: CurveRepository,
    *,
    exact_mz_atol: float = 1e-9,
    signal_rtol: float = 1e-7,
    signal_atol: float = 1e-12,
) -> CurveParityReport:
    """Raise on a shadow-write mismatch and return compact parity counts."""
    summaries = repository.list_channels()
    ordered_original = sorted(curves.values(), key=lambda curve: float(
        curve.get("mz_exact_mean", curve.get("mz", 0.0))
    ))
    if len(ordered_original) != len(summaries):
        raise ValueError(
            f"Channel count mismatch: memory={len(ordered_original)}, "
            f"sqlite={len(summaries)}"
        )
    point_count = 0
    for original, summary in zip(ordered_original, summaries, strict=True):
        exact_mz = float(original.get("mz_exact_mean", original.get("mz", 0.0)))
        if not np.isclose(exact_mz, summary.exact_mz, rtol=0.0, atol=exact_mz_atol):
            raise ValueError(
                f"Exact m/z mismatch: memory={exact_mz:.12g}, "
                f"sqlite={summary.exact_mz:.12g}"
            )
        nominal_mz = int(round(float(original.get("mz_rounded", exact_mz))))
        if nominal_mz != summary.nominal_mz:
            raise ValueError(
                f"Nominal m/z mismatch at {exact_mz:.12g}: "
                f"memory={nominal_mz}, sqlite={summary.nominal_mz}"
            )
        stored = repository.load_channel(summary.channel_id)
        original_rows = original.get("rows")
        if not isinstance(original_rows, pd.DataFrame):
            raise ValueError("Round-trip validation requires curve rows")
        original_rows = original_rows.reset_index(drop=True)
        stored_rows = stored.rows.reset_index(drop=True)
        if len(original_rows) != len(stored_rows):
            raise ValueError(
                f"Point count mismatch at m/z {exact_mz:.12g}: "
                f"memory={len(original_rows)}, sqlite={len(stored_rows)}"
            )
        for bound_column, stored_bound in (
            ("left_bound", summary.left_bound),
            ("right_bound", summary.right_bound),
        ):
            original_bound = _first_finite(original_rows, bound_column)
            if original_bound is None and stored_bound is None:
                continue
            if (
                original_bound is None
                or stored_bound is None
                or not np.isclose(
                    original_bound,
                    stored_bound,
                    rtol=0.0,
                    atol=exact_mz_atol,
                )
            ):
                raise ValueError(
                    f"{bound_column} mismatch at m/z {exact_mz:.12g}"
                )
        axis_column = "temperature" if stored.curve_type == "temperature" else "energy"
        _assert_numeric_column_equal(
            original_rows,
            stored_rows,
            axis_column,
            rtol=0.0,
            atol=0.0,
        )
        for signal_column in (
            ("area", "raw_area", "photon_normalized_area", "normalized_area")
            if stored.curve_type == "temperature"
            else (
                "normalized_intensity",
                "raw_area",
                "photon_normalized_intensity",
            )
        ):
            if signal_column in original_rows:
                _assert_numeric_column_equal(
                    original_rows,
                    stored_rows,
                    signal_column,
                    rtol=signal_rtol,
                    atol=signal_atol,
                )
        point_count += len(stored_rows)
    return CurveParityReport(len(summaries), point_count)


def _assert_numeric_column_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    if column not in right:
        raise ValueError(f"Stored curve is missing column {column}")
    left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(
        left_values,
        right_values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        raise ValueError(f"Curve column mismatch: {column}")


def _first_finite(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
