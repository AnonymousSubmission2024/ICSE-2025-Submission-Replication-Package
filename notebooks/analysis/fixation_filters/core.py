import dataclasses
import functools
from typing import Any, Callable, Optional, Protocol, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from analysis import np_utils


@dataclasses.dataclass
class GazeVector:
    coords: NDArray[np.float_ | np.int_]
    """gaze coordinates. shape(n_gazes, n_dims)"""
    time: NDArray[np.float_ | np.int_]
    """timestamps for each coord. shape: (n_gazes,)"""
    mask: NDArray[np.bool_]
    """indices which entries are valid. shape: (n_gazes,)"""

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        coords: Sequence[str],
        time: str,
        mask: Optional[ArrayLike] = None,
    ):
        if mask is None:
            mask = np.ones(len(coords), dtype=np.bool_)
        coord_arr = df[list(coords)].T.to_numpy()
        if coord_arr.ndim == 1:
            coord_arr = coord_arr[np.newaxis, :]

        return GazeVector(
            coords=coord_arr,
            time=cast(NDArray[np.float_ | np.int_], df[time].values),
            mask=cast(NDArray[np.bool_], mask),
        )

    @staticmethod
    def from_arrays(
        coords: Sequence | np.ndarray,
        time: Sequence | np.ndarray,
        mask: Optional[ArrayLike] = None,
    ):
        if mask is None:
            mask = np.ones(len(coords), dtype=np.bool_)

        return GazeVector(
            coords=np.array(coords), time=np.array(time), mask=np.array(mask)
        )


@dataclasses.dataclass
class InterpolationResult:
    source: NDArray[np.int_]
    target: NDArray[np.int_]

    def apply(self, gazes: GazeVector) -> GazeVector:
        coords = gazes.coords.copy()
        coords[:, self.source] = coords[:, self.target]

        mask = gazes.mask.copy()
        mask[self.source] = True

        return dataclasses.replace(gazes, coords=coords, mask=mask)


class Interpolator(Protocol):
    def __call__(self, gazes: GazeVector) -> InterpolationResult:
        ...


@dataclasses.dataclass
class FixationPoints:
    is_fixation: NDArray[np.bool_]

    @property
    def fixation_ids(self):
        fixation_ids = np_utils.bins_from_repeats(self.is_fixation)[self.is_fixation]
        return (
            np_utils.bins_from_repeats(fixation_ids) - 1
        )  # reset fixation ids and shift back to one


class FixationAlgorithm(Protocol):
    def __call__(self, gazes: GazeVector) -> FixationPoints:
        ...


FixationTransform = Callable[[GazeVector, FixationPoints], FixationPoints]


@dataclasses.dataclass
class FixationClassification:
    fixations: FixationPoints
    interpolations: Optional[InterpolationResult] = None


def nn_interpolator(gazes: GazeVector, radius=100) -> InterpolationResult:
    source, target = np_utils.find_nearest_valid_neighbor(
        ~gazes.mask, gazes.time, radius=radius
    )

    return InterpolationResult(source=source, target=target)


def gaze_velocity(coords: np.ndarray, ts: Optional[np.ndarray] = None) -> np.ndarray:
    """Computes gaze velocity.

    Args:
        coords: array of coordinates of shape [ndims, n_gazes]
        ts: array of timestamps for x/y coordinates of shape [n_gazes]

    Returns:
        np.ndarray: velocity of shape [n_gazes]
    """
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    if ts is not None:
        _, n_gazes = coords.shape
        assert ts.ndim == 1, "`ts` must be a 1D array"
        assert n_gazes == len(
            ts
        ), f"`ts`({len(ts)}) must have the same length as `coords`({coords.shape})"

    dcoords = np.diff(coords, prepend=np.nan) ** 2
    vs = np.sqrt(np.sum(dcoords, axis=0))
    if ts is not None:
        vs /= np.diff(ts, prepend=np.nan)
    # vs[0] = 0
    return vs


def ivt(
    gazes: GazeVector,
    velocity_threshold: float = 50,
) -> FixationPoints:
    """Computes fixation groups using IVT algorithm.

    Example:
        >>> gazes = pd.read_csv('gazes.csv')
        >>> ivt_result = compute_ivt_fixation_groups(gazes['x'], gazes['y'], gazes['timestamp'])
        >>> fixation_points = gazes[ivt_result.fixation_points]
        >>> fixation_groups = gazes.groupby(ivt_result.fixation_group_ids).apply(...compute fixation estimate...)

    Args:
        gazes: gaze data
        velocity_threshold: velocity threshold for IVT. Defaults to 50.

    Returns:
        FixationClassification: fixation points and fixation group ids
    """
    vs = gaze_velocity(gazes.coords)
    under_velocity_threshold = vs <= velocity_threshold
    is_fixation_point = under_velocity_threshold & gazes.mask

    return FixationPoints(
        is_fixation=is_fixation_point,  # fixation_ids=fixation_ids
    )


def constrain_gaps_by_time(
    gazes: GazeVector, classification: FixationPoints, max_gap: float = 100
) -> FixationPoints:
    # To account for changes in sampling frequency
    # due to delays in itrace or corresponding IDE plugins,
    # we reset fixation points that are separated by a time delta
    # greater than `max_delta_t`.
    dts = np.diff(gazes.time, prepend=np.nan)
    under_duration_threshold = (dts < max_gap) & ~np.isnan(dts)
    is_fixation = classification.is_fixation & under_duration_threshold

    return FixationPoints(is_fixation)


def fixation_duration_threshold(
    gazes: GazeVector, classification: FixationPoints, min_duration: float = 100
) -> FixationPoints:
    durations = np_utils.bin_sptp(
        gazes.time[classification.is_fixation], classification.fixation_ids, strict=True
    )
    over_duration_threshold = durations >= min_duration

    is_fixation = classification.is_fixation.copy()
    is_fixation[is_fixation] = over_duration_threshold[classification.fixation_ids]

    return FixationPoints(is_fixation)


def remove_singletons(gazes: GazeVector, classification: FixationPoints):
    fixation_ids = classification.fixation_ids
    fixation_sizes = np_utils.bin_sptp(np.arange(len(fixation_ids)), fixation_ids) + 1
    singletons = fixation_sizes == 1
    is_fixation = classification.is_fixation.copy()
    is_fixation[is_fixation] = ~singletons[fixation_ids]
    return FixationPoints(is_fixation)


def fixation_filter(
    gazes: GazeVector,
    algorithm: FixationAlgorithm,
    interpolator: Optional[Interpolator] = None,
    transforms: Sequence[FixationTransform] = (),
) -> FixationClassification:
    interpolated_gazes = gazes
    interpolation_result = None

    if interpolator is not None:
        interpolation_result = interpolator(gazes)
        interpolated_gazes = interpolation_result.apply(gazes)

    classification = algorithm(interpolated_gazes)
    for transform in transforms:
        classification = transform(interpolated_gazes, classification)

    return FixationClassification(classification, interpolations=interpolation_result)
