import functools
from typing import Any, Callable, Optional, Sequence

from analysis import np_utils
import numpy as np
import pandas as pd

from analysis.fixation_filters.core import FixationClassification, GazeVector


def annotate_gazes(
    gaze_df: pd.DataFrame,
    classification: FixationClassification,
    columns_to_interpolate: Optional[Sequence[str]] = None,
):
    gaze_df = gaze_df.copy()
    if classification.interpolations and columns_to_interpolate:
        source = classification.interpolations.source
        target = classification.interpolations.target
        for col in columns_to_interpolate:
            values = gaze_df[col].to_numpy().copy()
            values[source] = values[target]
            gaze_df.loc[:, col] = values
        is_interpolated = np.zeros(len(gaze_df), dtype=bool)
        is_interpolated[source] = True
        gaze_df["interpolated"] = is_interpolated

    gaze_df["is_fixation"] = classification.fixations.is_fixation
    fixation_ids = np.empty(len(gaze_df))
    fixation_ids[:] = np.nan
    fixation_ids[
        classification.fixations.is_fixation
    ] = classification.fixations.fixation_ids

    gaze_df = gaze_df.assign(fixation_id=fixation_ids)

    return gaze_df


def explode_tuple_dict(d: dict[tuple | str, Any]) -> dict[str, Any]:
    flattened = {}
    for key, value in d.items():
        if isinstance(key, tuple):
            for subkey in key:
                flattened[subkey] = value
        else:
            flattened[key] = value
    return flattened


def coerce_line_cols(line_col_modes, max_dispersion: float = 2):
    if len(line_col_modes) == 0:
        return np.nan

    if isinstance(line_col_modes[0], (float, int)):
        return line_col_modes
    xs = [x for x, _ in line_col_modes]
    ys = [y for _, y in line_col_modes]

    distances = np.sqrt(
        np.diff(
            xs,
        )
        ** 2
        + np.diff(
            ys,
        )
        ** 2
    )
    if (distances <= max_dispersion).all():
        return line_col_modes[0]

    return np.nan


def apply_interpolation(
    df: pd.DataFrame,
    cols: list[str] | str,
    interpolation_source,
    interpolation_targets: np.ndarray,
    indicator=None,
):
    source_mask = np.zeros(len(df), dtype=bool)
    source_mask[interpolation_source] = True
    target_mask = np.zeros(len(df), dtype=bool)
    target_mask[interpolation_targets] = True
    df.loc[source_mask, cols] = df[target_mask][cols]
    if indicator is not None:
        if indicator in df.columns:
            df[indicator] = source_mask | df[indicator]
        else:
            df[indicator] = source_mask

    return df


def interpolate_invalid_positions(
    df: pd.DataFrame, line: str, col: str, time: str, max_dt=100
):
    for col in [line, col]:
        invalid = (df[col] < 0).to_numpy()
        source, target = np_utils.find_nearest_valid_neighbor(
            invalid, df[time].to_numpy(), max_dt
        )
        df = apply_interpolation(
            df, col, source, target, indicator="position_interpolated"
        )

    return df


def compute_fixation_estimates(
    gazes: pd.DataFrame,
    time: str = "plugin_time",
    is_fixation: str = "is_fixation",
    fixation_id: str = "fixation_id",
    is_interpolated: str = "interpolated",
    line: str = "source_file_line",
    col: str = "source_file_col",
    aggregations: Optional[dict[tuple | str, Any]] = None,
    interpolate_positions: bool = True,  # whether or not to interpolate invalid document positions (line/col numbers)
    position_interpolation_max_dt: int = 100,  # maximum time difference (in ms) between two gazes to interpolate their positions
):
    gazes = gazes.copy()
    if interpolate_positions:
        # interpolate line and column numbers
        gazes = interpolate_invalid_positions(
            gazes, line, col, time, max_dt=position_interpolation_max_dt
        )
    # replace invalid line/col with nan
    gazes.loc[gazes[line] < 0, line] = np.nan
    gazes.loc[gazes[col] < 0, col] = np.nan

    gazes["line_col"] = [
        (l, c) if l >= 0 and c >= 0 else np.nan for l, c in zip(gazes[line], gazes[col])
    ]
    gazes["valid"] = (gazes[line] >= 0) & (gazes[col] >= 0)

    grouper = gazes[gazes[is_fixation]].groupby(gazes[fixation_id])
    result = grouper.agg(
        {
            **explode_tuple_dict(aggregations or {}),
            "valid": lambda x: x.sum()
            / x.count(),  # how many valid gazes / total gazes
            "line_col": pd.Series.mode,
            is_interpolated: "sum",
        }
    )
    result["n"] = grouper[time].count()
    result["start_time"] = grouper[time].agg("min")
    result["end_time"] = grouper[time].agg("max")
    result["duration"] = result["end_time"] - result["start_time"]
    result = result.assign(
        line_col=result["line_col"].apply(
            functools.partial(coerce_line_cols, max_dispersion=np.inf)
        )
    )
    result = result[result["line_col"].notna()]
    result[line] = result["line_col"].str[0].astype(int)
    result[col] = result["line_col"].str[1].astype(int)
    result = result.drop(
        columns=[
            "line_col",
        ]
    )

    return result.reset_index()


def apply_fixation_filter(
    gazes: pd.DataFrame,
    fixation_filter: Callable[[GazeVector], FixationClassification],
    time: str = "plugin_time",
    line: str = "source_file_line",
    col: str = "source_file_col",
    validation_cols: Sequence[str] = ("left_validation", "right_validation"),
    coords: Sequence[str] = ("x", "y"),
    aggregations: Optional[dict[tuple | str, Any]] = None,
    interpolate_positions: bool = True,  # whether or not to interpolate invalid document positions (line/col numbers)
    position_interpolation_max_dt: int = 100,  # maximum time difference (in ms) between two gazes to interpolate their positions
):
    if not isinstance(coords, list):
        coords = list(coords)

    gazes = gazes.sort_values(by=time)

    mask = (gazes[coords] < 0).any(axis=1)
    mask = mask | (gazes[list(validation_cols)] != 1).any(axis=1)

    gaze_vector = GazeVector.from_df(
        gazes, coords=coords, time=time, mask=(~mask).to_numpy()
    )
    classification = fixation_filter(gaze_vector)
    annotated_gazes = annotate_gazes(
        gazes,
        classification,
        columns_to_interpolate=[
            *coords,
            line,
            col,
        ],
    )

    fixation_estimates = compute_fixation_estimates(
        annotated_gazes,
        time=time,
        line=line,
        col=col,
        aggregations=aggregations,
        interpolate_positions=interpolate_positions,
        position_interpolation_max_dt=position_interpolation_max_dt,
    )

    return annotated_gazes, fixation_estimates
