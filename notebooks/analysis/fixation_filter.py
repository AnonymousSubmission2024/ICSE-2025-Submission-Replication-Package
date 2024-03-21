import dataclasses
import pandas as pd
import numpy as np
import collections

from typing import Optional


def interpolate_gazes(
    df: pd.DataFrame,
    x: str,
    y: str,
    left_validation: Optional[str] = None,
    right_validation: Optional[str] = None,
    partial=False,
):
    # Interpolation
    # We want to interpolate gaze points as a whole,
    # and not just x & y independently
    df = df.copy()
    if partial:
        # forward fill x/y independently
        df[x] = df[x].fillna(method="ffill")
        df[y] = df[y].fillna(method="ffill")
    else:
        # forward fill entire rows
        invalid_xys = (df[x] < 0) | (df[y] < 0)
        df.loc[invalid_xys, :] = np.nan
        df = df.fillna(method="ffill")

    invalid_gazes = np.zeros(len(df), dtype=bool)
    if left_validation is not None:
        invalid_gazes = invalid_gazes | (df[left_validation] == 0)
    if right_validation is not None:
        invalid_gazes = invalid_gazes | (df[right_validation] == 0)

    return df[~invalid_gazes]


@dataclasses.dataclass
class Fixations:
    gazes: pd.DataFrame
    is_fixation_point: pd.Series
    fixation_ids: pd.Series
    fixations: pd.DataFrame


def ivt_gaze_classification(
    gazes: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    line: str = "source_file_line",
    col: str = "source_file_col",
    time: str = "plugin_time",
    left_validation: str = "left_validation",
    right_validation: str = "right_validation",
    velocity_threshold: float = 50,
    duration_threshold: float = 80,
    line_col_estimation_frequency_threshold: int = 1,
    cols_to_average: list[str] = [
        "left_pupil_diameter",
        "right_pupil_diameter",
    ],
    constant_cols: list[str] = ["pid", "tid", "gaze_target"],
):
    gazes = interpolate_gazes(
        gazes, x, y, left_validation, right_validation, partial=False
    )
    is_valid = (
        (gazes[x] >= 0) & (gazes[y] >= 0) & (~gazes[x].isna()) & (~gazes[y].isna())
    )
    # Compute velocity
    velocity: pd.Series = np.sqrt(
        gazes[x].diff() ** 2 + gazes[y].diff() ** 2
    ).fillna(  # type: ignore
        0
    )
    is_fixation_point = velocity <= velocity_threshold
    fixation_group_ids = (is_fixation_point != is_fixation_point.shift()).cumsum()

    def compute_fixation_groups(df):
        duration = df[time].max() - df[time].min()

        positions = list(map(tuple, zip(df[line], df[col])))
        positions = [
            (line, col)
            for line, col in zip(df[line], df[col])
            if (line >= 0 and col >= 0)
        ]
        if not positions:
            return pd.DataFrame()

        frequencies = collections.Counter(positions)
        most_common, count = frequencies.most_common(1)[0]

        if (
            duration >= duration_threshold
            and count >= line_col_estimation_frequency_threshold
        ):
            return pd.DataFrame(
                {
                    "duration": [duration],
                    "n": [len(df)],
                    time: [df[time].min()],
                    line: [most_common[0]],
                    col: [most_common[1]],
                    **{k: [df[k].iloc[0]] for k in constant_cols if k in df.columns},
                    **{
                        k: [df[k].mean()]
                        for k in (cols_to_average + [x, y])
                        if k in df.columns
                    },
                }
            )
        return pd.DataFrame()

    gazes = gazes[is_valid]
    fixation_group_ids = fixation_group_ids[is_valid]
    fixation_df = (
        gazes[is_fixation_point]
        .groupby(fixation_group_ids[is_fixation_point])
        .apply(compute_fixation_groups)
    )

    return Fixations(gazes, is_fixation_point, fixation_group_ids, fixation_df)
