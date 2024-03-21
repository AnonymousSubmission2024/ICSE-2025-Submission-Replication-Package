import dataclasses
from typing import Sequence, Protocol, cast
import abc
import pandas as pd
import tree
import numpy as np
from analysis import np_utils


@dataclasses.dataclass
class FixationMapping:
    time: str = "system_time"
    fixation_duration: str = "duration"
    line: str = "source_file_line"
    col: str = "source_file_col"
    target: str = "gaze_target"
    snippet: str = "snippet"
    left_pupil_diameter: str = "left_pupil_diameter"
    right_pupil_diameter: str = "right_pupil_diameter"
    aoi: str = "aoi"
    aoi_size_in_lines: str = "aoi_size_in_lines"
    aoi_size_in_chars: str = "aoi_size_in_chars"


def assign_aois(
    df: pd.DataFrame,
    annotations: dict,
    source_files: dict[str, str],
    config: FixationMapping,
) -> pd.DataFrame:
    """Assigns AOIs to fixations.

    Args:
        df: dataframe containing fixation or gaze data
        annotations: aoi annotations
        source_files: all sourcefiles that this dataframe might refer to.
        config: A mapping describing columns in the fixation dataframe.

    Returns:
       the original dataframe with two added columns
    """
    for snippet, annotation in annotations.items():
        subset = (df[config.snippet] == snippet) & (
            df[config.target] == annotation["file"]
        )
        source_file = source_files[annotation["file"]]
        assigned_aoi_size_in_lines = 0
        assigned_aoi_size_in_chars = 0
        for aoi, aoi_ranges in annotation["aois"].items():
            if len(aoi_ranges) > 0 and not isinstance(aoi_ranges[0], (list, tuple)):
                aoi_ranges = (aoi_ranges,)
            aoi_ranges = cast(
                Sequence[tuple[int, int]],
                tree.map_structure(lambda d: d - 1, aoi_ranges),
            )
            aoi_size_in_lines = sum(
                (aoi_range[1] - aoi_range[0]) for aoi_range in aoi_ranges
            )
            aoi_size_in_chars = sum(
                len(source_file[aoi_range[0] : aoi_range[1] + 1])
                for aoi_range in aoi_ranges
            )
            assigned_aoi_size_in_lines += aoi_size_in_lines
            assigned_aoi_size_in_chars += aoi_size_in_chars
            for aoi_range in aoi_ranges:  # type: ignore
                in_aoi_range = subset & (df[config.line].between(*aoi_range))
                df.loc[in_aoi_range, config.aoi] = aoi
                # aois include last line
                df.loc[
                    in_aoi_range,
                    config.aoi_size_in_lines,
                ] = aoi_size_in_lines
                df.loc[
                    in_aoi_range,
                    config.aoi_size_in_chars,
                ] = aoi_size_in_chars

        # For anything we couldn't assign an AOI, assign it to a default value
        unassigned_aoi_range = subset & df[config.aoi].isna()
        df.loc[unassigned_aoi_range, config.aoi] = "other"
        df.loc[unassigned_aoi_range, config.aoi_size_in_chars] = (
            len(source_file) - assigned_aoi_size_in_chars
        )
        df.loc[unassigned_aoi_range, config.aoi_size_in_lines] = (
            len(source_file.splitlines()) - assigned_aoi_size_in_chars
        )
    return df


def line_regression_rate(fixations: pd.DataFrame):
    line_diff = np.diff(fixations["source_file_line"].to_numpy(), prepend=np.nan)
    col_diff = np.diff(fixations["source_file_col"].to_numpy(), prepend=np.nan)
    line_regressions = (line_diff == 0) & (col_diff < 0)
    return np.sum(line_regressions) / len(fixations)


def get_regressions(lines: np.ndarray, cols: np.ndarray):
    line_diffs = np.diff(lines, prepend=np.nan)
    col_diffs = np.diff(cols, prepend=np.nan)
    regressions = (line_diffs < 0) | ((line_diffs == 0) & (col_diffs < 0))
    return regressions


def regression_rate(fixations: pd.DataFrame) -> pd.Series:
    regressions = get_regressions(
        fixations["source_file_line"].to_numpy(),
        fixations["source_file_col"].to_numpy(),
    )
    return np.sum(regressions) / len(fixations)


def horizontal_later(fixations: pd.DataFrame):
    line_diff = np.diff(fixations["source_file_line"].to_numpy(), prepend=np.nan)
    col_diff = np.diff(fixations["source_file_col"].to_numpy(), prepend=np.nan)
    horizontal_later_fixations = (line_diff == 0) & (col_diff >= 0)

    return np.sum(horizontal_later_fixations) / len(fixations)


def vertical_later(fixations: pd.DataFrame):
    line_diff = np.diff(fixations["source_file_line"].to_numpy(), prepend=np.nan)
    vertical_later_fixations = line_diff >= 0
    return np.sum(vertical_later_fixations) / len(fixations)


def vertical_next(fixations: pd.DataFrame):
    line_diff = np.diff(fixations["source_file_line"].to_numpy(), prepend=np.nan)
    vertical_later_fixations = (line_diff >= 0) & (line_diff <= 1)
    return np.sum(vertical_later_fixations) / len(fixations)


# ************************************************
#       Fixation Sequence  Alignment
# ************************************************
def parse_model_sequence(lines_or_tuples):
    res = []
    for line in lines_or_tuples:
        if isinstance(line, (tuple, list)):
            res.extend(list(range(line[0], line[1] + 1)))
        else:
            res.append(line)
    return res


def get_reading_order_models(annotations: dict, source_index: dict):
    results = {}
    for snippet_name, contents in annotations.items():
        reading_orders = contents["reading_orders"]
        source = source_index[contents["file"]]
        source_lines = source.splitlines()
        results[snippet_name] = {}
        for reading_order, line_sequence in reading_orders.items():
            line_order = parse_model_sequence(line_sequence)
            # offset by 1 to account for 1-indexing in yaml
            line_order = [line - 1 for line in line_order]
            # remove empty lines
            line_order = [
                line for line in line_order if len(source_lines[line].strip()) > 0
            ]
            results[snippet_name][reading_order] = line_order
    return results


def create_binned_fixation_sequence(durations, line_numbers, binsize=10):
    repeats = np.round(durations / binsize).astype(int)
    return np.repeat(line_numbers, repeats)


def create_unique_fixation_sequence(line_numbers):
    return line_numbers[(line_numbers != np_utils.shift(line_numbers))]
