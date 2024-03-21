import argparse
import collections
import multiprocessing as mp
import pathlib
from typing import Any, Callable

import fiddle as fdl
import pandas as pd
import pandera as pa
from loguru import logger
from returns.result import Failure, Result, Success
from tqdm.auto import tqdm

from analysis import configs
from analysis.configs import get_fiddle_fingerprint, ivt_optimised_config
from analysis.fixation_filters import apply_fixation_filter


def get_fixations(
    df,
    cfg: fdl.Partial[Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]],
):
    return apply_fixation_filter(
        df,
        fdl.build(cfg),
        aggregations={
            ("left_pupil_diameter", "right_pupil_diameter", "x", "y"): "mean",
            "gaze_target": lambda x: x.iloc[0],
            "position_interpolated": "sum",
        },
    )


def run_fixation_filters(core_file, plugin_file, output_file):
    categorical_cols = [
        "pid",
        "tid",
        "gaze_target_type",
        "source_file_path",
        "gaze_target",
    ]
    # Load data
    logger.info(f'Loading data from "{str(core_file)}" and "{str(plugin_file)}"')
    core = pd.read_parquet(core_file)
    plugin = pd.read_parquet(plugin_file)

    core = core.astype(
        {k: "category" for k in core.columns.intersection(categorical_cols)}
    )
    plugin = plugin.astype(
        {k: "category" for k in plugin.columns.intersection(categorical_cols)}
    )
    logger.info("Merging core and plugin data")
    # Merge data
    merged = core.merge(plugin, on=core.columns.intersection(plugin.columns).tolist())

    # Run Fixation Filter
    logger.info("Running ivt fixation filter")
    fixations = merged.groupby(["pid", "tid"]).apply(
        lambda df: get_fixations(
            df.sort_values(by="plugin_time"), ivt_optimised_config()
        )[1]
    )
    fixations = fixations.reset_index(level=2, drop=True)
    print(fixations.head(5).to_markdown())
    # filtered = pd.concat(results)
    # Save data
    logger.info(f"Saving to {str(output_file)}")
    fixations.to_parquet(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core", help="path to core file", default="./data/raw/eyetracking/core.parq"
    )
    parser.add_argument(
        "--plugin",
        help="path to plugin file",
        default="./data/raw/eyetracking/plugin.parq",
    )
    parser.add_argument(
        "--output",
        help="path to output file",
        default="./data/processed/fixations.parq",
    )
    # parser.add_argument(
    #     "--log-errors",
    #     help="Whether or not to write a file with all folders that had errors in loading data.",
    #     action="store_true",
    # )

    args = parser.parse_args()
    run_fixation_filters(args.core, args.plugin, args.output)
    # collect_data(args.root, args.output, log_errors=args.log_errors)


if __name__ == "__main__":
    main()
