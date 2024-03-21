import pandas as pd
import collections
import pandera as pa
import argparse
import pathlib
import multiprocessing as mp
from tqdm.auto import tqdm
from loguru import logger
from typing import Any
from returns.result import Success, Failure, Result
from analysis.data import (
    iter_experiment_dir,
    find_eyetracking_files,
    load_xml,
    ITRACE_SCHEMAS,
    load_yaml,
)


def collect_eyetracking_files(root: str | pathlib.Path):
    if not isinstance(root, pathlib.Path):
        root = pathlib.Path(root)

    datafiles = []
    for pid, tid, task_folder in iter_experiment_dir(root):
        datafiles.append(
            {
                "pid": pid,
                "tid": tid,
                "folder": task_folder,
                "eyetracking": find_eyetracking_files(task_folder),
            }
        )
    return datafiles


def load_eyetracking_xml_files(task_info: dict[str, Any]) -> dict[str, Any]:
    def try_read_xml(d: dict[str, Any]) -> Result[dict[str, Any], Any]:
        result = {}
        for filetype, path in d.items():
            try:
                result[filetype] = list(load_xml(path, xpath=".//response"))
            except Exception as e:
                return Failure(
                    f"Error while processing {filetype} at {str(path)}: {str(e)}"
                )
        return Success(result)

    return {**task_info, "eyetracking": task_info["eyetracking"].bind(try_read_xml)}


def pmap_unordered(fn, iterable):
    tasks = list(iterable)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap_unordered(fn, tasks)
        for result in tqdm(results, total=len(tasks)):
            yield result


def collect_data(root, output, invalid_data=None, log_errors=True):
    # collect all file paths
    root = pathlib.Path(root)
    all_tasks = collect_eyetracking_files(root)
    if invalid_data is not None:
        invalid_data_keys = load_yaml(invalid_data)
        invalid_data_keys = [
            (pid, tid) for pid, tids in invalid_data_keys.items() for tid in tids
        ]
        logger.info(
            f"Removing {len(invalid_data_keys)} invalid data keys from {len(all_tasks)} tasks."
        )
        all_tasks = [
            task
            for task in all_tasks
            if (task["pid"], task["tid"]) not in invalid_data_keys
        ]
        logger.info(f"Remaining tasks: {len(all_tasks)}")
    results = pmap_unordered(load_eyetracking_xml_files, all_tasks)

    dfs = collections.defaultdict(list)
    errors = []

    for result in results:
        match result["eyetracking"]:
            case Success(data):
                for filetype, data in data.items():
                    df = pd.DataFrame(data)
                    schema = ITRACE_SCHEMAS[filetype]
                    df = schema.validate(df)
                    df = df.assign(pid=result["pid"], tid=result["tid"])
                    df = df.astype({"pid": "category", "tid": "category"})
                    dfs[filetype].append(df)
            case Failure(failure_type):
                errors.append({**result, "eyetracking": failure_type})

    error_df = pd.DataFrame(errors).sort_values(by=["pid", "tid"])

    n_errors = len(error_df)
    n_total_tasks = len(all_tasks)
    logger.info(
        f"Found {n_errors} errors out of {n_total_tasks} tasks ({n_errors / n_total_tasks * 100:.2f}%)"
    )

    output = pathlib.Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    if log_errors:
        error_output_path = output / "errors.csv"
        logger.info(f"Writing errors to {str(error_output_path)}")
        error_df.to_csv(error_output_path, index=False)

    for filetype in dfs:
        df = pd.concat(dfs[filetype])
        output_file = output / (filetype + ".parq")
        logger.info(f"Writing {filetype} data to {str(output_file)}")
        # df.to_parquet(output_file, index=False)
        # df.to_csv(output_file, index=True)
        df.to_parquet(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root directory of the all experiment files")
    parser.add_argument(
        "--invalid-data",
        help="A yaml file containing pid:[tid] pairs marking data that is invalid.",
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--log-errors",
        help="Whether or not to write a file with all folders that had errors in loading data.",
        action="store_true",
    )

    args = parser.parse_args()
    collect_data(
        args.root,
        args.output,
        invalid_data=args.invalid_data,
        log_errors=args.log_errors,
    )


if __name__ == "__main__":
    main()
