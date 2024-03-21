import dataclasses
import datetime
import pathlib
import re
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import pandera as pa
from returns.result import Failure, Result, Success, safe


def iter_experiment_dir(root: pathlib.Path | str):
    """Iterates over a directory that contains multiple experiment data of form `root/[pid]/[tid]`."""
    if not isinstance(root, pathlib.Path):
        root = pathlib.Path(root)

    for participant_folder in root.iterdir():
        if not participant_folder.is_dir():
            continue

        for task_folder in participant_folder.iterdir():
            if not task_folder.is_dir():
                continue

            pid = participant_folder.name
            tid = task_folder.name

            yield pid, tid, task_folder


# TODO: delete this, only kept for legacy purposes
def iter_experiment_folder(path: str):
    root = pathlib.Path(path)
    for pid, tid, task_folder in iter_experiment_dir(root):
        fixations_json = task_folder / "processed-new" / "fixations.json"

        if fixations_json.exists():
            yield pid, tid, fixations_json


@dataclasses.dataclass(frozen=True)
class Fixation:
    duration: int
    n: int
    system_time: datetime.datetime
    target: str
    source_file_line: int
    source_file_col: int
    left_pupil_diameter: float
    right_pupil_diameter: float

    @staticmethod
    def from_record(**kwargs) -> "Fixation":
        fields = set(field.name for field in dataclasses.fields(Fixation))
        kwargs = {k: v for k, v in kwargs.items() if k in fields}

        return Fixation(**kwargs)


def load_yaml(path: str):
    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(path) as f:
        return load(f, Loader=Loader)


def load_file_contents(path: str):
    with open(path) as f:
        return f.read()


def load_xml(filepath: str, xpath: str = "./"):
    """Faster XML loading than with pandas.read_xml."""
    tree = ElementTree.parse(filepath)
    for el in tree.iterfind(xpath):
        yield dict(el.items())


def reglob(path: pathlib.Path, pattern: str, glob_pattern: str = "**/*"):
    """Runs glob at the given path, and filters results with the given regex pattern."""
    for file in path.glob(glob_pattern):
        if re.match(pattern, str(file.name)):
            yield file


def find_eyetracking_files(path: pathlib.Path) -> Result[dict[str, pathlib.Path], str]:
    """Given an experiment folder, finds all itrace files and returns a dict of (type, filepath)."""
    core_files = list(reglob(path, "itrace_core-[0-9]+.xml$"))

    if len(core_files) == 0:
        return Failure("Missing itrace_core file")
    elif len(core_files) > 1:
        return Failure(
            f"Multiple itrace_core files found: {list(map(str, core_files))}"
        )

    core_file = core_files[0]
    plugin_file = next(re.finditer("itrace_core-([0-9]+).xml$", str(core_file.name)))

    plugin_file = path / f"gazeOutput-{plugin_file.group(1)}.xml"
    if not plugin_file.exists():
        return Failure(f"Missing plugin file {plugin_file} for core file {core_file}")

    return Success({"core": core_file, "plugin": plugin_file})


def set_dtypes(df, d: dict[tuple | str, Any]) -> pd.DataFrame:
    """Utility to set dtypes with a dict wherein each dict key can be one or more column names."""
    dtypes = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            for col in k:
                dtypes[col] = v
        else:
            dtypes[k] = v
    return df.astype(dtypes)


# Data schemas for eyetracking files
common_schema = {
    "x": pa.Column(pa.Float),
    "y": pa.Column(pa.Float),
    "event_id": pa.Column(pa.Int),
}

ITRACE_SCHEMAS = {
    "core": pa.DataFrameSchema(
        {
            **common_schema,
            "core_time": pa.Column(pa.Int),
            "tracker_time": pa.Column(pa.Int),
            "left_x": pa.Column(pa.Float),
            "right_x": pa.Column(pa.Float),
            "left_y": pa.Column(pa.Float),
            "right_y": pa.Column(pa.Float),
            "left_pupil_diameter": pa.Column(pa.Float),
            "right_pupil_diameter": pa.Column(pa.Float),
            "left_validation": pa.Column(pa.Int),
            "right_validation": pa.Column(pa.Int),
            "user_left_x": pa.Column(pa.Float),
            "user_left_y": pa.Column(pa.Float),
            "user_left_z": pa.Column(pa.Float),
            "user_right_x": pa.Column(pa.Float),
            "user_right_y": pa.Column(pa.Float),
            "user_right_z": pa.Column(pa.Float),
        },
        coerce=True,
    ),
    "plugin": pa.DataFrameSchema(
        {
            "x": pa.Column(pa.Float),
            "y": pa.Column(pa.Float),
            "event_id": pa.Column(pa.Int),
            "plugin_time": pa.Column(pa.Float),
            "gaze_target": pa.Column(pa.Category),
            "gaze_target_type": pa.Column(pa.Category),
            "source_file_path": pa.Column(pa.Category),
            "source_file_line": pa.Column(pa.Int),
            "source_file_col": pa.Column(pa.Int),
            "editor_line_height": pa.Column(pa.Float, nullable=True),
            "editor_font_height": pa.Column(pa.Float, nullable=True),
            # "editor_line_base_x": pa.Column(pa.Float, nullable=True),
            # "editor_line_base_y": pa.Column(pa.Float, nullable=True),
        },
        coerce=True,
    ),
}
