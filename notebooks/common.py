import pandas as pd
from analysis.data import set_dtypes, load_yaml
import intervaltree
from minineedle import needle, core
import pyarrow.parquet as pq


#def load_correctness_data(path="../../data/correctness.csv") -> pd.DataFrame:
def load_correctness_data(path="../data/correctness.csv") -> pd.DataFrame:    
    correctness = pd.read_csv(path)
    print(correctness.columns)

    correctness = correctness[["pid", "tid", "name", "Final Correctness"]].rename(
        {"Final Correctness": "correct", "name": "snippet"}, axis=1
    )
    correctness = correctness[
        (correctness.correct == "1") | (correctness.correct == "0")
    ].copy()
    correctness["correct"] = correctness["correct"].astype(int)
    return set_dtypes(
        correctness, {("pid", "tid", "snippet"): "category", "correct": "bool"}
    )


def load_fixation_data(path="../data/processed/fixations.parq"):
    fixations = pd.read_parquet(path)
    valid_snippets = [
        "graph_utils_c.cc",
        "rectangle-with-bug.cpp",
        "calculation-with-bug.cpp",
        "insertion-sort-with-bug.cpp",
        "number-checker-with-bug.cpp",
        "money-class-with-bug.cpp",
        "numbers_c.cc",
    ]

    fixations = fixations[fixations["gaze_target"].isin(valid_snippets)]

    # fixations = fixations.astype({k: "category" for k in categorical_cols})
    fixations = fixations.astype(
        {
            "source_file_line": int,
            "source_file_col": int,
            "n": int,
            # "plugin_time": "datetime64[ms]",
        }
    )

    return fixations


def get_fixation_intervals(fixations, indexed=False):
    seq = zip(fixations["start_time"], fixations["start_time"] + fixations["duration"])
    if indexed:
        return [(a, b, i) for i, (a, b) in enumerate(seq)]
    return list(seq)


def normalized_alignment_score(a, b, scoring_matrix=None):
    if scoring_matrix is None:
        scoring_matrix = core.ScoreMatrix(1, -1, -1)

    alignment = needle.NeedlemanWunsch(a, b)
    alignment.change_matrix(scoring_matrix)
    alignment.align()

    score = alignment.get_score()
    n = max(len(a), len(b))

    max_scoring_matrix = max(
        scoring_matrix.gap, scoring_matrix.match, scoring_matrix.miss
    )
    return score / (max_scoring_matrix * n)


def interval_subsumes(a, b) -> bool:
    "Checks if A subsumes B."
    return a[0] <= b[0] and a[1] >= b[1]


def compute_overlaps(old_fixation_intervals, new_fixation_intervals):
    tree = intervaltree.IntervalTree.from_tuples(old_fixation_intervals)

    splits = 0
    missing = 0
    overlaps = 0
    for new_fixation in new_fixation_intervals:
        overlapping = list(tree.overlap(*new_fixation))
        if not overlapping:
            missing += 1
        elif len(overlapping) == 1:
            if interval_subsumes(overlapping[0], new_fixation):
                overlaps += 1
            else:
                splits += 1
        elif len(overlapping) > 1:
            splits += 1
        else:
            raise Exception()

    results = {"splits": splits, "missing": missing, "overlaps": overlaps}
    results = {k: v / len(new_fixation_intervals) for k, v in results.items()}
    return results


def fixation_diff(
    old_fixations, new_fixations, position=["source_file_line", "source_file_col"]
):
    old_fixation_intervals = get_fixation_intervals(old_fixations)
    new_fixation_intervals = get_fixation_intervals(new_fixations)

    overlaps = compute_overlaps(new_fixation_intervals, old_fixation_intervals)
    old_position_sequence = [
        (a, b) for a, b in zip(*old_fixations[position].to_numpy().T)
    ]
    new_position_sequence = [
        (a, b) for a, b in zip(*new_fixations[position].to_numpy().T)
    ]

    position_diff = normalized_alignment_score(
        old_position_sequence, new_position_sequence
    )

    return {**overlaps, "sequence_diff": position_diff}


def load_experiment_data(filename, pid=None, tid=None):
    filters = []
    if pid is not None:
        filters.append(["pid", "==", pid])
    if tid is not None:
        filters.append(["tid", "==", tid])

    table = pq.read_table(filename, filters=filters if filters else None)
    return table.to_pandas()


def load_raw_gaze_data(pid=None, tid=None):
    plugin = load_experiment_data(
        "../data/raw/eyetracking/plugin.parq", pid=pid, tid=tid
    )
    core = load_experiment_data("../data/raw/eyetracking/core.parq", pid=pid, tid=tid)

    plugin = plugin.set_index(["pid", "tid", "event_id"])
    core = core.set_index(["pid", "tid", "event_id"])
    return plugin.merge(core)


def remove_invalid_eyetracking_data(df: pd.DataFrame, invalid_eyetracking_data: dict):
    for pid, tids in invalid_eyetracking_data.items():
        for tid in tids:
            df = df.drop((pid, tid), errors="ignore")
    return df


def load_fixation_data_for_diff(pid=None, tid=None):
    old_fixations = load_experiment_data(
        "../data/processed/fixations.parq", pid=pid, tid=tid
    )
    old_fixations = old_fixations.set_index(["pid", "tid"])
    old_fixations = old_fixations.rename({"plugin_time": "start_time"}, axis=1)

    new_fixations = load_experiment_data(
        "../data/processed/fixations-new.parq", pid=pid, tid=tid
    )
    new_fixations = new_fixations.reset_index(level=2)

    fixed_fixations = load_experiment_data(
        "../data/processed/fixations-fixed.parq", pid=pid, tid=tid
    )
    fixed_fixations = fixed_fixations.reset_index(level=2)

    invalid_eyetracking_data = load_yaml("../data/raw/invalid_eyetracking_data.yaml")

    old_fixations = remove_invalid_eyetracking_data(
        old_fixations, invalid_eyetracking_data
    )
    new_fixations = remove_invalid_eyetracking_data(
        new_fixations, invalid_eyetracking_data
    )
    fixed_fixations = remove_invalid_eyetracking_data(
        fixed_fixations, invalid_eyetracking_data
    )

    cols = ["start_time", "duration", "source_file_line", "source_file_col", "n"]
    old_fixations = old_fixations[cols]
    new_fixations = new_fixations[cols + ["interpolated"]]
    fixed_fixations = fixed_fixations[cols + ["interpolated", "position_interpolated"]]

    return old_fixations, new_fixations, fixed_fixations


def mp_worker_fixation_diff(keys):
    results = []
    for key in keys:
        old_fixations, new_fixations, fixed_fixations = load_fixation_data_for_diff(
            *key
        )
        result_key = {"pid": key[0], "tid": key[1]}
        results.append(
            {
                "type": "old_vs_new",
                **result_key,
                **fixation_diff(old_fixations, new_fixations),
            }
        )
        results.append(
            {
                "type": "old_vs_fixed",
                **result_key,
                **fixation_diff(old_fixations, fixed_fixations),
            }
        )

    return results
