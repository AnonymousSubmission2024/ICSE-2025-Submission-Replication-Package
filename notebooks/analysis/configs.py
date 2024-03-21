import fiddle as fdl
from analysis import fixation_filters
import fiddle.printing


def get_fiddle_fingerprint(cfg):
    str_repr = fiddle.printing.as_str_flattened(cfg)
    lines = [
        line for line in str_repr.splitlines() if not line.strip().endswith("<[unset]>")
    ]
    return "\n".join(lines)


def ivt_base_config(
    velocity_threshold=50, interpolation_radius=200, max_gap=200, min_duration=60
):
    return fdl.Partial(
        fixation_filters.fixation_filter,
        algorithm=fdl.Partial(
            fixation_filters.ivt, velocity_threshold=velocity_threshold
        ),
        interpolator=fdl.Partial(
            fixation_filters.nn_interpolator, radius=interpolation_radius
        ),
        transforms=[
            fdl.Partial(fixation_filters.constrain_gaps_by_time, max_gap=max_gap),
            fdl.Partial(
                fixation_filters.fixation_duration_threshold, min_duration=min_duration
            ),
            fixation_filters.remove_singletons,
        ],
    )


def ivt_optimised_config():
    return ivt_base_config(
        velocity_threshold=50,
        interpolation_radius=80,
        max_gap=80,
        min_duration=80,
    )


fixation_filter_configs = {
    "ivt_optimised": ivt_optimised_config,
    "ivt_base": ivt_base_config,
}
