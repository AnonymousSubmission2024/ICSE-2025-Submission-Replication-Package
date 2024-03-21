from typing import Any, Iterable
from matplotlib import pyplot as plt
from matplotlib import patches


def mark_ranges(ranges: Iterable[tuple], *, ax: plt.Axes, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for start, end in ranges:
        patch = patches.Rectangle(
            (start, min(ylim)),
            width=end - start,
            height=max(ylim) - min(ylim),
            **{
                **dict(
                    color="red",
                    fc=(1, 0, 0, 0.1),
                ),
                **kwargs,
            },
        )
        ax.add_patch(patch)


def ranges_from_fixations(start_times, durations):
    return [
        (start, start + duration) for start, duration in zip(start_times, durations)
    ]


def time_formatter(units="ms"):
    def _formatter(t, _=None):
        parts = []
        ms_t = None
        match units:
            case "ms":
                ms_t = t
            case "s":
                ms_t = t * 1e3
            case "ns":
                ms_t = t / 1e6
            case _:
                raise ValueError(
                    f'Unknown unit {units}, must be one of "ms", "ns" or "s"'
                )
        ms = ms_t % 1e3
        if ms > 0:
            parts.append(f"{int(ms):03d}")

        remaining_seconds = ms_t / 1e3
        seconds = remaining_seconds % 60
        if seconds > 0:
            parts.append(f"{int(seconds):02d}")

        remaining_minutes = remaining_seconds / 60
        minutes = remaining_minutes % 60
        if minutes > 0:
            parts.append(f"{int(minutes):02d}")
        return ":".join(reversed(parts))

    return _formatter
