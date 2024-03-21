import numpy as np
import numpy.typing as npt


def shift(x: npt.NDArray, n: int = 1, fill_value=np.nan):
    "Shifts a 1d array by `n` positions, similar to `pd.Shift`."
    if n > 0:
        return np.concatenate([np.full(n, fill_value), x[:-n]])

    return np.concatenate([x[-n:], np.full(-n, fill_value)])


def ffill(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """computes forward fill indices for an array of invalid values.

    Example:
        >>> xs = np.array([-1, -1, 20, 40, -1, -1, 30, -1, -1])
        >>> fill_idx, unfilled = ffill(xs == -1)
        >>> fill_idx
        array([0, 1, 2, 3, 3, 3, 6, 6, 6])
        >>> unfilled
        array([True, True, False, False, False, False, False, False, False])
        >>> xs[fill_idx]
        array([-1, -1, 20, 40, 40, 40, 30, 30, 30])
    Args:
        is_invalid: a boolean array of invalid values

    Returns:
        indices: an array of indices indicating the last valid value before the current value.
        unfilled: an boolean indicating which values were not filled.
    """
    indices = np.arange(len(mask), dtype=int)
    indices[mask] = -1
    np.maximum.accumulate(indices, out=indices)
    original_indices = np.arange(len(mask))
    valid_indices = np.where(indices >= 0, indices, original_indices)
    return valid_indices, indices == -1


def bfill(is_invalid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes backward fill indices for an array containing invalid values.

    Example:
        >>> xs = np.array([-1, -1, 20, 40, -1, -1, 30, -1, -1])
        >>> fill_idx, unfilled = ffill(xs == -1)
        >>> fill_idx
        array([2, 2, 2, 3, 6, 6, 6, 7, 8])
        >>> unfilled
        array([False, False, False, False, False, False, False,  True,  True])
        >>> xs[fill_idx]
        array([20, 20, 20, 40, 30, 30, 30, -1, -1])

    Args:
        is_invalid: boolean mask indicating invalid values.

    Returns:
        indices: an array of indices indicating the last valid value before the current value.
        unfilled: an boolean indicating which values were not filled.
    """
    # Backward fill
    # same approach as forward fill, but in this case,
    # invalid indices are set to mask.size + 1
    indices = np.arange(is_invalid.size)
    bfill_idx = np.where(~is_invalid, indices, is_invalid.size + 1)
    # accumulate backwards using minimum, then reverse the result
    # to get the correct order of indices
    bfill_values = np.minimum.accumulate(bfill_idx[::-1])[::-1]
    bfill_unfilled = bfill_values == is_invalid.size + 1
    bfill_values = np.where(bfill_unfilled, indices, bfill_values)
    return bfill_values, bfill_unfilled


def find_nearest_valid_neighbor(
    invalid_mask: npt.NDArray[np.bool_],
    positions: npt.NDArray[np.float_ | np.int_],
    radius: float = 4,
):
    """Given a mask representing invalid gaze values, and corresponding timestamps, finds nearest valid neighbor for each invalid gaze value.


    Args:
        invalid_mask: mask indicating invalid gaze values (true if gaze is invalid
            and false if gaze is valid)
        ts: timestamps for values in invalid_mask
        max_dt: constrain neareast neighbor search to within 'dt' of each valid value.
            Defaults to 4.

    Returns:
        idx_to_fill, fill_values_idx: indices of invalid values and corresponding indices
        of valid values to fill them with.
    """
    assert len(positions) == len(invalid_mask)

    ffill_values, ffill_unfilled = ffill(invalid_mask)
    forward_tdiffs = positions - positions[ffill_values]
    ffill_unfilled = ffill_unfilled | (forward_tdiffs > radius)

    bfill_values, bfill_unfilled = bfill(invalid_mask)
    backward_tdiffs = positions[bfill_values] - positions
    bfill_unfilled = bfill_unfilled | (backward_tdiffs > radius)

    # Combine forward and backward fills
    combined_fill_idx = np.where(
        forward_tdiffs <= backward_tdiffs, ffill_values, bfill_values
    )
    combined_fill_idx = np.where(ffill_unfilled, bfill_values, combined_fill_idx)
    combined_fill_idx = np.where(bfill_unfilled, ffill_values, combined_fill_idx)
    combined_fill_idx = np.where(
        bfill_unfilled & ffill_unfilled, np.arange(invalid_mask.size), combined_fill_idx
    )

    idx_to_fill = np.where(combined_fill_idx != np.arange(invalid_mask.size))[0]
    return idx_to_fill, combined_fill_idx[idx_to_fill]


def bin_sptp(arr: np.ndarray, bins: np.ndarray, strict=False):
    """Computes difference between max and min for each bin in a binned array.

    Args:
        arr: target array
        bins: bins for `arr`
        strict: Whether or not to enforce if `arr` is sorted. Defaults to False.

    Returns:
        an array same size as `bins` containing the difference between the max and min of each bin.
    """
    if strict:
        assert (np.diff(arr) >= 0).all(), f"Input array is not sorted."
    # Sorted ptp: Assumes arr is sorted
    bin_min_idx = np.where((bins != shift(bins)))[0]
    bin_max_idx = np.where(bins != shift(bins, -1))[0]

    return arr[bin_max_idx] - arr[bin_min_idx]


def bins_from_repeats(arr: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """Computes bin ids from a sorted array."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return (arr != shift(arr)).cumsum()
