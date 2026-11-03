import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import pick_patch


def test_snap_patch_snaps_size_and_keeps_center():
    assert pick_patch.snap_patch(10, 20, 74, 90, 265, 370) == (10, 19, 64, 72)


def test_snap_patch_grows_a_tiny_drag_to_the_minimum():
    x, y, w, h = pick_patch.snap_patch(100, 100, 105, 103, 265, 370)
    assert (w, h) == (32, 32)
    assert abs((x + w / 2) - 102.5) <= 1
    assert abs((y + h / 2) - 101.5) <= 1


def test_snap_patch_clamps_a_drag_outside_the_image():
    x, y, w, h = pick_patch.snap_patch(-50, -50, 500, 400, 265, 370)
    assert w % 8 == 0 and h % 8 == 0
    assert 0 <= x and x + w <= 370
    assert 0 <= y and y + h <= 265


def test_snap_patch_rejects_an_image_smaller_than_the_grid():
    with pytest.raises(ValueError):
        pick_patch.snap_patch(0, 0, 4, 4, 4, 4)


def test_hu_conversion_round_trips_inside_the_window():
    hu = np.array([[-2000, -1000, 0, 2000, 3000]], dtype=np.float32)
    unit = pick_patch.hu_to_unit(hu, (-1000, 2000))
    back = pick_patch.unit_to_hu(unit, (-1000, 2000))
    assert np.allclose(back, np.clip(hu, -1000, 2000), atol=1e-4)
