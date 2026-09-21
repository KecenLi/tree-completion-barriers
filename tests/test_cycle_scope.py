"""Checks explicit constructions; no finite numerical test proves the lower bound."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cycle_scope import endpoints, explicit_psd_path, interval_margins, old_loss


def test_explicit_path_has_required_endpoints_psd_rank_and_analytic_peak():
    path = explicit_psd_path(101)
    source, target = endpoints()
    assert np.linalg.norm(path[0]-source) < 1e-12
    assert np.linalg.norm(path[-1]-target) < 1e-12
    assert np.linalg.svd(path, compute_uv=False)[:, 2:].max() < 1e-12
    assert np.linalg.eigvalsh(path).min() > -1e-12
    peak = max(old_loss(w) for w in path)
    exact = 1/49+25/242+2*(6/np.sqrt(77.)-.5)**2
    assert abs(peak-exact) < 1e-12
    assert np.linalg.norm(path, axis=(1, 2)).max() < 4+1e-12


def test_endpoint_cross_determinants_and_rational_interval_margins():
    source, target = endpoints()
    for matrix, value in ((source, -.75), (target, .75)):
        assert abs(np.linalg.det(matrix[:2, 2:])-value) < 1e-12
        assert abs(np.linalg.det(matrix[2:, :2])-value) < 1e-12
    margins = interval_margins()
    assert margins['observed_distance_bound'] < .003
    for k in ('h1', 'h2'):
        assert margins[k+'_lower'] > .31
        assert margins[k+'_upper'] < .36
    assert margins['missing_lower'] > 1.5
    assert margins['opposite_cross_det_lower'] > 1.99


def test_nearest_local_saddle_is_feasible_but_in_the_other_cross_sign_region():
    saddle = np.array([[.75,.75,.5,.5], [.75,.75,.5,.5],
                       [1.75,.5,1.,.5], [.5,1.75,.5,1.]])
    assert np.linalg.svd(saddle, compute_uv=False)[2] < 1e-12
    assert abs(old_loss(saddle)-.125) < 1e-12
    assert abs(np.linalg.det(saddle[:2, 2:])) < 1e-12
    assert abs(np.linalg.det(saddle[2:, :2])-45/16) < 1e-12
