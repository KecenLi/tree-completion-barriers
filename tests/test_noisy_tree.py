import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from noisy_tree import repair_endpoint, plan_noisy_endpoints, target_set_margin, plan_noisy_target_set
from tree_planner import TreeData


def independent_case(shape="binary", seed=0, n=7):
    """Generate U,V directly; never use the planner's alpha reconstruction."""
    rng = np.random.default_rng(4300+seed)
    if shape == "star":
        edges = [(0, v) for v in range(1, n)]
    elif shape == "binary":
        edges = [((v-1)//2, v) for v in range(1, n)]
    else:
        edges = [(int(rng.integers(v)), v) for v in range(1, n)]
    weight = rng.uniform(.15, .5, n-1)
    scales = np.exp(rng.normal(0, .08, n))
    p, v = np.array(edges).T
    tree = TreeData(np.ones(n), edges, weight*scales[p]/scales[v], weight*scales[v]/scales[p])
    clean, actual = [], []
    for end in (0, 1):
        theta = np.zeros(n)
        for e, (p, v) in enumerate(tree.edges):
            sign = 1 if end == 0 or e % 2 == 0 else -1
            theta[v] = theta[p]+sign*np.arccos(np.sqrt(tree.b[e]*tree.c[e]))
        Z = np.column_stack((np.cos(theta), np.sin(theta)))
        U, V = scales[:, None]*Z, Z/scales[:, None]
        clean.append(U@V.T)
        actual.append((U+.005*rng.normal(size=U.shape)) @
                      (V+.005*rng.normal(size=V.shape)).T)
    return tree, clean, actual


def observed_mask(tree):
    mask = np.eye(len(tree.a), dtype=bool)
    for p, v in tree.edges:
        mask[p, v] = mask[v, p] = True
    return mask


def nominal_matrix(tree):
    W = np.diag(tree.a)
    for (p, v), b, c in zip(tree.edges, tree.b, tree.c):
        W[p, v], W[v, p] = b, c
    return W


def cross_orientations(W, tree):
    base = tree.edges[0]
    return np.array([np.sign(np.linalg.det(W[np.ix_(base, edge)])) for edge in tree.edges])


@pytest.mark.parametrize("shape", ["star", "binary", "random"])
@pytest.mark.parametrize("seed", [0, 1])
def test_independent_nonsymmetric_endpoints_and_continuous_peak(shape, seed):
    tree, _, (source, target) = independent_case(shape, seed)
    assert np.linalg.norm(source-source.T) > .01
    path, info = plan_noisy_endpoints(source, target, tree, points=5)
    np.testing.assert_array_equal(path[0], source)
    np.testing.assert_array_equal(path[-1], target)
    assert np.max(np.linalg.svd(path, compute_uv=False)[:, 2:]) < 1e-11
    nominal = nominal_matrix(tree)
    loss = .5*np.sum((path[:, observed_mask(tree)]-nominal[observed_mask(tree)])**2, axis=1)
    assert np.max(loss) == pytest.approx(info["predicted_peak"], abs=1e-12)
    # Independent mixed-minor labels determine which edge sets can be flipped.
    labels = cross_orientations(source, tree)*cross_orientations(target, tree)
    h = np.array([.5*np.linalg.svd(nominal[np.ix_(e, e)], compute_uv=False)[-1]**2
                  for e in tree.edges])
    expected_H = min(np.max(h[labels > 0], initial=0), np.max(h[labels < 0], initial=0))
    assert info["orientation_barrier"] == pytest.approx(expected_H, abs=1e-12)
    assert info["predicted_peak"] == pytest.approx(max(loss[0], loss[-1], expected_H), abs=1e-12)


def test_shared_diagonal_is_counted_once_and_all_observations_repair_together():
    n = 8
    c = .3
    tree = TreeData(np.ones(n), [(0, v) for v in range(1, n)],
                    np.full(n-1, c), np.full(n-1, c))
    Z = np.tile([c, np.sqrt(1-c*c)], (n, 1))
    Z[0] = [1, 0]
    endpoint = 1.07*(Z@Z.T)
    path, info = repair_endpoint(endpoint, tree, points=9)
    expected_loss = .5*.07**2*(n+2*(n-1)*c*c)
    assert info["endpoint_loss"] == pytest.approx(expected_loss, abs=1e-14)
    mask = observed_mask(tree)
    target = nominal_matrix(tree)
    for t, W in zip(np.linspace(0, 1, 9), path):
        np.testing.assert_allclose(W[mask], ((1-t)*endpoint+t*target)[mask], atol=1e-13, rtol=0)
    np.testing.assert_allclose(tree.old_losses(path),
                               (1-np.linspace(0, 1, 9))**2*expected_loss, atol=1e-14, rtol=0)


def test_same_component_can_be_limited_entirely_by_actual_endpoint_loss():
    tree, clean, actual = independent_case("star", 4)
    source, target = actual[0], 1.08*clean[0]
    path, info = plan_noisy_endpoints(source, target, tree, points=5)
    assert info["orientation_barrier"] == 0
    assert info["target_old_loss"] > info["source_old_loss"] > 0
    assert info["sampled_peak"] == pytest.approx(info["target_old_loss"], abs=1e-12)
    np.testing.assert_array_equal(path[-1], target)


def test_safe_neighborhood_can_include_negative_actual_directed_entry():
    c = np.array([.01, .2, .4])
    tree = TreeData(np.ones(4), [(0, 1), (0, 2), (0, 3)], c, c)
    U = np.column_stack((np.r_[1., c], np.r_[0., np.sqrt(1-c*c)]))
    V = U.copy()
    V[1, 0] = -.02
    W = U@V.T
    assert W[0, 1] < 0
    path, info = repair_endpoint(W, tree, points=7)
    assert info["max_local_noise_ratio"] < 1
    np.testing.assert_array_equal(path[0], W)
    np.testing.assert_allclose(path[-1][observed_mask(tree)],
                               nominal_matrix(tree)[observed_mask(tree)], atol=1e-14)
    assert np.max(np.linalg.svd(path, compute_uv=False)[:, 2:]) < 1e-12


def test_rerooting_and_reversing_input_edges_preserve_problem_and_peak():
    tree, _, (source, target) = independent_case("random", 10, n=9)
    alternate = TreeData(tree.a, [(v, p) for p, v in reversed(tree.edges)],
                          tree.c[::-1], tree.b[::-1], root=8)
    first, info1 = plan_noisy_endpoints(source, target, tree, points=5)
    second, info2 = plan_noisy_endpoints(source, target, alternate, points=5)
    assert info1["predicted_peak"] == pytest.approx(info2["predicted_peak"], abs=1e-12)
    np.testing.assert_array_equal(second[0], first[0])
    np.testing.assert_array_equal(second[-1], first[-1])


@pytest.mark.parametrize("shape", ["star", "binary", "random"])
def test_source_plus_local_q_and_certified_target_orientations(shape):
    tree, clean, actual = independent_case(shape, 3)
    q = tree.local_q(clean[1])
    rho, tau = .09, .08
    certificate = target_set_margin(tree, q, rho, tau)
    assert certificate["certified"]
    for target in (clean[1], actual[1]):
        mask = observed_mask(tree)
        assert np.linalg.norm((target-nominal_matrix(tree))[mask]) <= rho
        assert np.max(np.abs(tree.local_q(target)-q)) <= tau
        np.testing.assert_array_equal(cross_orientations(target, tree), certificate["direction_signs"])
    path, info = plan_noisy_target_set(actual[0], tree, q, rho, tau, points=5)
    np.testing.assert_array_equal(path[0], actual[0])
    np.testing.assert_allclose(tree.local_q(path[-1]), q, atol=1e-12, rtol=0)
    assert tree.old_losses(path[-1]) < 1e-24
    assert np.max(np.linalg.svd(path, compute_uv=False)[:, 2:]) < 1e-11
    assert info["sampled_peak"] == pytest.approx(info["predicted_peak"], abs=1e-12)
    # A local q task fixes a unique nominal completion in this tree family.
    np.testing.assert_allclose(path[-1], clean[1], atol=1e-11, rtol=0)


def test_insufficient_target_margins_are_explicitly_not_certified():
    tree, clean, actual = independent_case()
    q = tree.local_q(clean[1])
    threshold = target_set_margin(tree, q, 0., 0.)["old_radius_threshold"]
    assert not target_set_margin(tree, q, threshold, 0.)["certified"]
    assert not target_set_margin(tree, q, 0., 100.)["certified"]
    e, f, x, v, y = tree.constraints[0]
    central = q.copy()
    central[0] = tree.values[x, v]*tree.values[v, y]/tree.a[v]
    assert not target_set_margin(tree, central, 0., 0.)["certified"]
    with pytest.raises(ValueError, match="not certified"):
        plan_noisy_target_set(actual[0], tree, q, 0., 100.)


def test_unsafe_rank3_and_unobserved_nan_endpoints_are_rejected():
    tree, clean, _ = independent_case()
    with pytest.raises(ValueError, match="outside"):
        repair_endpoint(3*clean[0], tree)
    with pytest.raises(ValueError, match="rank exactly two"):
        repair_endpoint(clean[0]+.001*np.eye(len(tree.a)), tree)
    invalid = clean[0].copy()
    x, y = np.argwhere(~observed_mask(tree))[0]
    invalid[x, y] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repair_endpoint(invalid, tree)


@pytest.mark.parametrize("points", [True, 1, 2.5, np.nan])
def test_invalid_sample_counts(points):
    tree, clean, _ = independent_case()
    with pytest.raises(ValueError, match="points"):
        repair_endpoint(clean[0], tree, points=points)


@pytest.mark.parametrize("tol", [True, 0, -1, np.nan, np.inf])
def test_invalid_numerical_tolerances(tol):
    tree, clean, _ = independent_case()
    with pytest.raises(ValueError, match="tol"):
        repair_endpoint(clean[0], tree, tol=tol)


@pytest.mark.parametrize("field,value", [
    ("rho", -1), ("rho", np.inf), ("rho", np.nan),
    ("tau", -1), ("tau", np.inf), ("tau", np.nan),
    ("q", np.nan), ("q", np.inf),
])
def test_nonfinite_or_negative_target_set_data(field, value):
    tree, clean, _ = independent_case()
    kwargs = {"q": tree.local_q(clean[1]), "rho": .01, "tau": .01}
    kwargs[field] = np.full(len(tree.a)-2, value) if field == "q" else value
    with pytest.raises(ValueError):
        target_set_margin(tree, **kwargs)


def test_finite_inputs_with_overflowing_certificate_fail_closed():
    tree, _, _ = independent_case()
    with pytest.raises(ValueError, match="finite numerical range"):
        target_set_margin(tree, np.full(len(tree.a)-2, 1e308), .01, .01)


def test_wrong_target_shapes_and_mutated_nominal_data_are_rejected():
    tree, clean, _ = independent_case()
    q = tree.local_q(clean[1])
    with pytest.raises(ValueError):
        target_set_margin(tree, q[:-1], .01, .01)
    with pytest.raises(ValueError):
        target_set_margin(tree, q, .01, np.ones((len(q), 1)))
    tree.a[0] = np.nan
    with pytest.raises(ValueError):
        repair_endpoint(clean[0], tree)
