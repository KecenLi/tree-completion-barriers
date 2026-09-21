"""Actual noisy rank-two endpoints and certified target sets on a fixed tree.

This is the tree recurrence corollary of noisy_chain, not a new novelty claim.
Nominal observations are TreeData; actual edge observations may be negative.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import json

import numpy as np

from csv_io import write_csv
from tree_planner import TreeData, plan_tree

ROOT = Path(__file__).resolve().parents[1]


def _finite_operation(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            raise ValueError("construction exceeded finite numerical range") from exc
    return checked


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real finite numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain real finite numbers")
    return result


def _scalar(value, name, positive=False):
    result = _array(value, name)
    if result.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(result)
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _parameters(points, tol):
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer of at least 2 per segment")
    return _scalar(tol, "tol", positive=True)


def _tree_copy(tree):
    if not isinstance(tree, TreeData):
        raise ValueError("nominal tree must be a TreeData instance")
    # Revalidate mutable arrays and preserve caller-owned data.
    return TreeData(tree.a.copy(), list(tree.edges), tree.b.copy(), tree.c.copy(), root=tree.root)


def _block_sigmas(tree):
    blocks = np.empty((len(tree.edges), 2, 2))
    blocks[:, 0, 0] = tree.a[tree.parent]
    blocks[:, 1, 1] = tree.a[tree.child]
    blocks[:, 0, 1], blocks[:, 1, 0] = tree.b, tree.c
    sigma = _array(np.linalg.svd(blocks, compute_uv=False)[:, -1], "block singular values")
    if np.any(sigma <= 0):
        raise ValueError("nominal block nonsingularity is numerically unresolved")
    return blocks, sigma


def _matrix(tree, alpha, a, b, c):
    u, v = tree.factors(alpha, a=a, b=b, c=c)
    return _array(u @ v.T, "constructed matrix")


def _losses(tree, matrices):
    return _array(tree.old_losses(matrices), "old losses")


def _max_error(first, second):
    return float(np.max(np.abs(first - second), initial=0.0))


@_finite_operation
def repair_endpoint(W, tree, points=11, tol=1e-8):
    """Sample actual W -> nominal exact fiber, with quadratic loss decrease.

    Each actual 2-by-2 edge block must be strictly closer to its nominal block
    in Frobenius norm than the nominal smallest singular value.
    """
    tol = _parameters(points, tol)
    tree = _tree_copy(tree)
    W = _array(W, "endpoint")
    n = len(tree.a)
    if W.shape != (n, n):
        raise ValueError("endpoint must be a matrix of the tree dimension")
    nominal, sigma = _block_sigmas(tree)
    actual = np.stack([W[np.ix_([p, k], [p, k])] for p, k in tree.edges])
    distances = np.linalg.norm(actual - nominal, axis=(1, 2))
    ratios = _array(distances / sigma, "local noise ratios")
    if np.any(ratios >= 1):
        raise ValueError("endpoint is outside the certified local block neighborhood")

    aa = np.diag(W).copy()
    bb, cc = W[tree.parent, tree.child].copy(), W[tree.child, tree.parent].copy()
    determinants = aa[tree.parent] * aa[tree.child] - bb * cc
    if np.any(aa <= 0) or np.any(determinants <= 0):
        raise ValueError("positive actual diagonals/determinants are numerically unresolved")
    left, singular, right = np.linalg.svd(W, full_matrices=False)
    if singular[1] <= tol or np.max(singular[2:], initial=0.0) > tol:
        raise ValueError("endpoint must have numerical rank exactly two")
    U = left[:, :2] * np.sqrt(singular[:2])
    V = right[:2, :].T * np.sqrt(singular[:2])
    root = tree.root
    Q = np.column_stack((V[root] / np.sqrt(aa[root]), [-U[root, 1], U[root, 0]]))
    Ug = _array(U @ Q, "root-gauged factors")
    alpha = _array(
        (Ug[tree.parent, 0] * Ug[tree.child, 1]
         - Ug[tree.parent, 1] * Ug[tree.child, 0]) / aa[tree.parent],
        "edge orientations",
    )
    if np.any(alpha == 0):
        raise ValueError("numerically unresolved local orientation")

    values = []
    for t in np.linspace(0.0, 1.0, points):
        values.append(_matrix(tree, alpha, (1-t)*aa+t*tree.a,
                              (1-t)*bb+t*tree.b, (1-t)*cc+t*tree.c))
    values = np.asarray(values)
    reconstruction_error = _max_error(values[0], W)
    if reconstruction_error > tol:
        raise ValueError("endpoint factor reconstruction lost numerical accuracy")
    # Preserve the supplied floating-point endpoint exactly after certification.
    values[0] = W
    losses = _losses(tree, values)
    expected = (1-np.linspace(0, 1, points))**2 * losses[0]
    if _max_error(losses, expected) > tol:
        raise ValueError("radial repair lost its observed-data loss certificate")
    return values, {
        "max_local_noise_ratio": float(np.max(ratios)),
        "source_reconstruction_error": reconstruction_error,
        "endpoint_loss": float(losses[0]),
        "endpoint_rank2_tail": float(np.max(singular[2:], initial=0.0)),
    }


@_finite_operation
def plan_noisy_endpoints(source, target, tree, points=11, tol=1e-8):
    """Connect both actual rank-two endpoints; neither is replaced by a proxy."""
    tol = _parameters(points, tol)
    tree = _tree_copy(tree)
    src, srcinfo = repair_endpoint(source, tree, points, tol)
    dst, dstinfo = repair_endpoint(target, tree, points, tol)
    middle, middleinfo = plan_tree(src[-1], tree, tree.local_q(dst[-1]), points=points, tol=tol)
    if _max_error(middle[0], src[-1]) > tol or _max_error(middle[-1], dst[-1]) > tol:
        raise ValueError("middle planner did not reach the repaired actual endpoints")
    values = _array(np.concatenate((src, middle[1:], dst[-2::-1])), "planned path")
    losses = _losses(tree, values)
    predicted = max(srcinfo["endpoint_loss"], dstinfo["endpoint_loss"], middleinfo["predicted_peak"])
    if abs(float(np.max(losses)) - predicted) > tol:
        raise ValueError("noisy endpoint path lost its peak-loss certificate")
    return values, {
        "predicted_peak": predicted, "sampled_peak": float(np.max(losses)),
        "orientation_barrier": middleinfo["predicted_peak"],
        "source_old_loss": srcinfo["endpoint_loss"], "target_old_loss": dstinfo["endpoint_loss"],
        "source_error": _max_error(values[0], np.asarray(source)),
        "target_error": _max_error(values[-1], np.asarray(target)),
        "source_repair_error": srcinfo["source_reconstruction_error"],
        "target_repair_error": dstinfo["source_reconstruction_error"],
        "max_source_local_ratio": srcinfo["max_local_noise_ratio"],
        "max_target_local_ratio": dstinfo["max_local_noise_ratio"],
    }


@_finite_operation
def target_set_margin(tree, q, rho, tau):
    """Certify one orientation for every rank-two target in a tolerance set.

    rho bounds the Euclidean norm of all independent old-observation errors.
    tau is scalar or one nonnegative tolerance per tree.constraints entry.
    A false certificate does not establish infeasibility.
    """
    tree = _tree_copy(tree)
    q = _array(q, "two-hop targets")
    if q.shape != (len(tree.a)-2,):
        raise ValueError("n-2 two-hop targets required in tree.constraints order")
    rho = _scalar(rho, "rho")
    tau = _array(tau, "tau")
    if tau.ndim == 0:
        tau = np.full(q.shape, float(tau))
    if tau.shape != q.shape or np.any(tau < 0):
        raise ValueError("tau must be nonnegative and scalar or match the two-hop targets")
    _, sigma = _block_sigmas(tree)
    cross, uncertainty = [], []
    for qi, ti, (_, _, x, v, y) in zip(q, tau, tree.constraints):
        w1, w2 = tree.values[x, v], tree.values[v, y]
        cross.append(w1*w2-tree.a[v]*qi)
        uncertainty.append(rho*np.sqrt(w1*w1+w2*w2+qi*qi)+rho*rho/2
                           +(tree.a[v]+rho)*ti)
    cross = _array(cross, "nominal cross minors")
    uncertainty = _array(uncertainty, "target-set uncertainty")
    margin = _array(np.abs(cross)-uncertainty, "target-set margins")
    certified = bool(rho < np.min(sigma) and np.all(margin > 0))
    signs = None
    if certified:
        signs = np.zeros(len(tree.edges), dtype=int)
        signs[0] = 1
        for K, (e, f, _, v, _) in zip(cross, tree.constraints):
            # Incoming/outgoing coefficients can vary in magnitude under
            # noise, but their signs are fixed by the positive diagonals.
            se = 1 if tree.edges[e][0] == v else -1
            sf = 1 if tree.edges[f][0] == v else -1
            signs[f] = -int(np.sign(K))*se*sf*signs[e]
        if np.any(signs == 0):
            raise ValueError("line-graph orientation propagation was unresolved")
    return {"certified": certified, "cross_minors": cross, "uncertainty": uncertainty,
            "old_radius_threshold": float(np.min(sigma)),
            "minimum_target_margin": float(np.min(margin)), "direction_signs": signs}


@_finite_operation
def plan_noisy_target_set(source, tree, q, rho, tau, points=11, tol=1e-8):
    """Use only actual source and tolerated local targets, no full target."""
    tol = _parameters(points, tol)
    tree = _tree_copy(tree)
    certificate = target_set_margin(tree, q, rho, tau)
    if not certificate["certified"]:
        raise ValueError("target set orientation is not certified; infeasibility is not proved")
    repaired, info = repair_endpoint(source, tree, points, tol)
    middle, midinfo = plan_tree(repaired[-1], tree, q, points=points, tol=tol)
    values = _array(np.concatenate((repaired, middle[1:])), "planned target-set path")
    losses = _losses(tree, values)
    predicted = max(info["endpoint_loss"], midinfo["predicted_peak"])
    final_q_error = _max_error(tree.local_q(values[-1]), _array(q, "two-hop targets"))
    if final_q_error > tol or abs(float(np.max(losses))-predicted) > tol:
        raise ValueError("target-set path lost its observed-data certificate")
    return values, {
        "predicted_peak": predicted, "sampled_peak": float(np.max(losses)),
        "source_old_loss": info["endpoint_loss"], "target_old_loss": float(losses[-1]),
        "orientation_barrier": midinfo["predicted_peak"],
        "source_error": _max_error(values[0], np.asarray(source)),
        "source_repair_error": info["source_reconstruction_error"],
        "target_q_error": final_q_error,
        "minimum_target_margin": certificate["minimum_target_margin"],
    }


def _reproduction_case(n, shape, seed):
    """Generate endpoints directly as independently perturbed U V^T products."""
    rng = np.random.default_rng(73000+100*n+seed+sum(map(ord, shape)))
    if shape == "star":
        edges = [(0, v) for v in range(1, n)]
    elif shape == "binary":
        edges = [((v-1)//2, v) for v in range(1, n)]
    else:
        edges = [(int(rng.integers(v)), v) for v in range(1, n)]
    cosine = rng.uniform(.18, .48, n-1)
    scale = np.exp(rng.normal(0, .06, n))
    p, k = np.array(edges).T
    tree = TreeData(np.ones(n), edges, cosine*scale[p]/scale[k], cosine*scale[k]/scale[p])
    clean, noisy = [], []
    for _ in range(2):
        theta = np.zeros(n)
        for e, (parent, child) in enumerate(tree.edges):
            angle = np.arccos(np.sqrt(tree.b[e]*tree.c[e]))
            theta[child] = theta[parent]+rng.choice([-1., 1.])*angle
        Z = np.column_stack((np.cos(theta), np.sin(theta)))
        U, V = scale[:, None]*Z, Z/scale[:, None]
        clean.append(U@V.T)
        noise = .004+.003*seed
        noisy.append((U+noise*rng.normal(size=U.shape)) @
                     (V+noise*rng.normal(size=V.shape)).T)
    return tree, noisy[0], noisy[1], tree.local_q(clean[1])


def main():
    endpoint_rows, target_rows = [], []
    for n in (5, 9):
        for shape in ("star", "binary", "random"):
            for seed in (0, 1):
                tree, source, target, q = _reproduction_case(n, shape, seed)
                path, info = plan_noisy_endpoints(source, target, tree, points=7)
                meta = {"n": n, "shape": shape, "seed": seed, "points_per_segment": 7,
                        "old_count": 3*n-2, "new_count": n-2,
                        "edges": json.dumps(tree.edges), "two_hops": json.dumps(tree.constraints)}
                endpoint_rows.append({**meta, **info, "nodes": len(path),
                    "max_rank2_tail": float(np.max(np.linalg.svd(path, compute_uv=False)[:, 2:])),
                    "max_norm": float(np.max(np.linalg.norm(path, axis=(1, 2))))})
                path, info = plan_noisy_target_set(source, tree, q, .015, .025, points=7)
                target_rows.append({**meta, "rho": .015, "tau": .025, **info, "nodes": len(path),
                    "max_rank2_tail": float(np.max(np.linalg.svd(path, compute_uv=False)[:, 2:])),
                    "max_norm": float(np.max(np.linalg.norm(path, axis=(1, 2))))})
    write_csv(ROOT/"outputs"/"noisy_tree_endpoints.csv", endpoint_rows)
    write_csv(ROOT/"outputs"/"noisy_tree_target_set.csv", target_rows)
    print({"endpoint_cases": len(endpoint_rows), "target_set_cases": len(target_rows),
           "max_peak_error": max(abs(r["predicted_peak"]-r["sampled_peak"])
                                 for r in endpoint_rows+target_rows),
           "max_endpoint_error": max(max(r["source_error"], r["target_error"])
                                     for r in endpoint_rows)})


if __name__ == "__main__":
    main()
