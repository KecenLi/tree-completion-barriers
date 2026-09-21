"""Explicit C4 certificate checks; the continuous proofs are in the manuscript.

No optimizer is used to claim a lower bound. The strict lower bound permits
general nonsymmetric rank <= 2 paths without a matrix-norm constraint.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def old_mask():
    mask = np.eye(4, dtype=bool)
    for i in range(4):
        j = (i+1) % 4
        mask[i, j] = mask[j, i] = True
    return mask


def old_loss(w):
    values = .5*np.ones((4, 4)) + .5*np.eye(4)
    residual = (w-values)*old_mask()
    return .5*float(np.sum(residual**2))


def endpoints():
    theta = np.arccos(.5)
    def gram(angles):
        u = np.column_stack((np.cos(angles), np.sin(angles)))
        return u@u.T
    return gram(np.array([0., theta, 2*theta, theta])), gram(np.array([0., theta, 0., theta]))


def explicit_psd_path(points=501):
    if points < 2:
        raise ValueError('At least two points per segment are required')
    saddle_a, saddle_b, saddle_k = 6/7, 6/11, 6/np.sqrt(77.)
    path = []
    for sign, parameters in ((1., np.linspace(0., 1., points)),
                             (-1., np.linspace(1., 0., points)[1:])):
        for t in parameters:
            a, b, k = 1+t*(saddle_a-1), 1+t*(saddle_b-1), .5+t*(saddle_k-.5)
            v = np.array([np.sqrt(a), 0.])
            u1 = np.array([.5/np.sqrt(a), -np.sqrt(1-.25/a)])
            transverse_squared = b-k*k/a
            if transverse_squared < -1e-13:
                raise ArithmeticError('The analytically PSD interpolation became invalid')
            u3 = np.array([k/np.sqrt(a), sign*np.sqrt(max(0., transverse_squared))])
            u = np.vstack((u1, v, u3, v))
            path.append(u@u.T)
    return np.asarray(path)


def interval_margins():
    """Numerical display of rational inequalities proved in the manuscript."""
    h1_lo = (.49*.997-.503**2)/.759
    h1_hi = (.51*1.003-.497**2)/.741
    h2_lo = (.497*.997-.51*.503)/.759
    h2_hi = (.503*1.003-.49*.497)/.741
    missing_lo = (.747-.36*.503)/.36
    return dict(observed_distance_bound=np.sqrt(.000002/.49),
                h1_lower=h1_lo, h1_upper=h1_hi,
                h2_lower=h2_lo, h2_upper=h2_hi,
                missing_lower=missing_lo,
                opposite_cross_det_lower=1.5**2-.503**2)


def run():
    path = explicit_psd_path()
    source, target = endpoints()
    losses = np.array([old_loss(w) for w in path])
    singular = np.linalg.svd(path, compute_uv=False)
    saddle = np.array([[.75,.75,.5,.5], [.75,.75,.5,.5],
                       [1.75,.5,1.,.5], [.5,1.75,.5,1.]])
    row = dict(c=.5, tree_formula=.125, proven_lower=.125001,
               explicit_upper=1/49+25/242+2*(6/np.sqrt(77.)-.5)**2,
               sampled_peak=float(losses.max()), points=len(path),
               source_error=float(np.linalg.norm(path[0]-source)),
               target_error=float(np.linalg.norm(path[-1]-target)),
               largest_rank2_tail=float(singular[:, 2:].max()),
               smallest_psd_eigenvalue=float(np.linalg.eigvalsh(path).min()),
               largest_effective_norm=float(np.linalg.norm(path, axis=(1,2)).max()),
               nonsymmetric_saddle_loss=old_loss(saddle),
               nonsymmetric_saddle_alpha=float(np.linalg.det(saddle[:2, 2:])),
               nonsymmetric_saddle_beta=float(np.linalg.det(saddle[2:, :2])),
               **interval_margins())
    out = Path(__file__).resolve().parents[1]/'outputs'/'cycle_scope.csv'
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    for key, value in row.items():
        print(f'{key}: {value}')
    return row


if __name__ == '__main__':
    run()
