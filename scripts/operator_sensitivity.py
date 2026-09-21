"""Vanishing operator perturbations can destroy an unbounded-norm barrier."""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from csv_io import write_csv

ROOT=Path(__file__).resolve().parents[1]
POSITIONS=((0,0),(1,1),(2,2),(0,1),(1,0),(1,2),(2,1))


def operator_matrix(c,eta):
    if not 0<c<1 or eta<0:raise ValueError('0<c<1 and eta>=0 required')
    A=np.zeros((7,9))
    for i,(r,col) in enumerate(POSITIONS):A[i,3*r+col]=1
    correction=np.array([0,eta/c,0,eta,eta,eta,eta])
    A[:,2]-=correction
    A[:,6]+=correction
    return A


def path(c,eta,points=101):
    if not 0<c<1 or eta<=0 or points<3:
        raise ValueError('0<c<1, eta>0 and points>=3 required')
    zstar=-c/(2*eta)
    def state(z,m):
        d=1+2*eta*z/c
        b=c*d
        return np.array([[1,b,m+z],[b,d,b],[m-z,b,1.]])
    def branch(z,sign):
        d=1+2*eta*z/c
        return c*c*d+sign*math.sqrt((1-c*c*d)**2+z*z)
    values=[state(z,branch(z,1)) for z in np.linspace(0,zstar,points)]
    values += [state(zstar,m) for m in np.linspace(branch(zstar,1),branch(zstar,-1),points)[1:]]
    values += [state(z,branch(z,-1)) for z in np.linspace(zstar,0,points)[1:]]
    return np.asarray(values)


def symbolic():
    c,eta,z,m=sp.symbols('c eta z m',real=True,nonzero=True)
    d=1+2*eta*z/c
    b=c*d
    W=sp.Matrix([[1,b,m+z],[b,d,b],[m-z,b,1]])
    determinant=sp.factor(W.det())
    predicted=d*(1-m*m+z*z+2*c*c*d*(m-1))
    assert sp.simplify(determinant-predicted)==0
    assert W.subs(z,-c/(2*eta)).row(1)==sp.zeros(1,3)
    target=sp.Matrix([1,1,1,c,c,c,c])
    correction=[0,eta/c,0,eta,eta,eta,eta]
    observations=sp.Matrix([W[r,col]-correction[i]*(W[0,2]-W[2,0])
                            for i,(r,col) in enumerate(POSITIONS)])
    assert sp.simplify(observations-target)==sp.zeros(7,1)
    return {'determinant':str(determinant),'exact_observations':True,
            'zero_middle_row_at_bridge':True}


def main():
    rows=[]
    for c in (.2,.5,.8):
        target=np.array([1,1,1,c,c,c,c])
        A0=operator_matrix(c,0)
        for eta in (.1,.03,.01,.003,.001):
            W=path(c,eta)
            residual=W.reshape(len(W),9)@operator_matrix(c,eta).T-target
            original=W.reshape(len(W),9)@A0.T-target
            norms=np.linalg.norm(W,axis=(1,2))
            tail=np.linalg.svd(W,compute_uv=False)[:,-1]
            rows.append({'c':c,'eta':eta,'original_exact_barrier':.5*(1-c)**2,
                'operator_distance':eta*math.sqrt(2/c**2+8),
                'perturbed_peak_loss':.5*np.max(np.sum(residual**2,axis=1)),
                'original_peak_loss_of_constructed_path':.5*np.max(np.sum(original**2,axis=1)),
                'max_rank_tail':float(np.max(tail)),'max_norm':float(np.max(norms)),
                'normalized_resource':float(eta*np.max(norms)/c),
                'nodes':len(W)})
    write_csv(ROOT/'outputs'/'operator_sensitivity.csv',rows)
    (ROOT/'outputs'/'operator_sensitivity_symbolic.json').write_text(json.dumps(symbolic(),indent=2))
    print({'cases':len(rows),'max_perturbed_loss':max(r['perturbed_peak_loss'] for r in rows),
           'max_rank_tail':max(r['max_rank_tail'] for r in rows)})


if __name__=='__main__':main()
