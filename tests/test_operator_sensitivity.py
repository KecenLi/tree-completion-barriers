import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from operator_sensitivity import operator_matrix,path,symbolic


def test_symbolic_identity_and_exact_observations():
    assert symbolic()['exact_observations']


@pytest.mark.parametrize('c',[.2,.5,.8])
def test_operator_distance_and_unchanged_exact_endpoints(c):
    eta=.003
    A=operator_matrix(c,eta)
    A0=operator_matrix(c,0)
    assert np.linalg.norm(A-A0,2)==pytest.approx(eta*math.sqrt(2/c**2+8),rel=1e-13)
    source=np.array([[1,c,1],[c,1,c],[1,c,1.]])
    q=2*c*c-1
    target=np.array([[1,c,q],[c,1,c],[q,c,1.]])
    for W in (source,target):
        # The symbolic cancellation is exact; BLAS dot products may retain
        # sub-ulp roundoff even when the two input coordinates are identical.
        np.testing.assert_allclose((A-A0)@W.ravel(),np.zeros(7),rtol=0,atol=1e-17)
    values=path(c,eta)
    np.testing.assert_allclose(values[0],source,atol=2e-16)
    np.testing.assert_allclose(values[-1],target,atol=2e-16)


@pytest.mark.parametrize('c',[.2,.5,.8])
@pytest.mark.parametrize('eta',[.03,.001])
def test_finite_continuous_construction_is_zero_loss_rank_two(c,eta):
    W=path(c,eta,points=51)
    residual=W.reshape(len(W),9)@operator_matrix(c,eta).T-np.array([1,1,1,c,c,c,c])
    assert np.max(np.abs(residual))<2e-13
    assert np.max(np.linalg.svd(W,compute_uv=False)[:,-1])<3e-12
    # The whole bridge has its middle row/column identically zero, which
    # certifies every intermediate matrix rather than just its sampled nodes.
    np.testing.assert_allclose(W[50:101,1,:],0,atol=1e-15)
    np.testing.assert_allclose(W[50:101,:,1],0,atol=1e-15)
    M=np.max(np.linalg.norm(W,axis=(1,2)))
    necessary=(1-c)/(eta*math.sqrt(2/c**2+8))
    assert M>=necessary
