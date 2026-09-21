import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tree_planner import TreeData,plan_tree,plan_tree_rank3


@pytest.mark.parametrize('edges',[
    [(0,1),(1,2),(2,3),(3,4)],
    [(0,1),(0,2),(0,3),(0,4)],
    [(0,1),(0,2),(2,3),(2,4)],
    [(2,0),(1,0),(4,2),(3,2)],
])
def test_two_hop_values_recover_arbitrary_nonsymmetric_completion(edges):
    tree=TreeData(np.ones(5),edges,[.1,.4,.7,.2],[.3,.2,.5,.6])
    alpha=np.array([1.4,-.8,1.1,-1.7])
    u,v=tree.factors(alpha)
    W=u@v.T
    q=tree.q_from_alpha(alpha)
    np.testing.assert_allclose(q,tree.local_q(W),atol=1e-13)
    uu,vv=tree.factors(tree.infer_alpha(q))
    np.testing.assert_allclose(uu@vv.T,W,atol=1e-12)
    assert len(tree.constraints)==len(tree.a)-2
    assert np.max(np.abs(W-W.T))>1e-3


@pytest.mark.parametrize('rank',[2,3])
@pytest.mark.parametrize('shape',['star','binary'])
def test_shared_vertex_strong_coupling_and_final_rank_two(rank,shape):
    edges=([(0,i) for i in range(1,6)] if shape=='star'
           else [((i-1)//2,i) for i in range(1,6)])
    tree=TreeData(np.ones(6),edges,[.99,.1,.01,.6,.9],[.99,.1,.01,.6,.9])
    canonical=np.sqrt(tree.D)
    source_alpha=canonical*np.array([1.,1.,1.,1.,1.])
    target_alpha=canonical*np.array([1.,-1.,1.,1.,1.])
    u,v=tree.factors(source_alpha)
    source=u@v.T
    q=tree.q_from_alpha(target_alpha)
    planner=plan_tree if rank==2 else plan_tree_rank3
    path,info=planner(source,tree,q,points=11)
    assert info['predicted_peak']==pytest.approx(.405 if rank==2 else 0.,abs=1e-13)
    assert info['sampled_peak']==pytest.approx(info['predicted_peak'],abs=1e-12)
    assert np.max(np.linalg.svd(path,compute_uv=False)[:,rank:])<1e-12
    assert np.max(np.linalg.svd(path[-1],compute_uv=False)[2:])<1e-12
    np.testing.assert_allclose(path[0],source,atol=1e-13)
    np.testing.assert_allclose(tree.local_q(path[-1]),q,atol=1e-12)
    if rank==2:
        selected=1
        p,k=tree.edges[selected]
        changed={(p,p),(k,k),(p,k),(k,p)}
        for vertex in range(len(tree.a)):
            if (vertex,vertex) not in changed:
                np.testing.assert_allclose(path[:,vertex,vertex],tree.a[vertex],atol=1e-12)
        for edge,(parent,child) in enumerate(tree.edges):
            if edge!=selected:
                np.testing.assert_allclose(path[:,parent,child],tree.b[edge],atol=1e-12)
                np.testing.assert_allclose(path[:,child,parent],tree.c[edge],atol=1e-12)
        # The stronger neighboring edge really crosses into negative determinant;
        # this tests the shared-diagonal construction outside the PSD regime.
        p,k=tree.edges[0]
        determinant=path[:,p,p]*path[:,k,k]-path[:,p,k]*path[:,k,p]
        assert determinant.min()<-.1


def test_relabeling_root_and_directed_edge_reversal_preserve_the_problem():
    a=np.array([1.,1.2,.9,1.1])
    original=TreeData(a,[(0,1),(1,2),(1,3)],[.2,.3,.4],[.3,.2,.5])
    other=TreeData(a,[(1,0),(2,1),(3,1)],[.3,.2,.5],[.2,.3,.4],root=3)
    u,v=original.factors(np.array([1.,-.8,.7]))
    W=u@v.T
    assert original.old_losses(W)<1e-26
    assert other.old_losses(W)<1e-26
    U,V=other.factors(other.infer_alpha(other.local_q(W)))
    np.testing.assert_allclose(U@V.T,W,atol=1e-12)


def test_cycles_and_singular_new_cross_minors_are_rejected():
    with pytest.raises(ValueError):
        TreeData(np.ones(4),[(0,1),(1,2),(2,0)],[.2]*3,[.2]*3)
    tree=TreeData(np.ones(3),[(0,1),(1,2)],[.2,.3],[.2,.3])
    with pytest.raises(ValueError,match='cross minor'):
        tree.infer_alpha(np.array([.06]))


@pytest.mark.parametrize('planner',[plan_tree,plan_tree_rank3])
def test_independently_generated_source_and_fixed_target_survive_root_and_edge_changes(planner):
    edges=[(0,1),(1,2),(1,3),(3,4),(3,5)]
    theta=np.array([-.55,-.2,.3,.6,-.4,.1])
    phi=theta+.15*theta**3
    U=np.linspace(.8,1.2,6)[:,None]*np.column_stack((np.cos(theta),np.sin(theta)))
    V=np.linspace(1.3,.9,6)[:,None]*np.column_stack((np.cos(phi),np.sin(phi)))
    source=U@V.T  # independent of the tree recurrence under audit
    a=np.diag(source)
    def make_tree(input_edges,root):
        return TreeData(a,input_edges,[source[p,k] for p,k in input_edges],
                        [source[k,p] for p,k in input_edges],root=root)
    original=make_tree(edges,0)
    alpha=original.infer_alpha(original.local_q(source))
    alpha*=np.array([1.,-1.,1.,1.,-1.])*np.exp(np.array([.2,-.1,.3,.1,-.2]))
    U,V=original.factors(alpha)
    target=U@V.T
    variants=[(edges,0), ([(k,p) for p,k in edges[::-1]],5),
              ([(3,5),(2,1),(3,1),(0,1),(4,3)],2)]
    barriers=[]
    for input_edges,root in variants:
        tree=make_tree(input_edges,root)
        path,info=planner(source,tree,tree.local_q(target),points=7)
        np.testing.assert_allclose(path[0],source,atol=1e-11)
        np.testing.assert_allclose(path[-1],target,atol=1e-10)
        assert np.max(np.linalg.svd(path[-1],compute_uv=False)[2:])<1e-11
        barriers.append(info['predicted_peak'])
    np.testing.assert_allclose(barriers,barriers[0],atol=1e-12)


@pytest.mark.parametrize('planner',[plan_tree,plan_tree_rank3])
@pytest.mark.parametrize('bad_value',[np.nan,np.inf])
def test_nonfinite_source_cannot_hide_in_an_unobserved_non_two_hop_entry(planner,bad_value):
    tree=TreeData(np.ones(5),[(0,1),(1,2),(2,3),(3,4)],[.2]*4,[.2]*4)
    U,V=tree.factors(np.ones(4))
    source=U@V.T
    q=tree.local_q(source)
    source[0,4]=bad_value
    with pytest.raises(ValueError,match='finite matrix'):
        planner(source,tree,q)


@pytest.mark.parametrize('planner',[plan_tree,plan_tree_rank3])
@pytest.mark.parametrize('kwargs',[{'points':0},{'points':1},{'points':2.5},
                                  {'tol':np.nan},{'tol':np.inf},{'tol':0.}])
def test_invalid_sampling_or_tolerance_is_rejected_before_certification(planner,kwargs):
    tree=TreeData(np.ones(3),[(0,1),(1,2)],[.2,.3],[.2,.3])
    U,V=tree.factors(np.ones(2))
    source=U@V.T
    with pytest.raises(ValueError):
        planner(source,tree,tree.local_q(source),**kwargs)


def test_noninteger_matrix_indices_and_overflowed_determinants_are_rejected():
    for edges,root in [([(0.,1),(1,2)],0), ([(0,1),(1,2)],1.5)]:
        with pytest.raises(ValueError,match='integer'):
            TreeData(np.ones(3),edges,[.2,.3],[.2,.3],root=root)
    with pytest.raises(ValueError,match='finite and positive'):
        TreeData(np.full(3,1e200),[(0,1),(1,2)],[.2,.3],[.2,.3])
