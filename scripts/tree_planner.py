"""Positive directed tree observations: local-data continuous rank-2 planner.

Implements the tree construction in the accompanying manuscript.
Edge data are given in the input tuple's row/column direction.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math

import networkx as nx
import numpy as np

from csv_io import write_csv

ROOT=Path(__file__).resolve().parents[1]


def _checked_matrix(u,v):
    with np.errstate(over='ignore',invalid='ignore'):
        matrix=u@v.T
    if not np.all(np.isfinite(matrix)):
        raise ValueError('tree construction exceeded finite numerical range')
    return matrix


def _validated_source(source,tree,points,tol):
    if isinstance(points,(bool,np.bool_)) or not isinstance(points,(int,np.integer)) or points<2:
        raise ValueError('points must be an integer of at least 2 per segment')
    if not np.isscalar(tol) or not np.isfinite(tol) or tol<=0:
        raise ValueError('tol must be finite and strictly positive')
    source=np.asarray(source,dtype=float)
    if source.shape!=(len(tree.a),len(tree.a)) or not np.all(np.isfinite(source)):
        raise ValueError('source must be a finite matrix of the tree dimension')
    if np.sqrt(2*tree.old_losses(source))>tol:
        raise ValueError('source does not fit the nominal old observations')
    return source


@dataclass
class TreeData:
    a: np.ndarray
    edges: list
    b: np.ndarray
    c: np.ndarray
    root: int=0

    def __post_init__(self):
        self.a,self.b,self.c=(np.asarray(x,dtype=float) for x in (self.a,self.b,self.c))
        n=len(self.a)
        if self.a.ndim!=1 or n<3 or len(self.edges)!=n-1 or self.b.shape!=(n-1,) or self.c.shape!=(n-1,):
            raise ValueError('n >= 3, n-1 edges and matching directed data required')
        if any(not np.all(np.isfinite(x)) or np.any(x<=0) for x in (self.a,self.b,self.c)):
            raise ValueError('nominal observations must be finite and positive')
        g=nx.Graph()
        g.add_nodes_from(range(n))
        values={}
        for (u,v),b,c in zip(self.edges,self.b,self.c):
            if any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) for k in (u,v)):
                raise ValueError('edge endpoints must be integer matrix indices')
            if u==v or not 0<=u<n or not 0<=v<n or g.has_edge(u,v):
                raise ValueError('simple graph on the matrix indices required')
            g.add_edge(u,v)
            values[u,v]=b
            values[v,u]=c
        if (isinstance(self.root,(bool,np.bool_)) or not isinstance(self.root,(int,np.integer))
                or not nx.is_tree(g) or not 0<=self.root<n):
            raise ValueError('connected tree and valid integer root required')
        self.edges=list(nx.bfs_edges(g,self.root,sort_neighbors=sorted))
        self.b=np.array([values[u,v] for u,v in self.edges])
        self.c=np.array([values[v,u] for u,v in self.edges])
        self.parent=np.array([u for u,v in self.edges],dtype=int)
        self.child=np.array([v for u,v in self.edges],dtype=int)
        with np.errstate(over='ignore',invalid='ignore'):
            self.D=self.a[self.parent]*self.a[self.child]-self.b*self.c
        if not np.all(np.isfinite(self.D)) or np.any(self.D<=0):
            raise ValueError('all adjacent determinants must be finite and positive')
        self.values=values
        line=nx.Graph()
        line.add_nodes_from(range(n-1))
        for e in range(n-1):
            for f in range(e+1,n-1):
                if set(self.edges[e])&set(self.edges[f]):line.add_edge(e,f)
        self.constraints=[]
        for e,f in nx.bfs_edges(line,0,sort_neighbors=sorted):
            shared=next(iter(set(self.edges[e])&set(self.edges[f])))
            x=next(v for v in self.edges[e] if v!=shared)
            y=next(v for v in self.edges[f] if v!=shared)
            self.constraints.append((e,f,x,shared,y))
        assert len(self.constraints)==n-2

    def out_multiplier(self,edge,vertex):
        p,v=self.edges[edge]
        return 1. if p==vertex else -self.a[p]/self.a[vertex]

    def factors(self,alpha,a=None,b=None,c=None,active_edge=None,active_alpha=None):
        a=self.a if a is None else np.asarray(a)
        b=self.b if b is None else np.asarray(b)
        c=self.c if c is None else np.asarray(c)
        alpha=np.asarray(alpha,dtype=float).copy()
        if alpha.shape!=self.b.shape or not np.all(np.isfinite(alpha)) or np.any(alpha==0):
            raise ValueError('nonzero finite alpha required for each tree edge')
        D=a[self.parent]*a[self.child]-b*c
        beta=D/(a[self.parent]**2*alpha)
        if active_edge is not None:
            alpha[active_edge]=beta[active_edge]=active_alpha
        u,v=np.zeros((len(a),2)),np.zeros((len(a),2))
        u[self.root,0]=v[self.root,0]=np.sqrt(a[self.root])
        for e,(p,k) in enumerate(self.edges):
            u[k]=c[e]/a[p]*u[p]+alpha[e]*np.array([-v[p,1],v[p,0]])
            v[k]=b[e]/a[p]*v[p]+beta[e]*np.array([-u[p,1],u[p,0]])
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise ValueError('tree recurrence exceeded numerical range')
        return u,v

    def q_from_alpha(self,alpha):
        alpha=np.asarray(alpha,dtype=float)
        q=[]
        for e,f,x,v,y in self.constraints:
            ae=self.out_multiplier(e,v)*alpha[e]
            af=self.out_multiplier(f,v)*alpha[f]
            K=-self.D[f]*ae/af
            q.append((self.values[x,v]*self.values[v,y]-K)/self.a[v])
        return np.array(q)

    def infer_alpha(self,q,tol=1e-10):
        q=np.asarray(q,dtype=float)
        if q.shape!=(len(self.a)-2,) or not np.all(np.isfinite(q)):
            raise ValueError('n-2 finite directed two-hop observations required')
        alpha=np.zeros(len(self.edges))
        alpha[0]=np.sqrt(self.D[0])/self.a[self.parent[0]]
        for qi,(e,f,x,v,y) in zip(q,self.constraints):
            K=self.values[x,v]*self.values[v,y]-self.a[v]*qi
            if abs(K)<=tol*max(1.,abs(self.a[v]*qi)):
                raise ValueError('a two-hop cross minor is zero or numerically unresolved')
            alpha[f]=-self.D[f]*self.out_multiplier(e,v)*alpha[e]/(K*self.out_multiplier(f,v))
        if not np.all(np.isfinite(alpha)) or np.any(alpha==0):
            raise ValueError('two-hop reconstruction exceeded numerical range')
        return alpha

    def local_q(self,W):
        return np.array([W[x,y] for e,f,x,v,y in self.constraints])

    def old_losses(self,W):
        W=np.asarray(W)
        return .5*(np.sum((np.diagonal(W,axis1=-2,axis2=-1)-self.a)**2,axis=-1)
              +np.sum((W[...,self.parent,self.child]-self.b)**2,axis=-1)
              +np.sum((W[...,self.child,self.parent]-self.c)**2,axis=-1))


def plan_tree(source,tree,q,points=11,tol=1e-8):
    source=_validated_source(source,tree,points,tol)
    src=tree.infer_alpha(tree.local_q(source))
    dst=tree.infer_alpha(q)
    u,v=tree.factors(src)
    reconstructed=_checked_matrix(u,v)
    if np.max(np.abs(reconstructed-source))>tol:
        raise ValueError('source is not a numerically compatible rank-two completion')
    blocks=[np.array([[tree.a[p],tree.b[e]],[tree.c[e],tree.a[k]]])
            for e,(p,k) in enumerate(tree.edges)]
    decompositions=[np.linalg.svd(B) for B in blocks]
    h=np.array([.5*s[-1]**2 for u,s,vt in decompositions])
    trunc=[s[0]*np.outer(u[:,0],vt[0]) for u,s,vt in decompositions]
    signs=np.sign(src)
    cost=lambda a:float(np.max(h[signs!=np.sign(a)],initial=0.))
    if cost(-dst)<cost(dst):dst=-dst
    barrier=cost(dst)
    canonical=np.sqrt(tree.D)/tree.a[tree.parent]
    values=[reconstructed]
    def append(alpha):
        u,v=tree.factors(alpha)
        values.append(_checked_matrix(u,v))
    def adjust(start,end):
        if not np.array_equal(np.sign(start),np.sign(end)):
            raise ValueError('same-component adjustment cannot change orientation')
        for t in np.linspace(0,1,points)[1:]:
            append(np.sign(start)*np.exp((1-t)*np.log(np.abs(start))+t*np.log(np.abs(end))))
    adjust(src,signs*canonical)
    for edge in np.flatnonzero(signs!=np.sign(dst)):
        for reverse in (False,True):
            if reverse:signs[edge]*=-1
            for amount in np.linspace(1 if reverse else 0,0 if reverse else 1,points):
                block=(1-amount)*blocks[edge]+amount*trunc[edge]
                a,b,c=tree.a.copy(),tree.b.copy(),tree.c.copy()
                p,k=tree.edges[edge]
                a[p],a[k],b[edge],c[edge]=block[0,0],block[1,1],block[0,1],block[1,0]
                active=signs[edge]*np.sqrt(tree.D[edge]*(1-amount))/a[p]
                u,v=tree.factors(signs*canonical,a,b,c,edge,active)
                values.append(_checked_matrix(u,v))
    adjust(signs*canonical,dst)
    values=np.asarray(values)
    losses=tree.old_losses(values)
    final_q=tree.local_q(values[-1])
    if np.max(np.abs(final_q-q))>tol or abs(np.max(losses)-barrier)>tol:
        raise RuntimeError('tree construction failed its observed-data certificate')
    return values,{'predicted_peak':barrier,'sampled_peak':float(np.max(losses)),
        'source_error':float(np.max(np.abs(values[0]-source))),
        'target_q_error':float(np.max(np.abs(final_q-q))),
        'endpoint_old_loss':float(losses[-1]),'local_barriers':h}


def plan_tree_rank3(source,tree,q,points=11,tol=1e-8):
    """Zero-old-loss temporary rank-three path, returning to rank two."""
    source=_validated_source(source,tree,points,tol)
    src=tree.infer_alpha(tree.local_q(source))
    dst=tree.infer_alpha(q)
    u,v=tree.factors(src)
    reconstructed=_checked_matrix(u,v)
    if np.max(np.abs(reconstructed-source))>tol:
        raise ValueError('source is not a compatible rank-two completion')
    S=np.ones(len(tree.a))
    for e,(p,k) in enumerate(tree.edges):S[k]=S[p]*np.sqrt(tree.c[e]/tree.b[e])
    gram_magnitude=S[tree.parent]*S[tree.child]*np.sqrt(tree.D)/tree.a[tree.parent]
    values=[reconstructed]
    def adjust(start,end):
        for t in np.linspace(0,1,points)[1:]:
            alpha=np.sign(start)*np.exp((1-t)*np.log(np.abs(start))+t*np.log(np.abs(end)))
            u,v=tree.factors(alpha)
            values.append(_checked_matrix(u,v))
    def unit_factors(signs):
        theta=np.zeros(len(tree.a))
        for e,(p,k) in enumerate(tree.edges):
            cosine=np.sqrt(tree.b[e]*tree.c[e]/(tree.a[p]*tree.a[k]))
            theta[k]=theta[p]+signs[e]*np.arccos(cosine)
        return np.column_stack((np.cos(theta),np.sin(theta),np.zeros(len(theta))))
    def matrix(Z):
        return _checked_matrix((S*np.sqrt(tree.a))[:,None]*Z,(np.sqrt(tree.a)/S)[:,None]*Z)
    adjust(src,np.sign(src)*gram_magnitude)
    Z=unit_factors(np.sign(src))
    target_Z=unit_factors(np.sign(dst))
    if np.max(np.abs(values[-1]-matrix(Z)))>tol:
        raise RuntimeError('diagonal-similarity Gram representative mismatch')
    digraph=nx.DiGraph(tree.edges)
    for e,(p,k) in enumerate(tree.edges):
        axis=Z[p]/np.linalg.norm(Z[p])
        a=Z[k]-axis*(axis@Z[k])
        b=target_Z[k]-axis*(axis@target_Z[k])
        angle=math.atan2(float(axis@np.cross(a,b)),float(a@b))
        descendants=sorted({k}|nx.descendants(digraph,k))
        initial=Z[descendants].copy()
        for t in np.linspace(0,angle,points)[1:]:
            Z[descendants]=(initial*math.cos(t)+np.cross(axis,initial)*math.sin(t)
                            +np.outer(initial@axis,axis)*(1-math.cos(t)))
            values.append(matrix(Z))
    adjust(np.sign(dst)*gram_magnitude,dst)
    values=np.asarray(values)
    qerror=float(np.max(np.abs(tree.local_q(values[-1])-q)))
    losses=tree.old_losses(values)
    if qerror>tol or np.max(losses)>tol:
        raise RuntimeError('temporary-rank tree path failed observation checks')
    return values,{'predicted_peak':0.,'sampled_peak':float(np.max(losses)),
        'source_error':float(np.max(np.abs(values[0]-source))),
        'target_q_error':qerror,'endpoint_old_loss':float(losses[-1]),'local_barriers':[]}


def main():
    rows=[]
    for n in (8,16,32):
        shapes={
            'path':[(i,i+1) for i in range(n-1)],
            'star':[(0,i) for i in range(1,n)],
            'binary':[((i-1)//2,i) for i in range(1,n)],
            'random':[(int(np.random.default_rng(98000+i).integers(i)),i) for i in range(1,n)],
        }
        for shape,edges in shapes.items():
            rng=np.random.default_rng(94000+n+sum(map(ord,shape)))
            a=np.exp(rng.normal(0,.08,n))
            scale=np.sqrt(a[[u for u,v in edges]]*a[[v for u,v in edges]])*rng.uniform(.1,.8,n-1)
            skew=rng.normal(0,.08,n-1)
            tree=TreeData(a,edges,scale*np.exp(skew),scale*np.exp(-skew))
            canonical=np.sqrt(tree.D)/tree.a[tree.parent]
            src=canonical*rng.choice([-1.,1.],n-1)*np.exp(rng.normal(0,.08,n-1))
            dst=canonical*rng.choice([-1.,1.],n-1)*np.exp(rng.normal(0,.08,n-1))
            u,v=tree.factors(src)
            source=u@v.T
            q=tree.q_from_alpha(dst)
            for rank,planner in [(2,plan_tree),(3,plan_tree_rank3)]:
                path,info=planner(source,tree,q,points=7)
                tail=np.linalg.svd(path,compute_uv=False)[:,rank:]
                rows.append({'n':n,'shape':shape,'intermediate_rank':rank,
                    'old_count':3*n-2,'new_count':n-2,
                    'edges':json.dumps(tree.edges),'new_two_hops':json.dumps(tree.constraints),
                    **{k:v for k,v in info.items() if k!='local_barriers'},
                    'max_rank_tail':float(np.max(tail,initial=0)),
                    'final_rank2_tail':float(np.max(np.linalg.svd(path[-1],compute_uv=False)[2:],initial=0)),
                    'max_norm':float(np.max(np.linalg.norm(path,axis=(1,2)))),
                    'nodes':len(path)})
    write_csv(ROOT/'outputs'/'tree_planner.csv',rows)
    print({'cases':len(rows),'shapes':sorted({r['shape'] for r in rows}),
           'max_barrier_error':max(abs(r['predicted_peak']-r['sampled_peak']) for r in rows)})


if __name__=='__main__':main()
