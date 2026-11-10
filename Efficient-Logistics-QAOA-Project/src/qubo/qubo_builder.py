import numpy as np

def _one_hot(Q, group, penalty):
    for u in group: Q[u,u] -= penalty
    for k,u in enumerate(group):
        for v in group[k+1:]: Q[u,v] += penalty; Q[v,u] += penalty

def build_tsp_qubo(cost_matrix, penalty_A=500.0, penalty_B=500.0):
    costs=np.asarray(cost_matrix,dtype=float)
    if costs.ndim!=2 or costs.shape[0]!=costs.shape[1] or costs.shape[0]<2 or not np.isfinite(costs).all():
        raise ValueError("cost_matrix must be a finite square matrix")
    n=len(costs); Q=np.zeros((n*n,n*n)); var=lambda city,pos: city*n+pos
    for a in range(n):
        for b in range(n):
            if a==b: continue
            for pos in range(n):
                u,v=var(a,pos),var(b,(pos+1)%n); Q[u,v]+=costs[a,b]/2; Q[v,u]+=costs[a,b]/2
    for city in range(n): _one_hot(Q,[var(city,pos) for pos in range(n)],float(penalty_A))
    for pos in range(n): _one_hot(Q,[var(city,pos) for city in range(n)],float(penalty_B))
    return Q

def validate_qubo(Q):
    Q=np.asarray(Q,dtype=float)
    return Q.ndim==2 and Q.shape[0]==Q.shape[1] and np.isfinite(Q).all() and np.allclose(Q,Q.T)
