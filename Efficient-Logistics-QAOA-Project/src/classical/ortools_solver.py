from itertools import permutations
from time import perf_counter
import numpy as np

def calculate_route_cost(route,cost_matrix):
    m=np.asarray(cost_matrix,dtype=float); route=list(map(int,route))
    if len(route)<2 or len(set(route))!=len(route): raise ValueError("Route must visit each city once")
    return float(sum(m[route[k],route[(k+1)%len(route)]] for k in range(len(route))))

def solve_tsp(cost_matrix,time_limit_seconds=30):
    m=np.asarray(cost_matrix,dtype=float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or len(m)<2: raise ValueError("cost matrix must be square")
    start=perf_counter(); best=None; value=float("inf")
    for tail in permutations(range(1,len(m))):
        if perf_counter()-start>time_limit_seconds: break
        route=(0,)+tail; cost=calculate_route_cost(route,m)
        if cost<value: best,value=route,cost
    if best is None: raise RuntimeError("No route found")
    return {"route":list(best),"cost":float(value),"status":"OPTIMAL" if len(m)<=10 else "FEASIBLE","solve_time":perf_counter()-start}
