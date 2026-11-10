from time import perf_counter
import numpy as np

def run_qaoa(Q,reps=1,shots=512):
    """Dependency-free p=1 state-vector QAOA demo for up to 12 variables."""
    Q=np.asarray(Q,dtype=float); m=len(Q)
    if Q.ndim!=2 or Q.shape[0]!=Q.shape[1] or not np.isfinite(Q).all(): raise ValueError("Q must be finite and square")
    if m>12: return {"success":False,"message":f"State-vector QAOA supports 12 variables; this needs {m}.","execution_time":0}
    if reps!=1: return {"success":False,"message":"Only reps=1 is supported.","execution_time":0}
    started=perf_counter(); ids=np.arange(1<<m,dtype=np.uint32)
    bits=((ids[:,None]>>np.arange(m,dtype=np.uint32))&1).astype(float)
    energy=np.einsum("bi,ij,bj->b",bits,Q,bits,optimize=True)
    plus=np.full(len(ids),1/np.sqrt(len(ids)),dtype=np.complex128); best=1e100; best_state=None
    for gamma in np.linspace(0,2*np.pi,8,endpoint=False):
        cost=plus*np.exp(-1j*gamma*energy)
        for beta in np.linspace(0,np.pi,8,endpoint=False):
            state=cost.copy(); c,s=np.cos(beta),np.sin(beta)
            for q in range(m):
                mask=1<<q; lo=ids[(ids&mask)==0]; hi=lo|mask; a,b=state[lo].copy(),state[hi].copy()
                state[lo]=c*a-1j*s*b; state[hi]=-1j*s*a+c*b
            p=np.abs(state)**2; score=float(p@energy)
            if score<best: best,best_state=score,state
    p=np.abs(best_state)**2; rng=np.random.default_rng(17)
    sampled=rng.choice(len(ids),size=max(1,int(shots)),p=p/p.sum()); chosen=int(sampled[np.argmin(energy[sampled])])
    return {"success":True,"solution":bits[chosen].astype(int).tolist(),"cost":float(energy[chosen]),"expectation":best,"execution_time":perf_counter()-started,"circuit_depth":1,"shots":int(shots)}

def decode_qaoa_solution(solution,n_cities):
    n=int(n_cities); x=np.asarray(solution,dtype=int).reshape(n,n)
    route=[int(np.argmax(x[:,p])) for p in range(n)]; errors=[]
    if not np.all(x.sum(axis=1)==1): errors.append("each city must appear once")
    if not np.all(x.sum(axis=0)==1): errors.append("each position must contain one city")
    return {"route":route,"is_valid":not errors,"constraint_violations":errors}
