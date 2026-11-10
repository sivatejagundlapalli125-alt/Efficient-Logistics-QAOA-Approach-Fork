from pathlib import Path
import sys
import streamlit as st
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.preprocessing.preprocess import preprocess_pipeline
from src.qubo.qubo_builder import build_tsp_qubo,validate_qubo
from src.classical.ortools_solver import solve_tsp
from src.quantum.qiskit_qaoa import run_qaoa,decode_qaoa_solution
from src.viz.plots import plot_cost_heatmap,plot_route_graph
st.set_page_config(page_title="Efficient Logistics QAOA",page_icon="🚚",layout="wide")
st.title("🚚 Efficient Logistics Optimization Using QAOA")
st.write("Compare a classical delivery route with a small QAOA state-vector simulation.")
with st.sidebar:
    st.header("Route settings")
    count=st.slider("Number of locations",3,6,3)
    st.caption("QAOA state-vector demo supports up to 3 locations. Classical optimization supports the full sample.")
if st.button("Optimize route",type="primary"):
    try:
        matrix,names,_=preprocess_pipeline(ROOT/"data/supply_chain_data.csv",ROOT/"data/instance_small.csv",count)
        Q=build_tsp_qubo(matrix)
        if not validate_qubo(Q): st.error("QUBO validation failed."); st.stop()
        result=solve_tsp(matrix)
        a,b=st.columns(2)
        a.metric("Classical route cost",f"{result['cost']:.2f}")
        a.write(" → ".join(names[i] for i in result["route"]+[result["route"][0]]))
        a.caption(f"Exact enumeration · {result['solve_time']:.4f}s · {result['status']}")
        if count<=3:
            q=run_qaoa(Q,reps=1,shots=512)
            if q["success"]:
                decoded=decode_qaoa_solution(q["solution"],count)
                if decoded["is_valid"]:
                    route=decoded["route"]; cost=sum(matrix[route[i],route[(i+1)%count]] for i in range(count))
                    b.metric("QAOA route cost",f"{cost:.2f}"); b.write(" → ".join(names[i] for i in route+[route[0]]))
                else: b.info("QAOA bitstring constraints: "+"; ".join(decoded["constraint_violations"]))
                b.caption(f"p=1 simulation · {q['execution_time']:.4f}s · {q['shots']} shots")
            else: b.info(q["message"])
        else: b.info("QAOA is skipped for larger instances because their state vectors need more than 12 variables. Classical optimization still finds the exact route.")
        c,d=st.columns(2); c.pyplot(plot_cost_heatmap(matrix,names),clear_figure=True); d.pyplot(plot_route_graph(result["route"],names,matrix),clear_figure=True)
        st.success("Optimization finished.")
    except Exception as exc: st.exception(exc)
else:
    st.info("Choose the number of locations, then select Optimize route.")
    st.caption("Uses bundled sample data. No API key or external quantum service is needed.")
