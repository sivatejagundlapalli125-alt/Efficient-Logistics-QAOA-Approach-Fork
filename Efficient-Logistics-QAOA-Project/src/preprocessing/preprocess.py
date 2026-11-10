from pathlib import Path
import numpy as np
import pandas as pd

def preprocess_pipeline(supply_chain_path, instance_path, max_nodes=6):
    supply_path, instance_file = Path(supply_chain_path), Path(instance_path)
    if not supply_path.exists() or not instance_file.exists():
        raise FileNotFoundError("Missing bundled data CSV files")
    edges, instance = pd.read_csv(supply_path), pd.read_csv(instance_file)
    if not {"origin", "destination", "cost"}.issubset(edges.columns):
        raise ValueError("Supply data needs origin,destination,cost columns")
    col = "location" if "location" in instance.columns else instance.columns[0]
    names = instance[col].dropna().astype(str).drop_duplicates().tolist()[:int(max_nodes)]
    if len(names) < 2: raise ValueError("At least two locations are required")
    index = {name:i for i,name in enumerate(names)}
    matrix = np.full((len(names),len(names)), np.inf); np.fill_diagonal(matrix,0)
    for row in edges.itertuples(index=False):
        a,b,c = str(row.origin),str(row.destination),float(row.cost)
        if a in index and b in index:
            if not np.isfinite(c) or c < 0: raise ValueError("Costs must be non-negative")
            matrix[index[a],index[b]] = c
    missing = np.argwhere(~np.isfinite(matrix))
    if len(missing):
        i,j=missing[0]; raise ValueError(f"Missing cost from {names[i]} to {names[j]}")
    return matrix,names,index
