# Efficient Logistics Optimization Using QAOA

This GitHub fork contains a small interactive logistics-routing demonstration. The original upstream project is by Vidyasri; this copy adds the missing source modules and bundled sample data needed to run the demo.

## Run in VS Code or GitHub Codespaces

Open a terminal at the repository root and run:

```bash
cd Efficient-Logistics-QAOA-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run demo.py
```

On Windows PowerShell, activate the environment with:

```powershell
cd Efficient-Logistics-QAOA-Project
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
streamlit run demo.py
```

Streamlit prints a local URL. In Codespaces, open the forwarded port shown in the **Ports** panel to view the app.

## What the demo does

- Loads the included six-location travel-cost dataset.
- Builds and validates a QUBO for the cyclic traveling-salesperson problem.
- Finds an exact classical route for the small sample.
- Compares it with a local p=1 QAOA state-vector simulation for up to three locations.
- Displays the cost matrix and route diagram.

The state-vector simulation grows exponentially, so QAOA is skipped when the selected instance exceeds 12 binary variables. The classical result still runs. The demo uses bundled sample data and does not need an API key or remote quantum service.

## Project layout

```
Efficient-Logistics-QAOA-Project/
├── demo.py
├── data/                    # Sample locations and route costs
└── src/
    ├── preprocessing/       # CSV validation and cost matrix
    ├── qubo/                 # TSP QUBO formulation
    ├── classical/            # Exact small-instance route solver
    ├── quantum/              # p=1 QAOA state-vector simulation
    └── viz/                  # Cost heatmap and route plot
```
