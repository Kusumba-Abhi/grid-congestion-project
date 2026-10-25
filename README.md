# Power Grid Congestion Prediction and Redispatch

An end-to-end machine learning and deep learning system that predicts transmission line congestion in power grids and recommends corrective redispatch using optimal power flow (OPF). The system is validated across six structurally distinct grid topologies, including a custom Indian grid model built from real CEA and Grid-India data.

## Overview

Transmission congestion occurs when power flows exceed the thermal limits of grid lines, forcing operators to curtail renewable generation or shed load. This project predicts congestion before it happens and then suggests how to rebalance generation to relieve it.

The pipeline has three stages:

1. **Scenario generation**: thousands of operating scenarios are simulated in pandapower by varying load levels, wind speed, solar irradiance and line outage events.
2. **Congestion prediction**: two models classify each scenario as congested or not congested.
3. **Redispatch optimization**: when congestion is predicted, an OPF-based optimizer reschedules generation, prioritizing renewable sources, to bring line loading back within limits.

## Models

| Model | Inputs | Description |
|---|---|---|
| Model 1 | Structured features | Random Forest baseline on grid operating features |
| Model 2 | Structured features + text | Dual-branch LSTM fusion model combining structured features with natural-language grid alert descriptions |

At inference time, the system routes each scenario to the model with the higher validation F1 score for that topology.

## Topologies

The same pipeline is trained and validated separately on six grids:

| Topology | Description |
|---|---|
| `us_ieee118` | IEEE 118-bus test system (primary benchmark) |
| `uk_ieee30` | IEEE 30-bus test system |
| `fr_pegase89` | PEGASE 89-bus European test case |
| `gb_full` | Great Britain transmission network |
| `eu_case300` | IEEE 300-bus test system |
| `india_synthetic` | Custom 5-region Indian grid (Northern, Western, Southern, Eastern, North-Eastern) parameterized with real CEA generation capacity and Grid-India inter-regional transfer capacity data |

No public Indian grid dataset exists, so `india_synthetic` was built from published capacity data to provide an Indian case study.

## Results

Selected test-set results:

| Topology | Model 1 (RF) | Model 2 (LSTM fusion) |
|---|---|---|
| `us_ieee118` | 98.4% accuracy, 95.6% F1 | 99.1% accuracy, 97.2% F1 |
| `india_synthetic` | 91.2% accuracy, 88.7% F1 | 89.0% accuracy, 86.0% F1 |
| `gb_full` | 99.7% accuracy | 83.1% accuracy |
| `eu_case300` | 93.6% accuracy, 87.0% F1 | 71.7% accuracy, 41.8% F1 |

**Key finding:** the benefit of multimodal (structured + text) fusion depends on a topology's electrical operating margin. Fusion helps on grids with balanced margins, but underperforms the structured baseline on narrow-margin grids such as `gb_full` and `eu_case300`. This pattern is only visible because the system is tested across multiple topologies.

On the IEEE 118-bus system, the redispatch optimizer resolved all 40 of 40 convergent congested scenarios while increasing renewable usage from 11.5% to 28.3%.

## Project Structure

```
grid-congestion-project/
├── app/             # Trained models and application code
├── data/            # Generated scenarios and datasets
├── notebooks/       # Interactive demo notebook
├── src/             # Scenario generation, training and redispatch code
└── requirements.txt # Python dependencies
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/somashekar135-tech/grid-congestion-project.git
   cd grid-congestion-project
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the interactive demo:
   ```bash
   jupyter notebook notebooks/demo.ipynb
   ```
   Use the sliders to set wind speed, solar irradiance, load level and outage events, then click **Run Prediction** to see both models' predictions and the redispatch result.

## Tech Stack

Python, pandapower, scikit-learn, PyTorch, Jupyter and ipywidgets.

## Future Work

- A single graph neural network (GNN) model that generalizes across topologies, instead of separate weights per grid
- Integration of real-time grid telemetry

## Authors

Somashekar Patnam and Sai Abhinav, Department of Information Technology, Chaitanya Bharathi Institute of Technology (CBIT), under the guidance of Dr Prathima Thirumalareddy.