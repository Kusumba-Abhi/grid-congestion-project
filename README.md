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
| `ieee30` | IEEE 30-bus test system |
| `fr_pegase89` | Real RTE (France) 89-bus network snapshot |
| `gb_full` | Real Great Britain transmission network |
| `eu_case300` | IEEE 300-bus test system |
| `india_synthetic` | Custom 5-region Indian grid (Northern, Western, Southern, Eastern, North-Eastern) parameterized with real CEA generation capacity and Grid-India inter-regional transfer capacity data |

No public Indian grid dataset exists, so `india_synthetic` was built from published capacity data to provide an Indian case study.

## Results

Test-set results for all six topologies (2,500 scenarios per topology, 70/15/15 train/validation/test split):

| Topology | Model 1 (RF) Accuracy | Model 1 (RF) F1 | Model 2 (LSTM fusion) Accuracy | Model 2 (LSTM fusion) F1 |
|---|---|---|---|---|
| `us_ieee118` | 0.984 | 0.956 | 0.991 | 0.972 |
| `ieee30` | 0.968 | 0.885 | 0.965 | 0.879 |
| `fr_pegase89` | 0.963 | 0.900 | 0.955 | 0.878 |
| `gb_full` | 0.997 | 0.994 | 0.831 | 0.591 |
| `eu_case300` | 0.936 | 0.870 | 0.717 | 0.418 |
| `india_synthetic` | 0.912 | 0.887 | 0.890 | 0.860 |

**Key finding:** the value of multimodal (structured + text) fusion depends on a topology's electrical operating margin. On grids where congestion has several drivers, the two models perform within about three F1 points of each other, and fusion is ahead on `us_ieee118`. On narrow-margin, load-dominated grids such as `gb_full` and `eu_case300`, fusion clearly underperforms the structured baseline. This pattern is only visible because the system is tested across multiple topologies.

**Redispatch:** the OPF-based redispatch module resolves detected congestion on four of the six topologies. On the IEEE 118-bus system, it reduced line loading from 101.46% to 86.99% while raising the renewable share from 19.9% to 45.1%. On the Indian grid model, it resolved a severe 131.10% loading down to 100%. On `fr_pegase89` and `gb_full`, the OPF does not yet converge; this is a known limitation and is being investigated.

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

## Publication

A paper based on this project, *"A Multi-Topology Machine Learning Framework for Power Grid Congestion Prediction and Renewable-Prioritized Redispatch"*, has been submitted to IIPESS 2027 (International Conference on Intelligent Power, Energy and Sustainable Systems, NIT Jalandhar) and is currently under review.

## Future Work

- Resolve OPF convergence on `fr_pegase89` and `gb_full` through network-specific reactive-power and voltage-constraint tuning
- Add a loading-margin safety term to the redispatch objective
- A single graph neural network (GNN) model that generalizes across topologies, instead of separate weights per grid
- A more detailed multi-bus-per-region Indian grid model as more transmission data becomes available

## Authors

Somashekar Patnam and K. Sai Abhinav, Department of Information Technology, Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, under the guidance of Dr Prathima Thirumalareddy, with S Kamala Kumari (Sir C R Reddy College of Engineering).