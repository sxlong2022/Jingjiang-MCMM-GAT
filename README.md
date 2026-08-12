# Jingjiang-MCMM-GAT: Hybrid Wetted-Channel Geometry Forecasting Pipeline

This repository hosts the source code, configurations, and demonstration datasets for the paper:
**"Decoupling Geomorphic Potential and Anthropogenic Constraints in Hybrid Wetted-Channel Geometry Forecasting"**

The pipeline couples a physical planform prior (Meander Centerline Migration Model, MCMM) with a data-driven delayed stochastic State-Space Model (delayed SSM) of channel geometry, and a constraint-aware Graph Attention Network (GAT) to model wetted-channel centerlines and widths under managed conditions.

---

## 1. Directory Structure

```
Jingjiang-MCMM-GAT/
├── LICENSE.txt                  # Open-source license (MIT)
├── requirements.txt             # Python environment dependencies
├── README.md                    # This instruction file
├── data/
│   ├── raw/                     # Pre-packaged observation & hydrologic datasets
│   │   ├── jingjiang_centerline.xy               # Initial coordinate baseline (2016-01)
│   │   ├── jingjiang_monthly_2016_2024_final.csv # Historical observations (Q, W, D, E_rate)
│   │   ├── shashi_monthly_q_qs.csv               # Shashi station monthly discharge & sediment load
│   │   └── monthly_centerlines/                  # 108 monthly satellite centerline .npy files
│   ├── processed/               # Reach static features & forcing outputs
│   │   ├── engineering_features.csv             # 1000-node binary protection database (is_protected)
│   │   ├── geometric_features.csv               # 1000-node reach geometry features
│   │   ├── landuse_features.csv, soil_features.csv, vegetation_features.csv
│   │   ├── scenario_effects_effect_sizes.csv    # Pre-computed multi-scenario Cohen's d values
│   │   └── [forcing_q_d50_scenarios_*.csv]      # (Generated at runtime by Step 1)
│   └── graph/                   # (Generated at runtime by Step 2)
│       └── cv_monthly/                          # Compiled PyTorch Geometric graph datasets (.pt)
├── src/
│   ├── forcing/
│   │   ├── build_forcing_scenarios.py           # Vogel-style bootstrap generator for Q/Qs/d50
│   │   ├── wd_ssm_params.json                   # Fitted parameters for the delayed SSM
│   │   └── qc_wd_ssm_backtest_summary.json     # Backtest performance summary for width/depth
│   ├── mcmm/
│   │   ├── fortran/                             # Bogoni et al. (WRR 2017) Fortran meander kinematics core
│   │   ├── compile_windows.cmd                  # Gfortran compilation script for Windows
│   │   ├── compile_unix                         # Gfortran compilation script for UNIX/Linux/macOS
│   │   ├── prepare_mcmm_input.py                # Formatting scripts for monthly inputs
│   │   ├── apply_depth_series_to_timeseries.py  # Inundation-depth integration
│   │   ├── convert_centerline.py                # UTM-to-internal coordinate converter
│   │   ├── run_mcmm_sim.py                      # Batch Fortran driver
│   │   ├── parse_mcmm_output.py                 # Coordinate post-processing
│   │   ├── build_mcmm_timeseries_from_forcing.py# Coupling delayed SSM outputs to MCMM timeseries
│   │   └── run_forecast_rollout.py              # Long-horizon forecast rollout driver
│   └── gnn/
│       ├── train.py                             # GAT training script (with rolling Cross-Validation)
│       ├── models/
│       │   ├── gat_model.py                     # GAT architecture (with is_protected features)
│       │   └── loss_functions.py                # Geomorphic residual loss functions
│       └── scripts/
│           ├── graph_construction/
│           │   └── build_graph.py               # Graph dataset builder (merges engineering_features)
│           └── coupling/
│               └── rolling_hindcast.py          # Combined GNN-MCMM rolling backtest and forecasting
├── outputs/                     # (Generated at runtime by Step 3)
│   └── cv_monthly_results/                      # Model checkpoints (fold1_model.pt) & evaluation metrics
└── notebooks/
    └── demo.ipynb                   # End-to-end interactive demonstration notebook
```

---

## 2. Installation and Compilation

### Environment Setup
We recommend using Conda to manage Python dependencies. Set up the environment using:

```bash
# Create and activate environment
conda create -n riverpiv python=3.10 -y
conda activate riverpiv

# Install dependencies (ensure PyTorch and PyTorch Geometric match your CUDA capabilities)
pip install -r requirements.txt
```

### Compiling the MCMM Core
A Fortran compiler (such as `gfortran`) is required to compile the meander kinematics core:

*   **Windows (CMD/PowerShell)**:
    ```cmd
    cd src/mcmm
    compile_windows.cmd
    ```
*   **UNIX / Linux / macOS**:
    ```bash
    cd src/mcmm
    chmod +x compile_unix
    ./compile_unix
    ```
This will compile the Fortran files under `fortran/` and output a binary executable (`mcmm.exe` or `mcmm`) in the `src/mcmm/` directory.

---

## 3. Core Workflows

### Step 1: Future Scenario Forcing (2025–2030)
Generate synthetic daily flow discharge ($Q$), came sediment concentration ($Q_s$), and bed material grain size ($d_{50}$) for Normal ($S_0$), Wet ($S_{wet}$), and Dry ($S_{dry}$) scenarios:

```bash
python src/forcing/build_forcing_scenarios.py \
  --start 202501 --end 203012 \
  --n_members 30 --seed 0 \
  --q_method bootstrap_year \
  --scenarios S0,Swet,Sdry
```

### Step 2: Compile Graph Datasets
Construct GNN graphs for the 1000 nodes of the reach, merging the `is_protected` static features with MCMM simulated baselines:

```bash
python src/gnn/scripts/graph_construction/build_graph.py \
  --fold 1 --use_depth
```

### Step 3: Train GAT Error-Correction Model
Train the `MigrationGAT` model on rolling cross-validation folds to predict spatial centerline residuals:

```bash
python src/gnn/train.py \
  --fold 1 --use_width --use_depth --seed 0
```

### Step 4: Long-Horizon Rollouts and Stress-Testing
Execute the 30-member ensemble rollout under the $S_0$ scenario using the delayed SSM for width/depth and MCMM for curvature-driven centerline shifts:

```bash
python src/mcmm/run_forecast_rollout.py \
  --start 202501 --end 203012 \
  --initial_month 202412 \
  --scenario S0 --member 0 \
  --wd_method delayed_ssm \
  --wd_ssm_params src/forcing/wd_ssm_params.json \
  --wd_ssm_sample 1 \
  --wd_clip_ref hist_monthly_p05_p95 \
  --tag wdssm_acc_v1
```

---

## 4. Citation and Credits

If you use this codebase, models, or datasets in your research, please cite:

> **Song, X.**, Huang, H., Zhang, L., Xu, H., & Bai, Y. (2026). Decoupling Geomorphic Potential and Anthropogenic Constraints in Hybrid Wetted-Channel Geometry Forecasting. *Environmental Modelling & Software* (Under Review).

### BibTeX
```bibtex
@article{Song2026JingjiangMCMMGAT,
  author  = {Song, Xiaolong and Huang, Hai and Zhang, Lei and Xu, Haijue and Bai, Yuchuan},
  title   = {Decoupling Geomorphic Potential and Anthropogenic Constraints in Hybrid Wetted-Channel Geometry Forecasting},
  journal = {Environmental Modelling \& Software},
  year    = {2026},
  note    = {Under Review}
}
```

### Prior Model Credits
*   **MCMM Kinematics Core**: Derived from the original numerical model by Bogoni, M., Putti, M., and Lanzoni, S. (2017), *Modeling meander morphodynamics over self-formed heterogeneous floodplains*, Water Resources Research, 53, 5137–5157. https://doi.org/10.1002/2017WR020726
*   **Jingjiang Reach Application**: Reframed and coupled under the new hybrid GAT-SSM architecture for stage-conditioned wetted-channel geometry forecasting.
