# Circuit Subspace: SVD-based Transformer Circuit Discovery

This repository implements a method for discovering interpretable circuits in transformer language models using Singular Value Decomposition (SVD) and learnable masks on the singular value components.

## Overview

The core idea is to decompose each attention head's OV (Output-Value) and QK (Query-Key) matrices using SVD, then learn sparse masks over the singular value directions. This allows us to:

1. **Identify minimal circuits** - Find the smallest set of singular value directions that preserve model behavior
2. **Intervene on specific directions** - Swap activations along identified directions to flip model predictions
3. **Achieve high interpretability** - Each direction has a clear mathematical interpretation



## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Circuit_Subspace.git
cd Circuit_Subspace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
Circuit_Subspace/
├── src/                          # Core source code
│   ├── models/
│   │   └── masked_transformer_circuit.py  # Main circuit discovery model
│   ├── data/
│   │   └── data_loader.py        # Dataset loaders for IOI, GP, GT tasks
│   └── utils/
│       ├── utils.py              # Utility functions
│       └── visualization.py      # Plotting and visualization
├── experiments/
│   ├── train.py                  # Training script
│   ├── ablation/
│   │   ├── smart_range_ablation.py      # Range-swap intervention experiments
│   │   └── comprehensive_sigma_test.py  # Sigma amplification experiments
│   └── evaluation/
│       ├── comprehensive_metrics_table.py  # Sparsity vs accuracy evaluation
│       └── generate_sigma_table.py         # Generate results tables
├── configs/
│   └── gp_config.yaml            # Configuration for GP task
├── data/                         # Place datasets here
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Data

Place your dataset CSV files in the `data/` directory. Expected format:

**For Gender Pronoun (GP) task:**
- `train_1k_gp.csv`, `val_gp.csv`, `test_gp.csv`
- Columns: `prefix`, `pronoun`, `name`, `corr_prefix`, `corr_pronoun`, `corr_name`

**For Indirect Object Identification (IOI) task:**
- `train_1k_ioi.csv`, `val_ioi.csv`, `test_1k_ioi.csv`
- Columns: `ioi_sentences_input`, `ioi_sentences_labels`, `corr_ioi_sentences_input`, etc.

### 2. Train a Circuit

```bash
cd Circuit_Subspace
python experiments/train.py --config configs/gp_config.yaml
```

This will:
- Load GPT-2 small and compute SVD for all attention heads
- Learn sparse masks over singular value directions
- Save the trained model and visualizations to `logs/`

### 3. Run Intervention Experiments

After training, run intervention experiments to test the discovered circuit:

```bash
python experiments/ablation/smart_range_ablation.py
```

This performs "range-swap" interventions where activations along identified directions are swapped to their values for the opposite gender.

### 4. Generate Results Tables

```bash
python experiments/evaluation/generate_sigma_table.py
```



### Learnable Masks

We learn masks `m_i` for each singular value `σ_i`:

```
W_OV_masked = U @ diag(m * S) @ V^T
```

The masks are initialized near 1.0 and trained with:
- **KL divergence loss** - Preserve original model behavior
- **L1 sparsity penalty** - Encourage sparse circuits

### Intervention Formula

To flip a prediction from "he" to "she", we intervene on identified directions:

```python
intervention = (target_activation - current_activation) * sigma_i * v_i^T
```

Where:
- `current_activation = V' @ u_i` (projection of context onto direction)
- `target_activation` = mean activation for opposite gender
- The intervention is added to the residual stream before final LayerNorm

## Configuration Options

Key configuration parameters in `configs/gp_config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.learning_rate` | Mask learning rate | 2.0e-2 |
| `training.l1_weight` | Sparsity penalty weight | 1.95e-4 |
| `training.kl_opt` | Target KL divergence | 0.10 |
| `training.l1_opt` | Target L1 norm | 3500 |
| `masking.mask_init_value` | Initial mask values | 0.99 |
| `masking.sparsity_threshold` | Threshold for "active" | 1e-3 |

## Results

### Sigma Amplification Experiments

| Experiment | σ | Flip to He (%) | Flip to She (%) |
|------------|---|----------------|-----------------|
| E.1: All 4 dirs → She | 1.0 | 0.0 | 49.5 |
| E.1: All 4 dirs → She | 2.0 | 0.0 | 100.0 |
| E.2: All 4 dirs → He | 1.0 | 80.6 | 0.0 |
| E.2: All 4 dirs → He | 2.0 | 100.0 | 0.0 |

### Key Findings

1. **4 directions suffice** - Just 4 singular value directions (out of ~98,000+ possible) control gender prediction
2. **100% flip rate** - With σ=2.0 amplification, we achieve perfect prediction flipping
3. **Dose-response relationship** - Higher σ amplification → stronger intervention effect
4. **Bidirectional control** - Same directions can flip predictions in both directions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{circuit_subspace,
  title={Circuit Subspace: SVD-based Transformer Circuit Discovery},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Circuit_Subspace}
}
```

## License

MIT License

## Acknowledgments

This work builds on:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for transformer interpretability
- Research on mechanistic interpretability from Anthropic and others
