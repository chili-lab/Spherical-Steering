# Spherical Steering: Geometry-Aware Activation Rotation for Language Models

*Spherical Steering* is a inference-time activation steering method for controlling language models via geometry-consistent interventions. Instead of the standard activation addition, Spherical Steering performs a rotation: it treats steering as a directional update in representation space and rotates hidden activations along a geodesic toward a target direction, while keeping activation magnitudes intact.

This code base contains the code to replicate the experiments presented in the paper "Spherical Steering: Geometry-Aware Activation Rotation for Language Models".

## Table of Contents

- [Preparation](#preparation)
- [Data](#data)
- [Reproduce](#reproduce)
- [Key Modules](#key-modules)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Preparation

### Environment

- Python `3.10`
- Environment file: `environment.yml`

Create and activate:

```bash
conda env create -f environment.yml
conda activate spherical-steering
```

### TruthfulQA Repo

`evaluate_mc.py` imports evaluation metric from `./TruthfulQA`, so clone this repo first:

```bash
git clone https://github.com/sylinrl/TruthfulQA.git
```

## Data

- Main benchmark: `truthful_qa` (https://github.com/sylinrl/TruthfulQA).
- MC evaluation CSV default path: `./TruthfulQA/data/v1/TruthfulQA.csv`.
- Intermediate artifacts are written to:
  - `features/`
  - `prototypes/`
  - `results/`
  - `results_llm_judge/`
- Other benchmarks are under `./generic`.
- See `generic/README.md` for details.

## Usage

### TruthfulQA

Use the following scripts:

```bash
bash quickstart_llama.sh
bash quickstart_qwen.sh
```

Current quickstart defaults:

- `quickstart_llama.sh`
  - model: `llama3.1-8B-Instruct`
  - layer: `14`
  - `kappa=20.0`, `alpha=0.7`, `beta=-0.15`
- `quickstart_qwen.sh`
  - model: `Qwen2.5-7B-Instruct`
  - layer: `19`
  - `kappa=20.0`, `alpha=0.6`, `beta=0.4`

### Other Reasoning Benchmarks

For other reasoning multiple-choice benchmarks, use the pipeline in:

```bash
cd generic
```

See `generic/README.md` for details.

## Key Modules

- `get_activations.py`
  - extract last-token hidden states from answer pairs
- `get_prototypes.py`
  - compute `mu_T`, `mu_H` with 2-fold question-level split
- `evaluate_mc.py`
  - MC1/MC2/MC3 evaluation on held-out fold questions
- `evaluate_llm_judge.py`
  - open-ended generation + truth/info judge scoring
- `spherical_steering.py`
  - intervention hook and geometric steering logic
- `utils.py`
  - data loading and activation extraction helpers

## Citation

If you find this work useful, please cite:

```bibtex
@misc{you2026sphericalsteeringgeometryawareactivation,
      title={Spherical Steering: Geometry-Aware Activation Rotation for Language Models}, 
      author={Zejia You and Chunyuan Deng and Hanjie Chen},
      year={2026},
      eprint={2602.08169},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08169}, 
}
```

## Acknowledgements

- [baukit](https://github.com/davidbau/baukit)
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [ITI](https://github.com/likenneth/honest_llama)
