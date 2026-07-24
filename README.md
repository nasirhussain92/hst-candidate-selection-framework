# A Hybrid Sentiment–Trajectory AI Framework for Candidate Selection: Beyond Static Evaluation

**Authors:** Nasir Hussain (Karachi Institute of Economics and Technology) and [Co-Author — to be added]

Code, synthetic-data generator, and real-data validation supporting the paper
*"A Hybrid Sentiment–Trajectory AI Framework for Candidate Selection: Beyond
Static Evaluation."* This repository reproduces every number, table, and
figure reported in the paper.

## Repository structure

```
.
├── code/
│   ├── generator.py                 # Synthetic candidate-data generator (§4)
│   ├── model.py                     # Shape-feature extractor Φ(·) and fusion model (§3)
│   ├── run_eval.py                  # General + favorable scenario evaluation (§5.1–5.2, Table 1)
│   ├── neural_proxy.py              # Raw-sequence neural benchmark (§5.3, Table 2)
│   ├── real_data_validation.py      # Real-data validation on Campus Recruitment data (§6, Table 3, Table 4)
│   └── make_figure1.py              # Generates Figure 1 (conceptual framework diagram)
├── notebooks/
│   └── HST_Framework_Reproducibility.ipynb  # Runnable, cell-by-cell reproduction of Tables 1–4 and Figure 1
├── data/
│   └── Placement_Data_Full_Class.csv  # Campus Recruitment dataset (Roshan, n.d.), CC0 license
├── figures/
│   └── figure1_framework.png        # Output of make_figure1.py
├── results/
│   ├── synthetic_general_favorable_results.txt  # Raw output backing Table 1
│   ├── neural_benchmark_results.txt             # Raw output backing Table 2
│   └── real_data_validation_results.txt         # Raw output backing Tables 3–4
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Reproducing the paper's results

```bash
pip install -r requirements.txt

# Table 1: general vs. favorable synthetic scenarios (§5.1–5.2)
python code/run_eval.py

# Table 2: raw-sequence neural benchmark (§5.3)
python code/neural_proxy.py

# Table 3 & 4: real-data validation and fairness audit (§6–§7)
python code/real_data_validation.py

# Figure 1: conceptual framework diagram
python code/make_figure1.py
```

Each script is self-contained and prints its results to standard output;
the exact output produced for the paper is saved under `results/` for
reference. All synthetic-data experiments use fixed random seeds, so
re-running `run_eval.py` and `neural_proxy.py` will reproduce the reported
numbers exactly.

## Reproducibility notebook

For readers who prefer a single, guided, cell-by-cell walkthrough rather
than running each script separately, `notebooks/HST_Framework_Reproducibility.ipynb`
reproduces Tables 1–4 and Figure 1 in one place, with the relevant section
of the paper referenced next to each result.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasirhussain92/hst-candidate-selection-framework/blob/main/notebooks/HST_Framework_Reproducibility.ipynb)

The real-data section of the notebook (Tables 3–4) requires uploading
`data/Placement_Data_Full_Class.csv` when prompted; every other section
runs with no setup beyond Colab's pre-installed packages.

## Data source

`data/Placement_Data_Full_Class.csv` is the **Campus Recruitment** dataset
originally published on Kaggle by Ben Roshan
(<https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement>),
released under a **CC0: Public Domain** license. It is redistributed here
unmodified for reproducibility.

## Scope and limitations

This code supports a research paper whose empirical claims are, by design,
partly based on synthetic data with a known generative structure (see
paper §4, §8 for a full discussion of limitations). The real-data
validation in `real_data_validation.py` uses a single, small (N = 215),
non-hiring-specific dataset and should be read as preliminary, not
confirmatory — see the paper's Limitations section.

## AI-assisted development disclosure

Portions of this codebase (the synthetic-data generator, evaluation
scripts, and this documentation) were written with the assistance of
Claude (Anthropic). All code was reviewed, tested, and validated by the
authors. See the paper's AI Disclosure Statement for full detail.

## Citation

If you use this code or data, please cite both the paper and this
repository (see `CITATION.cff`).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The dataset in `data/` retains its original CC0 license from Kaggle.
