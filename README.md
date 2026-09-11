# AAV Directed-Evolution Scientific Agent

This project implements a small scientific agent for protein directed evolution.
It uses AAV variant fitness data to simulate an experimental workflow:
historical data analysis, mutation hypothesis generation, candidate selection,
fitness scoring, knowledge-enhanced filtering, and virtual evolution.

## Project Question

Can a lightweight LLM Agent help prioritize next-round protein variants for
directed evolution, beyond random selection or direct model ranking?

## Data

The project uses AAV directed-evolution fitness data stored locally as:

- `train.csv`
- `validation.csv`
- `test.csv`
- `two_vs_many.csv`
- `artifacts/observed_experiments.csv`

The train split is treated as completed experiments. The test split is treated
as unknown candidate variants for virtual evaluation.

For GitHub review and lightweight reproduction, this repository also includes
small sample splits:

- `sample_data/sample_train.csv`
- `sample_data/sample_validation.csv`
- `sample_data/sample_test.csv`

These sample files preserve the original columns (`sequence`, `target`, `set`,
`validation`) and are sufficient to run the small reproducible pipeline without
large embeddings, model checkpoints, or an OpenAI API key.

Large local files such as `.npy`, full CSV datasets, and model checkpoints may
be excluded from GitHub. If excluded, place them back under `aav_baseline/`
using the filenames above before running all notebooks.

## Code Structure

- `model_evaluation.py`: baseline metrics, Ridge baseline, and ESM fitness-head evaluation helpers.
- `llm_agent_core.py`: data processing, mutation parsing, agent modules, candidate generation, ESM scoring, and critic logic.
- `knowledge_base.py`: amino-acid physicochemical rules, conservative substitutions, mutation-count filtering, and knowledge graph triples.
- `virtual_evolution.py`: iterative virtual directed-evolution simulation and strategy comparison.
- `app_demo.py`: Streamlit interactive demo for mutation recommendation.
- `run_small_repro.py`: small GitHub-friendly reproduction using sample CSV files and a Ridge baseline.
- `test_llm_agent_core.py`: unit tests for mutation parsing, knowledge rules, virtual evolution, and demo behavior.

## Main Notebooks

Run notebooks in this order:

1. `01_model_evaluation.ipynb`
2. `02_llm_agent.ipynb`
3. `03_knowledge_enhanced_agent.ipynb`
4. `04_virtual_evolution.ipynb`

`02_llm_agent.ipynb` contains two switches:

```python
RUN_OPENAI = False
RUN_ESM = False
```

Set `RUN_OPENAI = True` to call the OpenAI API and generate LLM reasoning.
Set `RUN_ESM = True` to load ESM-2 and the saved fitness head for scoring.

## Small Reproducible Run

This command runs a lightweight version of the workflow on the included sample
data. It performs data processing, trains a one-hot Ridge fitness baseline,
generates knowledge-enhanced candidate recommendations, and evaluates them
against the sample test fitness values.

```bash
cd aav_baseline
python -m pip install -r requirements.txt
python run_small_repro.py
```

Expected output includes validation MSE, Top-k recommended mutations,
predicted fitness, true fitness, and a saved CSV:

```text
artifacts/small_repro_recommendations.csv
```

## Interactive Demo

The project includes a small Streamlit demo. It lets a user enter the AAV
wild-type mutation region, choose Top-k and maximum mutation count, and receive
knowledge-enhanced mutation recommendations with predicted fitness and reasons.

Install dependencies:

```bash
cd aav_baseline
python -m pip install -r requirements.txt
```

Run the demo:

```bash
cd aav_baseline
python -m streamlit run app_demo.py
```

The default input sequence is:

```text
DEEEIRTTNPVATEQYGSVSTNLQRGNR
```

This demo is calibrated for the AAV mutation region used in the dataset. It
uses `train.csv`, `test.csv`, and, when available,
`artifacts/test_esm2_mlp_predictions.csv`.

## Command-Line Runs

From this folder:

```bash
python run_llm_agent.py --prepare-only
```

Full LLM + ESM run:

```bash
python run_llm_agent.py
```

The full run requires:

- `OPENAI_API_KEY` in `.env`
- optional `OPENAI_MODEL` in `.env`
- `artifacts/esm2_fitness_head.pt`

## Outputs

Important generated artifacts:

- `artifacts/model_evaluation_metrics.csv`
- `artifacts/model_evaluation_scatter.png`
- `artifacts/knowledge_enhanced_candidates.csv`
- `artifacts/knowledge_before_after_summary.csv`
- `artifacts/virtual_evolution_results.csv`
- `artifacts/virtual_evolution_summary.csv`
- `artifacts/virtual_evolution_curve.png`
- `artifacts/llm_agent_final_recommendations.csv`
- `artifacts/small_repro_recommendations.csv`

## Main Results

- ESM2-MLP fitness predictor: test Pearson = 0.617 and Spearman = 0.616.
- Knowledge enhancement reduced the average mutation count from 17.4 to 4.0.
- Knowledge-enhanced recommendations improved mean true fitness from -4.743
  before knowledge filtering to 3.282 after knowledge filtering.
- In the three-round virtual evolution experiment, knowledge-enhanced LLM Agent
  reached best true fitness 6.262, while the original LLM Agent failed because
  it tended to combine too many mutations.

The full analysis is in:

- `report/final_report.md`
- `report/final_report.pdf`

## Upload To GitHub

From the `aav_baseline/` folder:

```bash
git init
git add README.md requirements.txt .gitignore __init__.py
git add *.py *.ipynb sample_data report/final_report.md report/final_report.pdf
git add artifacts/model_evaluation_metrics.csv artifacts/knowledge_before_after_summary.csv
git add artifacts/knowledge_before_after_comparison.csv artifacts/virtual_evolution_summary.csv
git add artifacts/virtual_evolution_topk_by_round.csv artifacts/virtual_evolution_curve.png
git commit -m "Add AAV directed evolution scientific agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Large local files are ignored by `.gitignore`, including `.env`, `.npy` files,
large CSV datasets, and model checkpoints. If your advisor needs to reproduce
everything locally, share these files separately or upload them through GitHub
Release / Google Drive / institutional storage:

- `train.csv`
- `validation.csv`
- `test.csv`
- `two_vs_many.csv`
- `X_train_esm2_35M.npy`
- `X_val_esm2_35M.npy`
- `X_test_esm2_35M.npy`
- `artifacts/esm2_fitness_head.pt`
- `artifacts/test_esm2_mlp_predictions.csv`

After cloning, your advisor can place those files back under `aav_baseline/`,
install `requirements.txt`, run the notebooks in order, and launch the demo.

If only the GitHub repository is available, your advisor can still run:

```bash
python run_small_repro.py
```

This reproduces the workflow on `sample_data/` without large files.


## Agent Design

The LLM Agent is modular:

- Data Analyst: summarizes top variants and recurring mutations.
- Hypothesis Generator: proposes testable mutation hypotheses.
- Mutation Designer: selects next-round candidate variants.
- Fitness Evaluator: scores candidates using the fitness model.
- Scientific Critic: explains recommendations and failure modes.

## Knowledge Enhancement

The rule base uses amino-acid properties and mutation heuristics:

- charge
- polarity
- size
- hydrophobicity
- conservative vs radical substitutions
- mutation-count penalty
- invalid amino-acid checks

The project compares original LLM recommendations with
knowledge-enhanced recommendations.

## Testing

Run:

```bash
cd /Users/feishenghan/Documents/VScode
/Users/feishenghan/Documents/VScode/.venv/bin/python -m unittest aav_baseline.test_llm_agent_core
```

## External Tools And Sources

This project uses:

- AAV protein fitness data from a public directed-evolution benchmark.
- ESM-2 protein language model through the `fair-esm` Python package.
- OpenAI API for LLM Agent reasoning.
- ChatGPT/Codex assistance for code organization, notebook generation, and report drafting.

These uses should be cited in the final PDF report.
