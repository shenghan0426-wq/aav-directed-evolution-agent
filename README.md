# AAV Directed-Evolution Scientific Agent

This project implements a lightweight scientific agent for protein directed
evolution. It uses AAV variant fitness data to simulate a closed-loop workflow:
data processing, fitness prediction, LLM-based mutation reasoning,
knowledge-enhanced candidate selection, and virtual directed evolution.

## Environment

Python 3.10 or newer is recommended.

Install dependencies:

```bash
cd aav_baseline
python -m pip install -r requirements.txt
```

Main Python packages:

- `pandas`, `numpy`, `scikit-learn`
- `torch`, `fair-esm`
- `openai`, `python-dotenv`, `pydantic`
- `matplotlib`
- `streamlit`

OpenAI API usage is optional. If LLM reasoning is enabled, create a `.env` file:

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_name
```

## Data Source

The project uses public AAV directed-evolution fitness data. Each sample
contains a protein sequence and a continuous fitness score. The full local data
files are:

- `train.csv`
- `validation.csv`
- `test.csv`
- `two_vs_many.csv`

The training split is treated as historical experimental data. The test split
is treated as the unknown candidate pool for virtual experimental evaluation.

For GitHub-friendly reproduction, the repository includes small sample splits:

- `sample_data/sample_train.csv`
- `sample_data/sample_validation.csv`
- `sample_data/sample_test.csv`

These sample files keep the same columns as the full data:

- `sequence`
- `target`
- `set`
- `validation`

Large files are ignored by `.gitignore`, including full datasets, ESM
embeddings, model checkpoints, and private `.env` files.

## Project Structure

- `01_model_evaluation.ipynb`: evaluates fitness prediction baselines and ESM2-MLP results.
- `02_llm_agent.ipynb`: runs the LLM Agent workflow for candidate recommendation.
- `03_knowledge_enhanced_agent.ipynb`: adds amino-acid properties, mutation rules, and knowledge graph triples.
- `04_virtual_evolution.ipynb`: simulates three rounds of virtual directed evolution.
- `model_evaluation.py`: evaluation metrics, Ridge baseline, and fitness-head prediction helpers.
- `llm_agent_core.py`: mutation parsing, history summarization, candidate generation, LLM calls, and scoring logic.
- `knowledge_base.py`: amino-acid physicochemical properties, conservative substitutions, mutation-count rules, and knowledge graph construction.
- `virtual_evolution.py`: iterative virtual evolution simulation and strategy comparison.
- `app_demo.py`: Streamlit demo for interactive mutation recommendation.
- `run_small_repro.py`: small reproducible workflow using sample data.
- `test_llm_agent_core.py`: unit tests.

## Run Commands

### 1. Small Reproducible Workflow

This is the simplest command to verify the project on the included sample data.
It trains a one-hot Ridge baseline, generates knowledge-enhanced recommendations,
and compares predicted fitness with true fitness.

```bash
cd aav_baseline
python run_small_repro.py
```

Output:

```text
artifacts/small_repro_recommendations.csv
```

### 2. Notebook Workflow

Run notebooks in this order:

```text
01_model_evaluation.ipynb
02_llm_agent.ipynb
03_knowledge_enhanced_agent.ipynb
04_virtual_evolution.ipynb
```

In `02_llm_agent.ipynb`, these switches control external calls:

```python
RUN_OPENAI = False
RUN_ESM = False
```

Set `RUN_OPENAI = True` to call the OpenAI API. Set `RUN_ESM = True` to load
ESM-2 and the saved fitness head.

### 3. Command-Line Agent

Prepare historical summaries and candidate pools:

```bash
cd aav_baseline
python run_llm_agent.py --prepare-only
```

Run the full LLM + ESM workflow:

```bash
python run_llm_agent.py
```

The full workflow requires `OPENAI_API_KEY` and
`artifacts/esm2_fitness_head.pt`.

### 4. Interactive Demo

Launch the Streamlit demo:

```bash
cd aav_baseline
python -m streamlit run app_demo.py
```

Default AAV mutation-region input:

```text
DEEEIRTTNPVATEQYGSVSTNLQRGNR
```

The demo returns candidate mutations, predicted fitness, true fitness when
available, knowledge-enhanced scores, and recommendation reasons.

### 5. Tests

```bash
cd ..
python -m unittest aav_baseline.test_llm_agent_core
```

## Main Results

- The ESM2-MLP fitness predictor achieved test Pearson correlation of 0.617
  and Spearman correlation of 0.616.
- Knowledge enhancement reduced the average number of mutations in recommended
  candidates from 17.4 to 4.0.
- Knowledge-enhanced recommendations improved mean true fitness from -4.743 to
  3.282 in the before/after comparison.
- In the virtual directed-evolution simulation, the knowledge-enhanced LLM Agent
  reached a best true fitness of 6.262.
- The original LLM Agent often selected candidates with too many mutations,
  illustrating the importance of mutation-count constraints and biochemical
  rules.

Important output files:

- `artifacts/model_evaluation_metrics.csv`
- `artifacts/knowledge_before_after_summary.csv`
- `artifacts/knowledge_before_after_comparison.csv`
- `artifacts/virtual_evolution_summary.csv`
- `artifacts/virtual_evolution_topk_by_round.csv`
- `artifacts/virtual_evolution_curve.png`
- `artifacts/small_repro_recommendations.csv`

## GitHub Upload

Create a new empty GitHub repository, then run:

```bash
cd aav_baseline
git init
git add README.md requirements.txt .gitignore __init__.py
git add *.py *.ipynb sample_data
git add artifacts/model_evaluation_metrics.csv
git add artifacts/knowledge_before_after_summary.csv
git add artifacts/knowledge_before_after_comparison.csv
git add artifacts/virtual_evolution_summary.csv
git add artifacts/virtual_evolution_topk_by_round.csv
git add artifacts/virtual_evolution_curve.png
git add artifacts/small_repro_recommendations.csv
git commit -m "Add AAV directed evolution scientific agent"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

Do not upload `.env`, full datasets, ESM embeddings, or model checkpoints to
the main repository.

## External Tools

This project uses public AAV fitness data, ESM-2 protein language model
features, OpenAI API for optional LLM reasoning, and Python scientific computing
libraries.
