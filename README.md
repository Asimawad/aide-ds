# AIDE: Machine Learning CodeGen with Inference-Time Scaling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![PyPI](https://img.shields.io/pypi/v/aideml?color=blue)](https://pypi.org/project/aideml/)  
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)  
[![Discord](https://dcbadge.vercel.app/api/server/Rq7t8wnsuA?compact=true&style=flat)](https://discord.gg/Rq7t8wnsuA)  
[![Twitter Follow](https://img.shields.io/twitter/follow/WecoAI?style=social)](https://twitter.com/WecoAI)  

AIDE is a machine learning code generation agent originally designed to solve data science tasks using natural language descriptions, leveraging APIs for models like OpenAI, Anthropic, and Google DeepMind. This forked version shifts the focus to enhancing small, open-source reasoning models (distilled DeepSeek models: 7B, 14B, 32B, 70B) through **inference-time scaling methods**. We aim to improve the reasoning and code generation capabilities of these models for machine learning tasks, starting with a self-reflection strategy that critiques and edits code in two distinct phases. In a benchmark of over 60 Kaggle data science competitions, AIDE demonstrated impressive performance, surpassing 50% of Kaggle participants on average (see the [technical report](https://www.weco.ai/blog/technical-report) for details).

Key features include:

1. **Instruct with Natural Language**: Describe your problem or requirements in natural language.
2. **Deliver Solution in Source Code**: AIDE generates Python scripts for tested machine learning pipelines, ensuring transparency, reproducibility, and the ability to further improve the code.
3. **Iterative Optimization**: AIDE iteratively runs, debugs, evaluates, and improves ML code autonomously.
4. **Visualization**: Tools to visualize the solution tree, providing insights into the experimentation process and what works or doesn’t.

## Project Focus: Inference-Time Scaling on DeepSeek Models

This project explores **inference-time scaling methods** to enhance the performance of small, open-source reasoning models (DeepSeek 7B, 14B, 32B, 70B) on machine learning tasks. Unlike the original AIDE, which relied on API calls to large proprietary models, we run these smaller models locally or on cloud infrastructure (e.g., GCP) to generate and refine ML solutions. Our initial strategy implements a **self-reflection mechanism** split into two phases:

- **Critique Phase**: The model reviews its own code, identifying bugs, incorrect logic, hallucinated imports/methods, and improper file paths (e.g., ensuring outputs save to `./submission/submission.csv`).
- **Edit Phase**: The model applies minimal edits to fix the identified issues, preserving the original code structure as much as possible.

This self-reflection approach is a stepping stone for more sophisticated inference-time scaling techniques, such as chain-of-thought prompting, multi-step reasoning, or tree-based search augmentation, to further improve the reasoning capabilities of these models.

## How to Use AIDE?

### Setup

Ensure you have `Python>=3.10` installed and run:

```bash
pip install -U aideml
```

Install `unzip` to allow the agent to autonomously extract your data.

#### Original API Setup (Optional)
The original AIDE required API keys for OpenAI or Anthropic:

```bash
export OPENAI_API_KEY=<your API key>
# or
export ANTHROPIC_API_KEY=<your API key>
```

#### New Setup for DeepSeek Models
This fork uses local or cloud-hosted DeepSeek models instead of API calls. To run the models:

1. **Install Dependencies**:
   - Install `vLLM` or  `HuggingFace`  for efficient local inference (recommended for DeepSeek models):
     ```bash
     pip install vllm
     ```
   - Ensure you have a compatible GPU (e.g., NVIDIA 3090 with 24GB VRAM for 7B in 4-bit quantization).

2. **Download DeepSeek Models**:
   - Download the distilled DeepSeek models (7B, 14B, 32B, or 70B) from their official repository or Hugging Face.
   - Example for DeepSeek 7B:
     ```bash
     huggingface-cli download deepseek-ai/deepseek-7b --local-dir ./models/deepseek-7b
     ```

3. **Set Model Path**:
   - Specify the model path in your configuration (see `config.yaml` or command-line options below).

### Running AIDE via the Command Line

To run AIDE with the new DeepSeek models:

```bash
aide data_dir="<path to your data directory>" goal="<describe the agent's goal for your task>" eval="<(optional) describe the evaluation metric>" agent.code.model="deepseek-7b" agent.model.path="./models/deepseek-7b"
```

For example, to run AIDE on the [House Prices prediction task](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) using DeepSeek 7B:

```bash
aide data_dir="example_tasks/house_prices" goal="Predict the sales price for each house" eval="Use the RMSE metric between the logarithm of the predicted and observed values." agent.code.model="deepseek-r1:latesb"
```

#### Options:
- `data_dir` (required): Directory containing your task data (`.csv` files, images, etc.).
- `goal`: Describe the prediction task (e.g., "Predict sales price for houses").
- `eval`: Evaluation metric (e.g., "RMSE between the logarithm of predicted and observed values").
- `agent.code.model`: Specify the DeepSeek model (e.g., `deepseek-7b`, `deepseek-14b`).
- `agent.model.path`: Path to the downloaded DeepSeek model weights.
- `agent.steps`: Number of improvement iterations (default: 20).
- `agent.search.num_drafts`: Number of initial drafts to generate (default: 5).

Alternatively, provide the task description as a `desc_str` string or in a plaintext file via `desc_file`:

```bash
aide data_dir="my_data_dir" desc_file="my_task_description.txt" agent.code.model="deepseek-7b" agent.model.path="./models/deepseek-7b"
```

#### Outputs:
The results are stored in the `logs` directory:
- `logs/<experiment-id>/best_solution.py`: Python code of the best solution based on the validation metric.
- `logs/<experiment-id>/journal.json`: Metadata of the experiment runs, including intermediate code, plans, and evaluation results.
- `logs/<experiment-id>/tree_plot.html`: Interactive visualization of the solution tree—open in a browser to explore the experimentation process.

The `workspaces` directory contains all generated files and data.

### Advanced Usage

Customize AIDE’s behavior with additional options:
- `agent.code.temp`: Temperature for code generation (default: 0.7—adjust for more/less creativity).
- `agent.code.quantization`: Quantization level for DeepSeek models (e.g., `4-bit` for 7B on a 6GB GPU).
- Check `config.yaml` for more options.

### Using AIDE in Python

Integrate AIDE into your Python scripts:

```python
import aide

exp = aide.Experiment(
    data_dir="example_tasks/house_prices",
    goal="Predict the sales price for each house.",
    eval="Use the RMSE metric between the logarithm of the predicted and observed values.",
    agent_config={
        "code": {
            "model": "deepseek-7b",
            "model_path": "./models/deepseek-7b",
            "quantization": "4-bit"
        }
    }
)

best_solution = exp.run(steps=10)

print(f"Best solution has validation metric: {best_solution.valid_metric}")
print(f"Best solution code: {best_solution.code}")
```

## Development

To install AIDE for development, clone this repository and install locally:

```bash
git clone https://github.com/Asimawad/aide-ds.git
cd aide-ds
uv pip install .  # or pip install -r requirements.txt
uv pip install -e .  # or pip install -e .
```

Contribution guide coming soon.

## Algorithm Description

AIDE’s problem-solving approach is inspired by human data scientists. It generates initial solution drafts and iteratively refines them based on performance feedback using **Solution Space Tree Search**, which consists of:

- **Solution Generator**: Proposes new solutions by creating drafts or modifying existing ones (e.g., fixing bugs, improving logic).
- **Evaluator**: Assesses solutions by running them and extracting the evaluation metric (e.g., RMSE) from logs.
- **Base Solution Selector**: Picks the most promising solution for the next iteration.

### Self-Reflection Strategy
This fork introduces a **self-reflection mechanism** to enhance the reasoning of DeepSeek models during inference:
1. **Critique Phase**: The model reviews its generated code, identifying issues like missing imports, incorrect file paths (e.g., `./submission/submission.csv`), hallucinated methods, or logical errors (e.g., incorrect RMSE calculation).
2. **Edit Phase**: The model applies minimal edits to fix the identified issues, preserving the original code structure.

This two-step self-reflection is a foundational inference-time scaling strategy, enabling the model to iteratively improve its outputs. It serves as a starting point for more advanced techniques, such as chain-of-thought prompting, multi-agent collaboration, or tree-based reasoning augmentation, to further boost the performance of small reasoning models on complex ML tasks.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)

## Solution Gallery

| Domain                           | Task                                                                    | Top%  | Solution Link                                                     | Competition Link                                                                                   |
|:---------------------------------|:------------------------------------------------------------------------|:------|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| Urban Planning                   | Forecast city bikeshare system usage                                    | 5%    | [link](sample_results/bike-sharing-demand.py)                           | [link](https://www.kaggle.com/competitions/bike-sharing-demand/overview)                           |
| Physics                          | Predicting Critical Heat Flux                                           | 56%   | [link](sample_results/playground-series-s3e15.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e15/overview)                       |
| Genomics                         | Classify bacteria species from genomic data                             | 0%    | [link](sample_results/tabular-playground-series-feb-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/overview)            |
| Agriculture                      | Predict blueberry yield                                                 | 58%   | [link](sample_results/playground-series-s3e14.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e14/overview)                       |
| Healthcare                       | Predict disease prognosis                                               | 0%    | [link](sample_results/playground-series-s3e13.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e13/overview)                       |
| Economics                        | Predict monthly microbusiness density in a given area                   | 35%   | [link](sample_results/godaddy-microbusiness-density-forecasting.py)     | [link](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/overview)     |
| Cryptography                     | Decrypt shakespearean text                                              | 91%   | [link](sample_results/ciphertext-challenge-iii.py)                      | [link](https://www.kaggle.com/competitions/ciphertext-challenge-iii/overview)                      |
| Data Science Education           | Predict passenger survival on Titanic                                   | 78%   | [link](sample_results/tabular-playground-series-apr-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2021/overview)            |
| Software Engineering             | Predict defects in c programs given various attributes about the code   | 0%    | [link](sample_results/playground-series-s3e23.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e23/overview)                       |
| Real Estate                      | Predict the final price of homes                                        | 5%    | [link](sample_results/home-data-for-ml-course.py)                       | [link](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)                       |
| Real Estate                      | Predict house sale price                                                | 36%   | [link](sample_results/house-prices-advanced-regression-techniques.py)   | [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)   |
| Entertainment Analytics          | Predict movie worldwide box office revenue                              | 62%   | [link](sample_results/tmdb-box-office-prediction.py)                    | [link](https://www.kaggle.com/competitions/tmdb-box-office-prediction/overview)                    |
| Entertainment Analytics          | Predict scoring probability in next 10 seconds of a rocket league match | 21%   | [link](sample_results/tabular-playground-series-oct-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/overview)            |
| Environmental Science            | Predict air pollution levels                                            | 12%   | [link](sample_results/tabular-playground-series-jul-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2021/overview)            |
| Environmental Science            | Classify forest categories using cartographic variables                 | 55%   | [link](sample_results/forest-cover-type-prediction.py)                  | [link](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview)                  |
| Computer Vision                  | Predict the probability of machine failure                              | 32%   | [link](sample_results/playground-series-s3e17.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e17/overview)                       |
| Computer Vision                  | Identify handwritten digits                                             | 14%   | [link](sample_results/digit-recognizer.py)                              | [link](https://www.kaggle.com/competitions/digit-recognizer/overview)                              |
| Manufacturing                    | Predict missing values in dataset                                       | 70%   | [link](sample_results/tabular-playground-series-jun-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/overview)            |
| Manufacturing                    | Predict product failures                                                | 48%   | [link](sample_results/tabular-playground-series-aug-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)            |
| Manufacturing                    | Cluster control data into different control states                      | 96%   | [link](sample_results/tabular-playground-series-jul-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/overview)            |
| Natural Language Processing      | Classify toxic online comments                                          | 78%   | [link](sample_results/jigsaw-toxic-comment-classification-challenge.py) | [link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) |
| Natural Language Processing      | Predict passenger transport to an alternate dimension                   | 59%   | [link](sample_results/spaceship-titanic.py)                             | [link](https://www.kaggle.com/competitions/spaceship-titanic/overview)                             |
| Natural Language Processing      | Classify sentence sentiment                                             | 42%   | [link](sample_results/sentiment-analysis-on-movie-reviews.py)           | [link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)           |
| Natural Language Processing      | Predict whether a tweet is about a real disaster                        | 48%   | [link](sample_results/nlp-getting-started.py)                           | [link](https://www.kaggle.com/competitions/nlp-getting-started/overview)                           |
| Business Analytics               | Predict total sales for each product and store in the next month        | 87%   | [link](sample_results/competitive-data-science-predict-future-sales.py) | [link](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview) |
| Business Analytics               | Predict book sales for 2021                                             | 66%   | [link](sample_results/tabular-playground-series-sep-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/overview)            |
| Business Analytics               | Predict insurance claim amount                                          | 80%   | [link](sample_results/tabular-playground-series-feb-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2021/overview)            |
| Business Analytics               | Minimize penalty cost in scheduling families to santa's workshop        | 100%  | [link](sample_results/santa-2019-revenge-of-the-accountants.py)         | [link](https://www.kaggle.com/competitions/santa-2019-revenge-of-the-accountants/overview)         |
| Business Analytics               | Predict yearly sales for learning modules                               | 26%   | [link](sample_results/playground-series-s3e19.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e19/overview)                       |
| Business Analytics               | Binary classification of manufacturing machine state                    | 60%   | [link](sample_results/tabular-playground-series-may-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/overview)            |
| Business Analytics               | Forecast retail store sales                                             | 36%   | [link](sample_results/tabular-playground-series-jan-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/overview)            |
| Business Analytics               | Predict reservation cancellation                                        | 54%   | [link](sample_results/playground-series-s3e7.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e7/overview)                        |
| Finance                          | Predict the probability of an insurance claim                           | 13%   | [link](sample_results/tabular-playground-series-mar-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-mar-2021/overview)            |
| Finance                          | Predict loan loss                                                       | 0%    | [link](sample_results/tabular-playground-series-aug-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2021/overview)            |
| Finance                          | Predict a continuous target                                             | 42%   | [link](sample_results/tabular-playground-series-jan-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/overview)            |
| Finance                          | Predict customer churn                                                  | 24%   | [link](sample_results/playground-series-s4e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e1/overview)                        |
| Finance                          | Predict median house value                                              | 58%   | [link](sample_results/playground-series-s3e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e1/overview)                        |
| Finance                          | Predict closing price movements for nasdaq listed stocks                | 99%   | [link](sample_results/optiver-trading-at-the-close.py)                  | [link](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)                  |
| Finance                          | Predict taxi fare                                                       | 100%  | [link](sample_results/new-york-city-taxi-fare-prediction.py)            | [link](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)            |
| Finance                          | Predict insurance claim probability                                     | 62%   | [link](sample_results/tabular-playground-series-sep-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/overview)            |
| Biotech                          | Predict cat in dat                                                      | 66%   | [link](sample_results/cat-in-the-dat-ii.py)                             | [link](https://www.kaggle.com/competitions/cat-in-the-dat-ii/overview)                             |
| Biotech                          | Predict the biological response of molecules                            | 62%   | [link](sample_results/tabular-playground-series-oct-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/overview)            |
| Biotech                          | Predict medical conditions                                              | 92%   | [link](sample_results/icr-identify-age-related-conditions.py)           | [link](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview)           |
| Biotech                          | Predict wine quality                                                    | 61%   | [link](sample_results/playground-series-s3e5.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e5/overview)                        |
| Biotech                          | Predict binary target without overfitting                               | 98%   | [link](sample_results/dont-overfit-ii.py)                               | [link](https://www.kaggle.com/competitions/dont-overfit-ii/overview)                               |
| Biotech                          | Predict concrete strength                                               | 86%   | [link](sample_results/playground-series-s3e9.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e9/overview)                        |
| Biotech                          | Predict crab age                                                        | 46%   | [link](sample_results/playground-series-s3e16.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e16/overview)                       |
| Biotech                          | Predict enzyme characteristics                                          | 10%   | [link](sample_results/playground-series-s3e18.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e18/overview)                       |
| Biotech                          | Classify activity state from sensor data                                | 51%   | [link](sample_results/tabular-playground-series-apr-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/overview)            |
| Biotech                          | Predict horse health outcomes                                           | 86%   | [link](sample_results/playground-series-s3e22.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e22/overview)                       |
| Biotech                          | Predict the mohs hardness of a mineral                                  | 64%   | [link](sample_results/playground-series-s3e25.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e25/overview)                       |
| Biotech                          | Predict cirrhosis patient outcomes                                      | 51%   | [link](sample_results/playground-series-s3e26.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e26/overview)                       |
| Biotech                          | Predict obesity risk                                                    | 62%   | [link](sample_results/playground-series-s4e2.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e2/overview)                        |
| Biotech                          | Classify presence of feature in data                                    | 66%   | [link](sample_results/cat-in-the-dat.py)                                | [link](https://www.kaggle.com/competitions/cat-in-the-dat/overview)                                |
| Biotech                          | Predict patient's smoking status                                        | 40%   | [link](sample_results/playground-series-s3e24.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e24/overview)                       |
