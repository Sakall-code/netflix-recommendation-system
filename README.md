# 🎬 Netflix Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)
[![Scikit-Surprise](https://img.shields.io/badge/Library-Scikit--Surprise-green.svg)](http://surpriselib.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-red.svg)](https://grouplens.org/datasets/movielens/100k/)

An end-to-end Collaborative Filtering recommendation engine powered by **Matrix Factorization (Singular Value Decomposition - SVD)**. This project evaluates recommendation performance on benchmark movie rating data using rigorous **50-fold cross-validation**, detailed statistical robustness analysis, and publication-ready diagnostic visualizations.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Methodology & Architecture](#-methodology--architecture)
- [Dataset](#-dataset)
- [Performance & Benchmark Results](#-performance--benchmark-results)
- [Visualizations & Diagnostics](#-visualizations--diagnostics)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quickstart & Usage](#-quickstart--usage)
- [Roadmap & Enhancements](#-roadmap--enhancements)
- [References & Acknowledgments](#-references--acknowledgments)
- [License](#-license)

---

## 📖 Overview

Recommendation systems are central to modern streaming services like Netflix, Spotify, and YouTube. When millions of users interact with vast catalogs of content, recommendation engines filter through the noise to deliver hyper-personalized suggestions.

This project implements a **Collaborative Filtering** recommendation model using **Singular Value Decomposition (SVD)** to decompose user-item rating matrices into lower-dimensional latent representations. It models both global rating baselines (user and movie biases) and latent user-item interaction dynamics.

### Key Highlights
- **Matrix Factorization:** SVD algorithm parameterized with 100 latent factors, stochastic gradient descent (SGD) learning, and $L_2$ regularization.
- **Statistical Rigor:** 50-Fold Cross-Validation (`KFold` with randomized shuffling) to ensure models are impervious to split variance and data leakage.
- **Comprehensive Evaluation:** Tracked across both **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.
- **Visual Analytics:** Generates 5 distinct diagnostic plots (line plots, error bars, violin/box distribution, and overlay comparison).

---

## 🔬 Methodology & Architecture

### Collaborative Filtering via SVD
The core prediction formula decomposes the predicted rating $\hat{r}_{u,i}$ of user $u$ on item $i$ as:

$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$

Where:
- $\mu$: Global mean rating across all ratings in the dataset.
- $b_u$: Bias (baseline deviation) of user $u$ from the global average.
- $b_i$: Bias (baseline deviation) of movie $i$ from the global average.
- $p_u \in \mathbb{R}^k$: Latent factor vector representing user preferences ($k = 100$).
- $q_i \in \mathbb{R}^k$: Latent factor vector representing movie characteristics ($k = 100$).

### Optimization Objective
The model minimizes regularized squared error using Stochastic Gradient Descent (SGD):

$$\min_{p_*, q_*, b_*} \sum_{(u,i) \in R_{train}} \left(r_{u,i} - \hat{r}_{u,i}\right)^2 + \lambda \left(b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2\right)$$

### Model Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_factors` | `100` | Number of latent factor dimensions |
| `n_epochs` | `25` | Number of SGD iteration epochs |
| `lr_all` | `0.005` | Learning rate for all parameters |
| `reg_all` | `0.02` | $L_2$ regularization penalty term |
| `random_state` | `42` | Seed ensuring deterministic reproducibility |

---

## 📊 Dataset

The model benchmarks on the **MovieLens 100K** dataset curated by GroupLens Research at the University of Minnesota, the industry-standard benchmark for collaborative filtering and the Netflix Prize:

- **Total Ratings:** 100,000 ratings (scale: 1 to 5 stars)
- **Unique Users:** 943 users
- **Unique Movies:** 1,682 movies
- **Sparsity:** ~93.7% sparse user-item matrix
- **Demographics:** Each user rated at least 20 movies

The dataset is automatically fetched and cached via `surprise.Dataset.load_builtin('ml-100k')`.

---

## 📈 Performance & Benchmark Results

The model was evaluated using **50-Fold Cross-Validation** to rigorously assess model stability and generalization error across diverse test splits:

| Evaluation Metric | Mean Score | Standard Deviation ($\sigma$) | Min Score | Max Score |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE** (Root Mean Squared Error) | **0.9236** | $\pm 0.0183$ | 0.8839 | 0.9646 |
| **MAE** (Mean Absolute Error) | **0.7268** | $\pm 0.0152$ | 0.6985 | 0.7580 |
| **Fit Time per Fold** | **~1.74 s** | $\pm 0.22\text{ s}$ | 1.58 s | 2.35 s |
| **Test Time per Fold** | **~0.015 s** | $\pm 0.02\text{ s}$ | 0.01 s | 0.13 s |

### Key Takeaways
1. **Low Variance ($\sigma < 0.02$):** Demonstrates that the model generalizes consistently regardless of how user-item interactions are partitioned.
2. **Predictive Accuracy:** An average MAE of **~0.72** indicates that predicted ratings deviate from actual user feedback by less than three-quarters of a star on a 1–5 scale.
3. **Computational Efficiency:** Fitting complete folds in under 2 seconds facilitates fast re-training cycles and hyperparameter tuning.

---

## 📉 Visualizations & Diagnostics

The notebook generates 5 vector-quality publication figures (`.pdf` / inline display):

1. **Fig. X — Fold-wise RMSE Performance:** Line plot tracking RMSE across all 50 folds with an overlaid dashed threshold for the mean RMSE ($0.924$).
2. **Fig. Y — Fold-wise MAE Performance:** Line plot tracking MAE stability across all 50 folds with mean MAE threshold ($0.727$).
3. **Fig. Z — Mean Error Comparison (RMSE vs MAE):** Bar chart displaying comparative aggregate error with error bars ($\pm \sigma$) and value labels.
4. **Fig. A — Statistical Robustness (Violin & Box Plot):** Visualizes the full distribution shape, median, interquartile range (IQR), and spread across both error metrics.
5. **Fig. B — Comparative Error Overlay Analysis:** Dual-line overlay with a shaded difference area highlighting the consistent offset between RMSE and MAE across every fold.

---

## 📂 Project Structure

```text
Netflix_Reco/
├── Netflix_Recommender.ipynb    # Main Jupyter Notebook with model training & visualizations
├── README.md                    # Project documentation
└── [Generated Figures]
    ├── Fig_X_Fold_RMSE.pdf             # Line plot of fold-wise RMSE
    ├── Fig_Y_Fold_MAE.pdf              # Line plot of fold-wise MAE
    ├── Fig_Z_Mean_Comparison.pdf       # Bar chart comparing RMSE vs MAE
    ├── Fig_A_Statistical_Robustness.pdf# Violin + Box plot of error distributions
    └── Fig_B_Overlay_Comparison.pdf    # Overlay plot of RMSE vs MAE across folds
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Netflix_Reco.git
cd Netflix_Reco
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
> **Note:** `scikit-surprise` requires NumPy `< 2.0`. Ensure you install compatible versions as specified:

```bash
pip install "numpy<2.0" scikit-surprise pandas matplotlib seaborn jupyter
```

---

## 🚀 Quickstart & Usage

You can run the notebook directly or execute the core pipeline in Python:

### Run the Notebook
```bash
jupyter notebook Netflix_Recommender.ipynb
```

### Minimal Python Example
```python
import numpy as np
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import KFold

# 1. Load benchmark dataset
data = Dataset.load_builtin('ml-100k')

# 2. Instantiate SVD algorithm
algo = SVD(n_factors=100, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)

# 3. 5-Fold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for fold, (trainset, testset) in enumerate(kf.split(data), start=1):
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# 4. Predict rating for a specific user and movie
user_id = str(196)
movie_id = str(302)
pred = algo.predict(user_id, movie_id)
print(f"\nUser {user_id} predicted rating for Movie {movie_id}: {pred.est:.2f} (Actual: {pred.r_ui})")
```

---

## 🗺️ Roadmap & Enhancements

- [ ] **Top-$K$ Recommendation Pipeline:** Implement precision@k, recall@k, and generate Top-10 personalized movie lists per user.
- [ ] **Hybrid Recommender:** Combine collaborative filtering with content-based signals (genres, actors, directors) to resolve the Cold-Start problem.
- [ ] **Deep Learning Models:** Benchmark against Neural Collaborative Filtering (NCF) and autoencoders.
- [ ] **Interactive Dashboard:** Build a Streamlit / FastAPI web application enabling users to rate movies and view live recommendations.

---

## 📚 References & Acknowledgments

- **GroupLens Research:** [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- **Scikit-Surprise:** Nicolas Hug, *Surprise: A Python library for recommender systems*, [JOSS (2020)](https://joss.theoj.org/papers/10.21105/joss.02174).
- **Matrix Factorization Techniques:** Koren, Bell, & Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer (2009).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it for research, study, or commercial purposes.
