# Movie Recommendation System
Matrix Factorization with PyTorch | MovieLens 100K

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![License](https://img.shields.io/badge/license-MIT-green)

A collaborative filtering recommender built from scratch in PyTorch. The model learns user and movie embeddings through matrix factorization and uses them to predict ratings and generate top-N recommendations.

---

## Results

| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 5     | 0.8976    | 0.9277    |
| 10    | 0.8108    | 0.9073    |
| 15    | 0.7457    | 0.9045    |
| 20    | 0.6991    | 0.9055    |

Test RMSE stabilizes around 0.90 after epoch 15. The ~0.20 gap between train and test at epoch 20 is expected given the high sparsity of the dataset (~93–94%).

---

## How It Works

Matrix factorization decomposes the user-movie rating matrix into two lower-dimensional embedding matrices — one for users, one for movies. Each user and movie is represented as a vector of **latent factors** (learned features). The predicted rating for a user-movie pair is the dot product of their respective embedding vectors, plus learned bias terms.

```
predicted_rating = sigmoid(user_embedding · movie_embedding + user_bias + movie_bias)
```

The model is trained end-to-end with backpropagation, minimizing Mean Squared Error (MSE) loss between predicted and actual ratings.

---


## Dataset

**MovieLens 100K** — [grouplens.org](https://grouplens.org/datasets/movielens/100k/)

| Stat | Value |
|------|-------|
| Total ratings | 100,000 |
| Unique users | 943 |
| Unique movies | 1,682 |
| Rating scale | 1–5 stars |
| Train / Test split | 80,000 / 20,000 |
| Matrix sparsity | ~93–94% |

Ratings are normalized to [0, 1] during training and converted back to the 1–5 scale for evaluation.

---

## Model Architecture

```python
MatrixFactorization(
  (user_embeddings):  Embedding(943,  64)   # latent user factors
  (movie_embeddings): Embedding(1682, 64)   # latent movie factors
  (user_bias):        Embedding(943,  1)    # per-user bias
  (movie_bias):       Embedding(1682, 1)    # per-movie bias
  (dropout):          Dropout(p=0.1)
)
```

Design decisions:
- Xavier initialization on embedding weights to keep gradients stable early in training
- Bias terms per user and movie to account for individual rating tendencies (e.g. a user who consistently rates low)
- Dropout (0.1) on embeddings to limit overfitting on sparse data
- Sigmoid on output to keep predictions within the normalized rating range

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Latent factors | 64 |
| Epochs | 20 |
| Batch size | 256 |
| Optimizer | Adam (lr=0.001) |
| Weight decay | 1e-5 |
| Loss function | MSE |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

**2. Install dependencies**
```bash
pip3 install torch torchvision pandas numpy scikit-learn
```

**3. Download the dataset**

Get MovieLens 100K from [grouplens.org](https://grouplens.org/datasets/movielens/100k/) and unzip it into a `ml-100k/` folder in the project root.

**4. Run**
```bash
python3 movie_rec.py
```

---

## Generating Recommendations

```python
# Get top 10 movie recommendations for user ID 1
recommendations = recommend_for_user(user_id=1, n=10)
print(recommendations)
```

Example output:
```
   movie_id                          title
0       357              One Flew Over ...
1       427               They Made Me ...
2       318    Schindler's List (1993)
...
```

---

## Potential Next Steps

- Neural Collaborative Filtering (NCF) — pass concatenated embeddings through linear layers instead of a dot product to model non-linear interactions
- Hybrid model — combine embedding scores with content-based features like genre
- Learning rate scheduler — decay LR on plateau for additional accuracy gains
- Early stopping — stop training when test loss stops improving
- Streamlit UI — simple front end for querying recommendations interactively
- Implicit feedback — replace explicit star ratings with watch history or click data

---

## Stack

- [PyTorch](https://pytorch.org/) — model, training loop
- [pandas](https://pandas.pydata.org/) — data loading and preprocessing
- [NumPy](https://numpy.org/) — matrix operations
- [scikit-learn](https://scikit-learn.org/) — train/test split

---

## License

MIT

---

## Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- [PyTorch](https://pytorch.org/) team for the framework
