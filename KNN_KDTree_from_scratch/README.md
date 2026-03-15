# K-Nearest Neighbours with KD-Tree from Scratch

A complete implementation of KNN classification built entirely from scratch using NumPy — no sklearn, no ML libraries. Covers brute-force KNN and a full KD-Tree with backtracking search, tested on MNIST handwritten digits.

---

## Contents

| Notebook | Description |
|---|---|
| `KNN_KDTree_from_Scratch.ipynb` | Full implementation, benchmarks, and visualisations |

---

## What's Inside

**Part 1 — Naive KNN (Brute Force)**
- Euclidean distance computed via vectorised NumPy
- Majority vote with deterministic tie-breaking
- Tested on 1,000 MNIST samples — achieves 95%+ accuracy at K=5

**Part 2 — KD-Tree KNN**
- Full KD-Tree built from scratch: recursive median splitting on highest-variance axis
- Nearest-neighbour search with backtracking and hypersphere pruning
- Identical prediction interface to naive KNN

**Part 3 — Comparison**
- Wall-clock query time vs dataset size (100 to 25,000 points) — crossover point clearly visible at ~2,500 points on 2D data
- Accuracy vs K (k=1,3,5,7,9,11) for both methods — curves overlap exactly, confirming KD-Tree correctness

**Part 4 — Decision Boundary**
- 2D synthetic dataset (3 classes, 300 points)
- KNN decision boundary plotted as a colour mesh using the KD-Tree implementation

---

## Results

| | Naive KNN | KD-Tree KNN |
|---|---|---|
| **Build time** | O(1) | O(n log n) |
| **Query time** | O(n·d) | O(log n·d) avg |
| **MNIST accuracy (K=5)** | 95%+ | identical |
| **Predictions match** | — | ✓ all K values |

---

## Requirements

```
numpy
matplotlib
tensorflow  # used only for keras.datasets.mnist — data loading only
```

Install with:

```bash
pip install numpy matplotlib tensorflow
```

---

## Usage

Open directly in Google Colab:

1. Upload `KNN_KDTree_from_Scratch.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells top to bottom (`Runtime → Run all`)

Or locally with Jupyter:

```bash
jupyter notebook KNN_KDTree_from_Scratch.ipynb
```

---

## Key Implementation Notes

- **KD-Tree tie-breaking:** each node stores its original training index (`orig_idx`). The heap key is `(-dist², orig_idx, label)` so equidistant candidates are evicted in the same order as `np.argsort`, guaranteeing identical predictions to brute force.
- **Voting tie-breaking:** when two classes have equal vote counts, the class whose nearest member is closest to the query wins — same rule applied in both implementations.
- **Speed benchmark:** run on 2D synthetic data (not raw MNIST) because KD-Tree pruning degrades in high dimensions — the curse of dimensionality makes the tree visit nearly every node at 784-d.

---

## License

MIT
