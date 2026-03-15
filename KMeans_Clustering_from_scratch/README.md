# K-Means Clustering from Scratch using NumPy

Building K-Means clustering from the ground up using only NumPy and Matplotlib — no sklearn, no ML libraries. Applied to synthetic 2D data and real MNIST handwritten digit images.

---

## What this covers

- **K-Means algorithm** — random initialisation, assignment step, update step, convergence check
- **K-Means++ initialisation** — smarter centroid seeding compared against random init
- **Elbow method** — finding the optimal K by plotting inertia across K=1 to 10
- **MNIST experiment** — clustering 10,000 digit images and visualising what each cluster centre looks like

---

## Results

| Experiment | Result |
|---|---|
| 2D synthetic (3 clusters) | Converged in 4 iterations |
| Elbow method | Correctly identified K=3 |
| MNIST cluster centres | All 10 centres visually resemble digits |
| MNIST purity score | **> 70%** |

---

## Outputs

- Step-by-step convergence subplots showing centroids moving across iterations
- Side-by-side elbow curves for random init vs K-Means++
- MNIST cluster centres rendered as 28×28 images
- Cluster vs true digit confusion heatmap

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | All clustering logic |
| `matplotlib` | All visualisations |
| `keras.datasets` | Loading MNIST data only |

---

## How to Run

Open the notebook in Google Colab and run all cells top to bottom. No additional installs needed — NumPy and Matplotlib are available by default, and Keras comes pre-installed in Colab.

---

## File Structure

```
├── kmeans_from_scratch.ipynb   # Main notebook
└── README.md
```
