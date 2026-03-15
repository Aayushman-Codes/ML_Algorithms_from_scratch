# CNN from Scratch using NumPy

A convolutional neural network built entirely in NumPy — no PyTorch, no TensorFlow, no autograd. Every forward pass, backward pass, and weight update is implemented by hand.

Trained and evaluated on MNIST handwritten digits. Achieves **~93% test accuracy** after 5 epochs on 10,000 training samples.

---

## Architecture

```
Input (28×28×1)
  → Conv2D(8 filters, 3×3, pad=1)  → ReLU
  → MaxPool2D(2×2)
  → Conv2D(16 filters, 3×3, pad=1) → ReLU
  → MaxPool2D(2×2)
  → Flatten
  → Dense(128)                      → ReLU
  → Dense(10)                       → Softmax
```

## What's implemented

- **Conv2D** — im2col/col2im for efficient forward and backward passes
- **ReLU** — with binary mask for backprop
- **MaxPool2D** — with max-position mask for gradient routing
- **Flatten, Dense** — standard layers with full backprop
- **Softmax + Cross-Entropy** — fused for numerical stability
- **Mini-batch SGD** — with configurable batch size and learning rate
- **Numerical gradient check** — finite-difference verification of backprop correctness

## Results

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 5) | ~93% |
| Test accuracy | ~93% |
| Training samples | 10,000 |
| Epochs | 5 |
| Batch size | 32 |

**Top misclassifications:** 4→9 (29), 7→9 (16), 2→7 (8) — all structurally reasonable given handwriting variation.

## Usage

Open `CNN_from_Scratch_NumPy.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells top to bottom. The only external dependency beyond NumPy is `keras.datasets.mnist`, used solely to download the data.

```
pip install numpy matplotlib tensorflow  # tensorflow for keras.datasets only
```

## Notebook structure

| Section | Contents |
|---------|----------|
| 1 — Math & Theory | Convolution, im2col worked example, backprop derivations, gradient flow diagram, SGD vs Adam |
| 2 — Implementation | Class-based layers with forward + backward |
| 3 — Training | Mini-batch SGD, loss/accuracy curves |
| 4 — Results | Test accuracy, sample predictions, confusion matrix |

## License

MIT
