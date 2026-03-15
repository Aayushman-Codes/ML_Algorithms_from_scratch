# Micrograd — Backpropagation from Scratch

A from-scratch implementation of reverse-mode automatic differentiation and a working neural network, built using only Python and NumPy. No PyTorch, no TensorFlow, no autograd library.

Inspired by Andrej Karpathy's micrograd, but independently implemented with a focus on teaching.

---

## What's inside

The notebook walks through five sections:

1. **Theory** — computation graphs, reverse-mode autodiff, and the chain rule with LaTeX derivations
2. **Value engine** — a scalar `Value` class with operator overloading and a full `backward()` implementation
3. **Graph visualisation** — renders the computation graph with gradients using Graphviz
4. **MLP implementation** — `Neuron → Layer → MLP` built entirely on top of `Value`
5. **Training** — trains on `make_moons` (2D binary classification) with loss curves and decision boundary plots

Final accuracy: **>95%** on the make_moons dataset.

---

## Key concepts implemented

- Forward pass via operator overloading (`+`, `*`, `**`, `-`, `/`)
- Activation functions: `tanh`, `relu`, `exp`
- Reverse-mode autodiff via iterative topological sort
- He initialisation for ReLU networks
- Mini-batch SGD with step learning rate decay
- SVM hinge loss (`max(0, 1 - y·ŷ)`) built from `relu`

---

## Running it

Open directly in Google Colab:

1. Upload `micrograd_backprop.ipynb`
2. Runtime → Run all

Dependencies are all available in the default Colab environment (`graphviz`, `sklearn`, `matplotlib`, `numpy`).

---

## Results

| | |
|---|---|
| Dataset | `make_moons`, 200 points, noise=0.15 |
| Architecture | 2 → 32 (ReLU) → 32 (ReLU) → 1 (tanh) |
| Parameters | 1,185 |
| Training | Mini-batch SGD, batch size 32, 400 epochs |
| Final accuracy | >95% |

---

## The core insight

Every arithmetic operation on a `Value` node records itself in a graph. Calling `.backward()` walks that graph in reverse topological order and applies the chain rule at each node. This is exactly what PyTorch's autograd does — just at tensor scale instead of scalar scale.
