# RNN and LSTM from Scratch using NumPy

Character-level language modelling on Shakespeare text, built with nothing but NumPy. No PyTorch, no TensorFlow, no ML libraries.

---

## What it does

Trains a Vanilla RNN and an LSTM on the first 50,000 characters of the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, then compares them side by side — loss curves, generated text samples, and gradient norm plots that visually prove why the LSTM is better.

## Structure

| Section | Contents |
|---|---|
| **1 — Math** | LaTeX derivations of RNN and LSTM equations, gate diagram, vanishing gradient proof |
| **2 — Implementation** | `myRNN` and `LSTM` classes written as explicit NumPy matrix operations |
| **3 — Training** | Loss curves for both models on the same plot |
| **4 — Results** | Generated text samples + gradient norm comparison |
| **Final cell** | Evidence-backed explanation of why LSTM solves vanishing gradients |

## Results

Both models train from ~4.1 nats (random chance) and decrease over 5,000 iterations:

- **RNN** converges to ~2.42 nats
- **LSTM** converges to ~1.85 nats (~24% better)

The gradient norm plots show the RNN signal decaying toward the earliest timesteps, while the LSTM stays stable — the additive cell-state update at work.

## Running it

Open in Google Colab and run all cells top to bottom. No setup needed beyond a standard Colab environment.


## Dependencies

Only the Python standard library + NumPy + Matplotlib. The Shakespeare dataset is fetched automatically at runtime.

## Key concepts covered

- Backpropagation through time (BPTT)
- Vanishing gradient problem
- LSTM gates: forget, input, cell, output
- Gradient clipping
- AdaGrad optimizer
- Character-level language modelling
