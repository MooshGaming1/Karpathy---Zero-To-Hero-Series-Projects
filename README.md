# Karpathy Zero-to-Hero (with Custom Adjustments)

This repository contains my progress through Andrej Karpathy's Neural Networks: Zero to Hero series. My overarching goal is to complete every single project in the series, but with a twist: I am adjusting and enhancing each project in some specific way to deepen my understanding.

## 1. Autograd (Micrograd but for Tensor inputs)
**File:** `Autograd.py`

While the original Micrograd builds an autograd engine for scalar values, I extended it to work with full **Tensor** inputs using `numpy`. This implementation supports:
* Matrix multiplication (`@`) and broadcasting (`_unbroadcast`).
* Elementwise operations (`+`, `-`, `*`, `/`, `**`).
* Reductions (`sum`, `mean`).
* Activations (`relu`, `tanh`, `sigmoid`, `exp`, `log`).

## 2. Trigram Model (Makemore Pt 1)
**File:** `trigram.py`

The original Makemore Part 1 implements a bigram character-level language model. I adjusted it in two major ways:
* **Trigram Model:** Instead of predicting the next character based on a single previous character, it uses a sliding window of two previous characters (a trigram) to predict the third.
* **Batch SGD:** Instead of computing the loss on the entire dataset at once, the training loop uses Batch Stochastic Gradient Descent (SGD), randomly sampling mini-batches of 64 examples per iteration using PyTorch.

## 3. MLP (Makemore Pt 2)
*Upcoming* - Adjustments TBD.

## 4. RNN & GRU (Makemore Pt 3)
*Upcoming* - Adjustments TBD.

## 5. nanoGPT & minGPT
*Upcoming* - Adjustments TBD.
