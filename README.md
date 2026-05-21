# Karpathy — Zero to Hero Series Projects

## Project Vision
This repository is a long-term learning and productization journey based on Andrej Karpathy’s **Neural Networks: Zero to Hero** series.

The goal is to:
1. Recreate and upload projects from **all 7 lectures** in an organized, easy-to-follow format.
2. Preserve the educational progression from fundamentals to more advanced model-building concepts.
3. Wrap the full collection into a **single plug-and-play UI** where anyone can experiment with different model types, activation functions, hyperparameters, and datasets.

In short: this repo starts as a structured learning archive and evolves into an interactive playground for neural network experimentation.

---

## What Will Be Uploaded
The plan is to include projects corresponding to all seven lectures in the Zero to Hero sequence.

Each lecture folder/project will aim to include:
- A clear summary of the lecture’s main concepts.
- The implementation work completed from that lecture.
- Notes on key design decisions and learnings.
- Reproducible steps for running or exploring that specific project.
- Inputs/outputs or observations that show how training behavior changes.

This keeps each lecture self-contained while still contributing to the larger unified system.

---

## End Goal: Unified Plug-and-Play UI
After all lecture projects are in place, the repository will be wrapped into a single interface that makes experimentation simple.

### Core UI Goals
- **Model selection:** Choose among available model types implemented across the 7 projects.
- **Activation function selection:** Swap activation functions and compare training behavior.
- **Dataset upload:** Bring your own dataset and run training without rewriting project internals.
- **Training controls:** Adjust important training settings (for example epochs, learning rate, batch size, and other configurable options available in the integrated models).
- **Progress visualization:** Observe training progress, key metrics, and performance trends over time.
- **Results comparison:** Compare runs across different configurations to understand trade-offs.

### Why This Matters
The UI is intended to remove setup friction and make the series more interactive.
Instead of manually jumping between separate notebooks/scripts, users can explore concepts through one consistent workflow and quickly see how architectural choices affect outcomes.

---

## Intended User Experience
A typical flow for a user will be:
1. Open the app.
2. Select a model derived from one of the lecture projects.
3. Choose activation/training options.
4. Upload a dataset of their choice.
5. Start training and monitor progress.
6. Compare results with previous runs or alternate settings.

This is designed for learners, tinkerers, and anyone who wants to understand neural network behavior by experimentation.

---

## Repository Evolution Plan
This repository is expected to evolve in phases:

### Phase 1 — Lecture-by-Lecture Uploads
- Add all 7 lecture projects.
- Keep project boundaries clear.
- Ensure each project remains understandable on its own.

### Phase 2 — Standardization Layer
- Normalize interfaces between projects.
- Align configuration handling so projects can be controlled in a consistent way.
- Prepare each project for UI integration.

### Phase 3 — Unified UI Integration
- Build one interface that can orchestrate model selection, configuration, dataset input, and training feedback.
- Expose comparisons and insights in a learner-friendly format.

### Phase 4 — Refinement
- Improve usability and clarity.
- Expand supported workflows and dataset handling.
- Polish documentation and examples for broader use.

---

## Project Principles
- **Education first:** Keep the original learning intent visible.
- **Clarity over complexity:** Make experimentation accessible.
- **Modularity:** Treat each lecture project as a composable unit.
- **Reproducibility:** Make it easy to rerun and compare experiments.

---

## Current Status
This README defines the roadmap and intended final product direction.
As lecture projects are uploaded, this document will be updated to reflect concrete progress, integration status, and usage instructions.

---

## Future Documentation Additions
As implementation progresses, this README can be expanded with:
- A progress tracker for all 7 lectures.
- Setup and run instructions for each project.
- Unified UI usage guide.
- Supported dataset format requirements.
- Known limitations and planned improvements.

