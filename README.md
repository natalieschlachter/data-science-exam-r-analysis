# Data Science Exam 

This repository contains my individual written exam submission for a course **Data Science: Data Driven Decision Making**, plus the **R implementation** used for the analyses (extracted from the appendix of the report).

## Contents

- **Part 1 — Supervised Learning**: Logistic Regression, SVM (linear & RBF), Classification Trees (tree & rpart), Random Forest, k-NN  
- **Part 2 — Unsupervised Learning**: Hierarchical clustering, K-means + elbow method  
- **Part 3 — Visualization**: PCA + 2D/3D plots, Multidimensional Scaling (MDS) with different distance metrics

## Repository structure

```
.
├── report/
│   └── DataScienceExam_167055.pdf
├── src/
│   ├── part1_supervised_learning.R
│   ├── part2_unsupervised_learning.R
│   └── part3_visualization.R
└── data/
    └── (place datasets here — not included)
```

## Datasets

The original code referenced local file paths. In this repo the scripts expect the following files in `data/`:

- `student.txt`
- `studentadditional.txt`
- `wine.txt`
- `seeds.txt`

If you have these datasets, place them in the `data/` folder with the exact names above.

## How to run

1. Install R (and optionally RStudio).
2. Open a script from `src/` and run it top-to-bottom.
3. Packages are installed/loaded inside the scripts (you may want to comment out `install.packages(...)` once installed).

> Note: The scripts were extracted from an R Markdown appendix, so they include some `summary()/str()/print()` calls used for reporting.

## Citation

If you reference this work, please cite the PDF in `report/`.

