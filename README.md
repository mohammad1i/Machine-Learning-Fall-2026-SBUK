# Machine Learning: Course Assignments

Code and reports for the assignments of the Machine Learning course, Department of Computer Engineering, Shahid Bahonar University of Kerman (SBUK).

## Assignments

| # | Folder | Topic | Main content |
|--:|--------|-------|--------------|
| 1 | [`ml-hw01-concept-learning`](ml-hw01-concept-learning) | Concept learning on EnjoySport | Find-S from scratch, a reproducible sample-complexity experiment, Candidate-Elimination traced by hand |

Each new assignment is added as a new folder next to the existing ones, and a new row is added to this table.

## Repository layout

```
Machine-Learning-Fall-2026-SBUK/
├── README.md                      # this file: index of all assignments
├── .gitignore                     # shared by all assignments
├── ml-hw01-concept-learning/
│   ├── README.md
│   ├── run_all.py
│   ├── output.txt
│   ├── exercise_2_10_plot.png
│   └── requirements.txt
└── ml-hw02-.../                   # next assignment
```

## Conventions

- **Folder names:** `ml-hwNN-short-topic`, all lowercase, words separated by hyphens, no spaces (for example `ml-hw02-linear-regression`).
- **Self-contained folders:** every assignment has its own `README.md`, `requirements.txt` and run script, so it can be run without the other folders.
- **Reproducibility:** experiments use recorded random seeds, and each assignment's README says how to reproduce its results.

## Running an assignment

```bash
git clone https://github.com/mohammad1i/Machine-Learning-Fall-2026-SBUK.git
cd Machine-Learning-Fall-2026-SBUK/ml-hw01-concept-learning
pip install -r requirements.txt
python run_all.py
```

Replace `ml-hw01-concept-learning` with the folder of the assignment you want to run.

## Author

GitHub: [@mohammad1i](https://github.com/mohammad1i)

*This repository contains student homework solutions, shared for reference. If you are taking this course, please write your own solutions.*
