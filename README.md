# Jurkat-Cell-Cycle-Classification

Code and results for "Quantifying the Impact of Dataset Quality on Convolutional Neural Network Performance in Cell-Cycle Image Classification" (Nayoung Kim). A CNN classifying seven cell-cycle stages of Jurkat cells (BBBC048) was trained under four data-quality conditions (clean or degraded images, correct or noisy labels), 30 seeds each, 120 runs in total.

## Contents

- `Nayoung_Kim_Research_paper.ipynb`: the complete experiment, executed, with its outputs. It downloads BBBC048, builds the dataset and the shared test set, trains and evaluates all 120 runs, runs the collapse analysis and draws the figures. Open it in Google Colab with the badge at the top of the notebook.
- `results/`: the files the notebook saved.
  - `all_runs.csv`: accuracy, macro precision, recall and F1, most-predicted class, its share of test predictions and collapse status for each of the 120 runs.
  - `summary_by_condition.csv`: mean and standard deviation of each metric by condition (Table 2 of the paper).
  - `collapse_summary.csv`: collapsed runs by condition with 95% Wilson intervals (Table 3).
  - `classification_report_<condition>_seed42.csv`: per-class precision, recall and F1 for seed 42.
  - `confusion_matrices_figure4.csv`: confusion matrices of the three runs shown in Figure 4.
  - `run_info.json`: GPU, TensorFlow version and seed list.
  - `*.png`: the figures as the notebook saved them.
- `make_paper_figures.py`: draws Figures 2, 3 and 4 of the paper from `results/` (`python make_paper_figures.py`; needs numpy, pandas, matplotlib and seaborn).
- `paper_figures/`: Figures 1 to 4 as they appear in the paper.

## Data

BBBC048 (Cell Cycle Jurkat Cells) is available from the Broad Bioimage Benchmark Collection, https://bbbc.broadinstitute.org/BBBC048 (Ljosa et al., 2012; Eulenberg et al., 2017), under CC BY-NC-SA 3.0.
