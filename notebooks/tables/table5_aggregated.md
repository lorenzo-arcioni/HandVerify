## Tabella 5 — Aggregazione same/cross-dataset

| loss | split | auc_mean | auc_std | eer_mean | eer_std |
|---|---|---|---|---|---|
| bce | iam_to_iam | 0.956 | 0.025 | 9.0 | 4.0 |
| contrastive | iam_to_iam | 0.990 | 0.004 | 4.9 | 1.3 |
| triplet | iam_to_iam | 0.924 | 0.029 | 15.3 | 3.6 |
| bce | iam_to_rimes | 0.621 | 0.082 | 40.2 | 6.9 |
| contrastive | iam_to_rimes | 0.738 | 0.032 | 32.2 | 2.4 |
| triplet | iam_to_rimes | 0.699 | 0.034 | 35.5 | 2.5 |
| bce | rimes_to_iam | 0.715 | 0.133 | 32.9 | 12.8 |
| contrastive | rimes_to_iam | 0.950 | 0.014 | 11.7 | 2.3 |
| triplet | rimes_to_iam | 0.869 | 0.048 | 20.4 | 4.9 |
| bce | rimes_to_rimes | 0.734 | 0.180 | 30.9 | 15.3 |
| contrastive | rimes_to_rimes | 0.927 | 0.007 | 12.9 | 1.1 |
| triplet | rimes_to_rimes | 0.888 | 0.014 | 18.3 | 1.1 |
