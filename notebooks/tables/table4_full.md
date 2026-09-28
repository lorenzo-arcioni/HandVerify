## Tabella 4 — Metriche complete per esperimento

| backbone | loss | split | auc | eer | d_prime | accuracy | precision | recall | f1 | mu_genuine | mu_impostor | sigma_genuine | sigma_impostor | gar_far1pct | gar_far0.1pct | far0.1pct_valid | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | contrastive | iam_to_iam | 0.980 | 6.9 | 3.471 | 0.931 | 0.931 | 0.931 | 0.931 | 0.835 | 0.014 | 0.174 | 0.286 | 66.9 | 19.8 | yes |  |
| mobilenet_v3_small | contrastive | iam_to_rimes | 0.783 | 28.7 | 0.921 | 0.713 | 0.713 | 0.713 | 0.713 | 0.844 | 0.651 | 0.171 | 0.242 | 14.9 | 4.4 | yes |  |
| mobilenet_v3_small | contrastive | rimes_to_iam | 0.864 | 21.8 | 1.520 | 0.782 | 0.782 | 0.782 | 0.782 | 0.744 | 0.223 | 0.280 | 0.395 | 27.9 | 9.7 | yes |  |
| mobilenet_v3_small | contrastive | rimes_to_rimes | 0.987 | 4.7 | 3.497 | 0.953 | 0.953 | 0.953 | 0.953 | 0.886 | 0.012 | 0.135 | 0.326 | 78.6 | 34.6 | yes |  |
| resnet18 | contrastive | iam_to_iam | 0.977 | 7.4 | 3.208 | 0.926 | 0.926 | 0.926 | 0.926 | 0.842 | 0.005 | 0.175 | 0.325 | 61.5 | 18.3 | yes |  |
| resnet18 | contrastive | iam_to_rimes | 0.824 | 24.6 | 1.239 | 0.754 | 0.754 | 0.754 | 0.754 | 0.749 | 0.394 | 0.257 | 0.312 | 16.5 | 4.2 | yes |  |
| resnet18 | contrastive | rimes_to_iam | 0.941 | 11.9 | 2.191 | 0.881 | 0.881 | 0.881 | 0.881 | 0.845 | 0.212 | 0.186 | 0.363 | 37.0 | 9.5 | yes |  |
| resnet18 | contrastive | rimes_to_rimes | 0.986 | 4.7 | 3.553 | 0.953 | 0.953 | 0.953 | 0.953 | 0.892 | 0.014 | 0.130 | 0.325 | 72.1 | 27.9 | yes |  |
