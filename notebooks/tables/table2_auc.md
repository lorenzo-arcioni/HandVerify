## Tabella 2 — AUC per backbone x loss

#### Split: `iam_to_iam`

| backbone | bce | contrastive | triplet |
|---|---|---|---|
| efficientnet_b0 | 0.961 | 0.991 | 0.925 |
| efficientnet_b1 | 0.988 | 0.988 | 0.946 |
| mobilenet_v3_large | 0.952 | 0.995 | 0.871 |
| mobilenet_v3_small | 0.960 | 0.992 | 0.914 |
| resnet18 | 0.963 | 0.990 | 0.947 |
| resnet34 | 0.910 | 0.985 | 0.938 |

#### Split: `iam_to_rimes`

| backbone | bce | contrastive | triplet |
|---|---|---|---|
| efficientnet_b0 | 0.631 | 0.756 | 0.683 |
| efficientnet_b1 | 0.655 | 0.772 | 0.738 |
| mobilenet_v3_large | 0.528 ⚠️ | 0.696 | 0.666 |
| mobilenet_v3_small | 0.516 ⚠️ | 0.702 | 0.657 |
| resnet18 | 0.717 | 0.764 | 0.720 |
| resnet34 | 0.679 | 0.736 | 0.726 |

#### Split: `rimes_to_iam`

| backbone | bce | contrastive | triplet |
|---|---|---|---|
| efficientnet_b0 | 0.674 | 0.967 | 0.905 |
| efficientnet_b1 | 0.775 | 0.954 | 0.944 |
| mobilenet_v3_large | 0.814 | 0.965 | 0.807 |
| mobilenet_v3_small | 0.540 ⚠️ | 0.939 | 0.853 |
| resnet18 | 0.888 | 0.943 | 0.852 |
| resnet34 | 0.598 | 0.932 | 0.853 |

#### Split: `rimes_to_rimes`

| backbone | bce | contrastive | triplet |
|---|---|---|---|
| efficientnet_b0 | 0.886 | 0.924 | 0.891 |
| efficientnet_b1 | 0.468 ⚠️ | 0.932 | 0.895 |
| mobilenet_v3_large | 0.917 | 0.925 | 0.862 |
| mobilenet_v3_small | 0.569 | 0.916 | 0.897 |
| resnet18 | 0.821 | 0.934 | 0.882 |
| resnet34 | 0.744 | 0.931 | 0.900 |
