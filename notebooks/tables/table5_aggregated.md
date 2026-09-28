## Tabella 5 — Aggregazione same/cross-dataset

| loss | split | auc_mean | auc_std | eer_mean | eer_std |
|---|---|---|---|---|---|
| bce | iam_to_iam | — | — | — | — |
| contrastive | iam_to_iam | — | — | — | — |
| triplet | iam_to_iam | — | — | — | — |
| bce | iam_to_rimes | — | — | — | — |
| contrastive | iam_to_rimes | — | — | — | — |
| triplet | iam_to_rimes | — | — | — | — |
| bce | rimes_to_iam | — | — | — | — |
| contrastive | rimes_to_iam | — | — | — | — |
| triplet | rimes_to_iam | — | — | — | — |
| bce | rimes_to_rimes | — | — | — | — |
| contrastive | rimes_to_rimes | — | — | — | — |
| triplet | rimes_to_rimes | — | — | — | — |

### Combinazioni incomplete
- **bce/iam_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **contrastive/iam_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, resnet34
- **triplet/iam_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **bce/iam_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **contrastive/iam_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, resnet34
- **triplet/iam_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **bce/rimes_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **contrastive/rimes_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, resnet34
- **triplet/rimes_to_iam**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **bce/rimes_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
- **contrastive/rimes_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, resnet34
- **triplet/rimes_to_rimes**: backbone mancanti -> efficientnet_b0, efficientnet_b1, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet34
