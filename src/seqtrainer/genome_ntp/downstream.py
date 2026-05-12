"""Downstream protocol stubs for promoter tasks."""

PROTOCOL = {
    "classification": ["scratch", "finetune", "frozen_encoder"],
    "classification_metrics": ["auroc", "auprc", "f1"],
    "activity_metrics": ["pearson", "spearman", "rmse"],
}
