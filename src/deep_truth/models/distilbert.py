"""Construct pretrained DistilBERT with the shared six-label classification head."""

from transformers import DistilBertForSequenceClassification

from deep_truth.config import Config

RUNS = ("distilbert_frozen", "distilbert_finetuned")


def configure_encoder(model: DistilBertForSequenceClassification, run_name: str):
    if run_name not in RUNS:
        raise ValueError(f"Unknown DistilBERT run: {run_name}")
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = run_name == "distilbert_finetuned"
    return model


def build_distilbert_model(config: Config, run_name: str):
    if run_name not in RUNS:
        raise ValueError(f"Unknown DistilBERT run: {run_name}")
    settings = config.data["distilbert"]
    model = DistilBertForSequenceClassification.from_pretrained(
        settings["model_name"],
        num_labels=len(config.labels),
        problem_type=settings["problem_type"],
        id2label=dict(enumerate(config.labels)),
        label2id=config.label_to_index,
    )
    return configure_encoder(model, run_name)
