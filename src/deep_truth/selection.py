"""Validation selection rules shared by model searches."""

from deep_truth.config import Config


def selection_key(record: dict, config: Config) -> tuple:
    """Lower keys win; use validation scores followed by configured tie-breaks."""
    values = {**record["parameters"], **record["validation"]}
    key = []
    for rule in config.data["selection"]["tie_break"]:
        field = rule["field"]
        if field not in values:
            continue
        value = values[field]
        if field in {"macro_f1", "accuracy"}:
            value = round(value, config.comparison_decimals)
        if isinstance(value, list):
            value = tuple(value)
        key.append(-value if rule["order"] == "desc" else value)
    return tuple(key)

