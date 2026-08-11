"""
Tests del registro de algoritmos NLP en hyperparams.py.
"""

import pytest

from app.core_ml.hyperparams import (
    get_all_nlp_algorithms,
    get_nlp_algorithm_info,
    get_nlp_default_hyperparams,
    NLP_ALGORITHMS,
)


def test_nlp_algorithms_registry_has_six_models():
    assert set(NLP_ALGORITHMS.keys()) == {
        "distilbert",
        "bert-base",
        "bert-multilingual",
        "roberta",
        "xlm-roberta",
        "biobert",
    }


def test_nlp_algorithm_hf_models():
    assert NLP_ALGORITHMS["distilbert"]["hf_model"] == "distilbert-base-uncased"
    assert NLP_ALGORITHMS["bert-base"]["hf_model"] == "bert-base-uncased"
    assert NLP_ALGORITHMS["xlm-roberta"]["hf_model"] == "xlm-roberta-base"


def test_nlp_task_types():
    for algo in NLP_ALGORITHMS.values():
        assert "text_classification" in algo["task_types"]
        assert "token_classification" in algo["task_types"]
        assert "sentiment_analysis" in algo["task_types"]


def test_get_nlp_algorithm_info_valid():
    info = get_nlp_algorithm_info("distilbert")
    assert info["display_name"] == "DistilBERT"
    assert info["framework"] == "huggingface"


def test_get_nlp_algorithm_info_invalid():
    with pytest.raises(ValueError):
        get_nlp_algorithm_info("nonexistent-algo")


def test_get_all_nlp_algorithms_shape():
    result = get_all_nlp_algorithms()
    assert len(result) == 6
    first = result[0]
    assert {"id", "display_name", "framework", "task_types", "hyperparams"} <= set(first.keys())
    # hiperparámetros requeridos
    hp_names = {hp["name"] for hp in first["hyperparams"]}
    assert {"epochs", "learning_rate", "batch_size", "weight_decay", "warmup_ratio", "max_seq_length"} <= hp_names


def test_get_nlp_default_hyperparams():
    defaults = get_nlp_default_hyperparams("distilbert")
    assert defaults["epochs"] == 3
    assert defaults["learning_rate"] == 0.00002
    assert defaults["max_seq_length"] == 128
