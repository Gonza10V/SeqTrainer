import pandas as pd
import pytest

from seqtrainer.models import PromoterActivityKNN, ScalarKNNRetriever, build_scalar_knn_retriever


def test_promoter_activity_knn_returns_ranked_nearest_sequences():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "label": [0.1, 0.4, 0.43, 1.0],
        }
    )

    retriever = PromoterActivityKNN(n_neighbors=2).fit(df)
    results = retriever.search(0.41)

    assert list(results["sequence"]) == ["CCCC", "GGGG"]
    assert list(results["rank"]) == [1, 2]
    assert results.loc[0, "distance"] == pytest.approx(0.01)
    assert results.loc[1, "distance"] == pytest.approx(0.02)


def test_top_k_overrides_default_neighbor_count():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "label": [0.1, 0.4, 0.43, 1.0],
        }
    )

    retriever = PromoterActivityKNN(n_neighbors=1).fit(df)
    results = retriever.search(0.41, top_k=3)

    assert len(results) == 3
    assert list(results["sequence"]) == ["CCCC", "GGGG", "AAAA"]


def test_factory_preserves_metadata_columns():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC"],
            "label": [0.1, 0.4],
            "part_id": ["p1", "p2"],
        }
    )

    retriever = build_scalar_knn_retriever(
        df,
        value_column="label",
        metadata_columns=["part_id"],
    )
    results = retriever.search(0.39, top_k=1)

    assert results.loc[0, "part_id"] == "p2"


def test_fit_drops_null_sequences_and_invalid_values_before_conversion():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", None, float("nan"), pd.NA, "CCCC", "GGGG"],
            "label": [0.1, 0.2, 0.3, 0.4, None, "not-a-number"],
        }
    )

    retriever = ScalarKNNRetriever().fit(df)

    assert list(retriever.records_["sequence"]) == ["AAAA"]
    assert list(retriever.records_["label"]) == [0.1]


@pytest.mark.parametrize("missing_sequence", [None, float("nan"), pd.NA])
def test_fit_rejects_data_with_only_null_sequences(missing_sequence):
    df = pd.DataFrame({"sequence": [missing_sequence], "label": [0.1]})

    with pytest.raises(ValueError, match="no valid records"):
        ScalarKNNRetriever().fit(df)


def test_save_and_load_round_trip(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA", "CCCC"], "label": [0.1, 0.4]})
    retriever = PromoterActivityKNN(n_neighbors=1).fit(df)

    model_path = retriever.save(tmp_path / "promoter_knn.pkl")
    loaded = PromoterActivityKNN.load(model_path)

    assert loaded.search(0.39, top_k=1).loc[0, "sequence"] == "CCCC"


def test_search_before_fit_raises():
    with pytest.raises(RuntimeError, match="Fit the KNN retriever"):
        ScalarKNNRetriever().search(0.4)


def test_ties_are_deterministic_even_when_they_exceed_top_k():
    retriever = ScalarKNNRetriever().fit(
        pd.DataFrame({"sequence": ["AAAA", "CCCC", "GGGG"], "label": [0.4, 0.6, 0.4]})
    )

    assert list(retriever.search(0.5, top_k=2)["sequence"]) == ["AAAA", "GGGG"]


@pytest.mark.parametrize("query", [float("nan"), float("inf"), "not-a-number"])
def test_search_rejects_non_finite_or_non_numeric_query(query):
    retriever = ScalarKNNRetriever().fit(pd.DataFrame({"sequence": ["AAAA"], "label": [0.4]}))

    with pytest.raises(ValueError, match="finite"):
        retriever.search(query)
