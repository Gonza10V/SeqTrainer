import importlib
import warnings



def test_preprocessing_wrapper_emits_deprecation_warning():
    import seqtrainer.preprocessing as preprocessing

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        preprocessing.pad_sequence("ACGT", 6)

    assert any("seqtrainer.preprocessing" in str(w.message) for w in caught)


def test_dataset_builder_wrapper_emits_deprecation_warning():
    import seqtrainer.dataset_builder as dataset_builder

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        dataset_builder.find_possible_y_uris("dummy.xml")

    assert any("seqtrainer.dataset_builder" in str(w.message) for w in caught)


def test_gnn_module_import_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("seqtrainer.gnn")
        importlib.reload(module)

    assert any("seqtrainer.gnn" in str(w.message) for w in caught)


def test_legacy_cli_alias_emits_deprecation_warning(capsys):
    from seqtrainer.cli.main import main

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        code = main([
            "build-dataset",
            "data/sbol_data/sample_design_0.xml",
            "--recipe",
            "local-sbol-sequence-only",
        ])

    assert code == 0
    assert any("seqtrainer build-dataset" in str(w.message) for w in caught)

    out = capsys.readouterr().out
    assert "sequence" in out
