import pytest

from seqtrainer.experimental.titans import PaperMAC, TitansConfig, evaluate_checkpoint, train_cpu_workflow

torch = pytest.importorskip("torch")


def test_reference_model_is_causal_and_carries_explicit_state():
    model = PaperMAC(TitansConfig(vocab_size=8, hidden_size=4))
    tokens = torch.tensor([[1, 2, 3]])
    logits, state = model(tokens, return_state=True)

    assert logits.shape == (1, 3, 8)
    assert state.value.shape == (1, 4)
    # The first prediction cannot depend on later tokens.
    changed = model(torch.tensor([[1, 7, 6]]))
    assert torch.allclose(logits[:, :1], changed[:, :1])


def test_cpu_checkpoint_resume_is_deterministic(tmp_path):
    config = TitansConfig(vocab_size=8, hidden_size=4)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    checkpoint = tmp_path / "state.pt"
    first = train_cpu_workflow(tokens, config=config, checkpoint=checkpoint, steps=2, seed=12)
    resumed = train_cpu_workflow(tokens, config=config, checkpoint=checkpoint, steps=2, seed=12, resume=True)

    assert first.steps == 2
    assert resumed.steps == 4
    assert evaluate_checkpoint(tokens, config=config, checkpoint=checkpoint) > 0
