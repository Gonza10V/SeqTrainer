from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from seqtrainer.torch.titans_paper_mac_stage_c.resume_archive import (
    ResumeArchiveManager,
    validate_resume_archive,
    write_resume_state,
)


CONTRACT = {"experiment": "v2", "evaluator_commit": "abc", "study_version": "v2"}


def _registry(path: Path, stage: str) -> Path:
    root = path / stage
    root.mkdir()
    write_resume_state(root, {"immutable_contract": CONTRACT, "stage": stage})
    (root / "payload.txt").write_text(stage)
    return root


def test_archive_rotates_and_falls_back_to_previous_generation(tmp_path: Path) -> None:
    durable = tmp_path / "drive"
    first = _registry(tmp_path, "first")
    ResumeArchiveManager(first, durable).save(tmp_path / "first-archive")
    second = _registry(tmp_path, "second")
    manager = ResumeArchiveManager(second, durable)
    manager.save(tmp_path / "second-archive")
    assert manager.previous.is_file()
    manager.current.write_bytes(b"corrupt")

    restored = tmp_path / "restored"
    source, state = ResumeArchiveManager(restored, durable).restore(expected_contract=CONTRACT)  # type: ignore[misc]
    assert source.name == "03q_resume.previous.zip"
    assert state["stage"] == "first"
    assert (restored / "payload.txt").read_text() == "first"


def test_archive_refuses_contract_mismatch_and_copy_corruption(tmp_path: Path) -> None:
    durable = tmp_path / "drive"
    registry = _registry(tmp_path, "registry")
    manager = ResumeArchiveManager(registry, durable)
    archive, _ = manager.save(tmp_path / "archive")
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_resume_archive(archive, expected_contract={"experiment": "other"})

    def corrupt_copy(source: str | Path, destination: str | Path) -> object:
        result = shutil.copyfile(source, destination)
        Path(destination).write_bytes(b"truncated")
        return result

    with pytest.raises(RuntimeError, match="identity mismatch"):
        ResumeArchiveManager(registry, tmp_path / "bad-drive", copy=corrupt_copy).save(
            tmp_path / "bad-archive"
        )


def test_archive_requires_state_and_preserves_json_contract(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ResumeArchiveManager(empty, tmp_path / "drive").save(tmp_path / "empty-archive")
    state = json.loads(write_resume_state(empty, {
        "immutable_contract": CONTRACT, "stage": "preflight"
    }).read_text())
    assert state["immutable_contract"] == CONTRACT
