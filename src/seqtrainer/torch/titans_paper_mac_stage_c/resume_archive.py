"""Validated two-generation archive persistence for interruptible Stage C jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping
from zipfile import BadZipFile, ZipFile


STATE_NAME = "resume_state.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_resume_state(root: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(root) / STATE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.partial")
    partial.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    return destination


def validate_resume_archive(
    path: str | Path, *, expected_contract: Mapping[str, object] | None = None
) -> dict[str, object]:
    archive = Path(path)
    try:
        with ZipFile(archive) as handle:
            corrupt = handle.testzip()
            if corrupt is not None:
                raise ValueError(f"resume archive has a corrupt member: {corrupt}")
            if STATE_NAME not in handle.namelist():
                raise ValueError(f"resume archive is missing {STATE_NAME}")
            state = json.loads(handle.read(STATE_NAME))
    except BadZipFile as error:
        raise ValueError(f"invalid resume ZIP: {archive}") from error
    if not isinstance(state, dict) or not isinstance(state.get("immutable_contract"), dict):
        raise ValueError("resume state has no immutable contract")
    expected = dict(expected_contract or {})
    actual = state["immutable_contract"]
    mismatches = {key: (actual.get(key), value) for key, value in expected.items()
                  if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"resume immutable contract mismatch: {mismatches}")
    return state


class ResumeArchiveManager:
    """Publish and restore one current plus one previous validated ZIP."""

    def __init__(
        self,
        registry: str | Path,
        durable_directory: str | Path,
        *,
        copy: Callable[[str | Path, str | Path], object] = shutil.copyfile,
        replace: Callable[[str | Path, str | Path], object] = os.replace,
    ) -> None:
        self.registry = Path(registry)
        self.durable_directory = Path(durable_directory)
        self.current = self.durable_directory / "03q_resume.zip"
        self.previous = self.durable_directory / "03q_resume.previous.zip"
        self.partial = self.durable_directory / "03q_resume.partial.zip"
        self.copy = copy
        self.replace = replace

    def restore(self, *, expected_contract: Mapping[str, object]) -> tuple[Path, dict[str, object]] | None:
        failures: list[str] = []
        for archive in (self.current, self.previous):
            if not archive.is_file():
                continue
            try:
                state = validate_resume_archive(archive, expected_contract=expected_contract)
                self.registry.mkdir(parents=True, exist_ok=True)
                shutil.unpack_archive(archive, self.registry)
                return archive, state
            except (OSError, ValueError) as error:
                failures.append(f"{archive.name}: {error}")
        if failures:
            raise RuntimeError("no valid resume archive; " + "; ".join(failures))
        return None

    def save(self, local_archive_base: str | Path) -> tuple[Path, str]:
        state_path = self.registry / STATE_NAME
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        self.durable_directory.mkdir(parents=True, exist_ok=True)
        base = Path(local_archive_base)
        local_zip = Path(shutil.make_archive(str(base), "zip", root_dir=self.registry))
        state = validate_resume_archive(local_zip)
        expected_contract = state["immutable_contract"]
        local_hash = sha256_file(local_zip)
        self.copy(local_zip, self.partial)
        if self.partial.stat().st_size != local_zip.stat().st_size or sha256_file(self.partial) != local_hash:
            raise RuntimeError("resume archive copy identity mismatch")
        validate_resume_archive(self.partial, expected_contract=expected_contract)
        if self.current.is_file():
            self.replace(self.current, self.previous)
        self.replace(self.partial, self.current)
        return self.current, local_hash
