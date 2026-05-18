# tests/logging/test_csv_rotation.py
"""Tests for rotate_log_file() and NluCsvLogger rotate_on_start."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.logging.nlu_csv_logger import NluCsvLogger, rotate_log_file


class TestRotateLogFile:
    def test_moves_file_to_older_dir(self, tmp_path):
        f = tmp_path / "nlu_log.csv"
        f.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        dest = rotate_log_file(f)
        assert dest is not None
        assert dest.parent.name == "older"
        assert dest.exists()
        assert not f.exists()

    def test_original_content_preserved(self, tmp_path):
        f = tmp_path / "nlu_log.csv"
        content = "a,b\n1,2\n"
        f.write_text(content, encoding="utf-8")

        dest = rotate_log_file(f)
        assert dest.read_text(encoding="utf-8") == content

    def test_timestamp_in_filename(self, tmp_path):
        f = tmp_path / "nlu_log.csv"
        f.write_text("data", encoding="utf-8")
        dest = rotate_log_file(f)
        # filename should be nlu_log_YYYYMMDD_HHMMSS.csv
        assert dest.stem.startswith("nlu_log_")
        assert dest.suffix == ".csv"

    def test_returns_none_when_file_absent(self, tmp_path):
        missing = tmp_path / "not_there.csv"
        result = rotate_log_file(missing)
        assert result is None

    def test_no_collision_on_multiple_rotations(self, tmp_path):
        for i in range(3):
            f = tmp_path / "nlu_log.csv"
            f.write_text(f"row{i}", encoding="utf-8")
            rotate_log_file(f)

        older = tmp_path / "older"
        archived = list(older.glob("*.csv"))
        assert len(archived) == 3
        # All names unique
        assert len({a.name for a in archived}) == 3

    def test_does_not_raise_on_bad_path(self, tmp_path):
        # Simulate un-rotatable scenario by pointing to a dir
        d = tmp_path / "subdir"
        d.mkdir()
        # rotate_log_file on a directory (not a file) should not raise,
        # because the directory exists but is not a regular file to move.
        # The function should return None or the moved-dir path — just not raise.
        try:
            rotate_log_file(d)
        except Exception as exc:
            pytest.fail(f"rotate_log_file raised unexpectedly: {exc}")


class TestNluCsvLoggerRotateOnStart:
    def _write_csv_with_header(self, path: Path, headers: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow({h: f"old_{h}" for h in headers})

    def test_rotate_on_start_archives_existing_file(self, tmp_path):
        log_dir = tmp_path / "nlu"
        log_dir.mkdir()
        log_path = log_dir / "nlu_log.csv"
        self._write_csv_with_header(log_path, NluCsvLogger.HEADERS)

        logger = NluCsvLogger(
            enabled=True,
            log_dir=str(log_dir),
            rotate_on_start=True,
        )
        logger.shutdown()

        # Old file should be archived; new (header-only) file at log_path
        older_dir = log_dir / "older"
        assert older_dir.exists()
        archived = list(older_dir.glob("*.csv"))
        assert len(archived) == 1

        # The new CSV must exist with the updated headers
        assert log_path.exists()
        with log_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            written_headers = next(reader)
        assert written_headers == NluCsvLogger.HEADERS

    def test_no_rotation_when_flag_false(self, tmp_path):
        log_dir = tmp_path / "nlu"
        log_dir.mkdir()
        log_path = log_dir / "nlu_log.csv"
        log_path.write_text("old content", encoding="utf-8")

        logger = NluCsvLogger(
            enabled=True,
            log_dir=str(log_dir),
            rotate_on_start=False,
        )
        logger.shutdown()

        # No older/ dir — original file still contains old content prefix
        older_dir = log_dir / "older"
        assert not older_dir.exists()

    def test_rotate_on_start_when_no_existing_file(self, tmp_path):
        log_dir = tmp_path / "nlu"
        log_dir.mkdir()

        # Should not raise even when the file doesn't exist yet
        logger = NluCsvLogger(
            enabled=True,
            log_dir=str(log_dir),
            rotate_on_start=True,
        )
        logger.shutdown()
        assert (log_dir / "nlu_log.csv").exists()
