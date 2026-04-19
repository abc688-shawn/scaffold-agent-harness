"""Tests for advanced tools — organize, tag, compare."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.safety.sandbox import PathSandbox
import fs_agent.tools.file_tools as ft_module
from fs_agent.tools.advanced_tools import organize_files, tag_files, search_by_tag, compare_files


@pytest.fixture(autouse=True)
def _set_sandbox(tmp_path):
    old = ft_module._sandbox
    ft_module._sandbox = PathSandbox([str(tmp_path)])
    yield
    ft_module._sandbox = old


class TestOrganizeFiles:
    def test_dry_run_by_extension(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("img")
        (tmp_path / "doc.pdf").write_text("pdf")
        (tmp_path / "code.py").write_text("py")
        result = organize_files(str(tmp_path), strategy="extension", dry_run=True)
        assert "DRY RUN" in result
        assert (tmp_path / "photo.jpg").exists()

    def test_execute_by_extension(self, tmp_path):
        (tmp_path / "a.py").write_text("python")
        (tmp_path / "b.py").write_text("python2")
        (tmp_path / "c.txt").write_text("text")
        result = organize_files(str(tmp_path), strategy="extension", dry_run=False)
        assert "Executed" in result

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = organize_files(str(empty))
        assert "No files" in result

    def test_invalid_strategy(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        with pytest.raises(ToolError) as exc:
            organize_files(str(tmp_path), strategy="invalid")
        assert exc.value.code == ToolErrorCode.INVALID_ARGUMENTS

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ToolError):
            organize_files(str(f))


class TestTagFiles:
    def test_tag_and_read_back(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Doc")
        tags_file = tmp_path / ".file_tags.json"
        result = tag_files(str(f), "important,review", tags_file=str(tags_file))
        assert "important" in result
        assert "review" in result
        db = json.loads(tags_file.read_text())
        assert len(db) == 1
        tags = list(db.values())[0]
        assert "important" in tags
        assert "review" in tags

    def test_tag_append(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Doc")
        tags_file = tmp_path / ".file_tags.json"
        tag_files(str(f), "first", tags_file=str(tags_file))
        tag_files(str(f), "second", tags_file=str(tags_file))
        db = json.loads(tags_file.read_text())
        tags = list(db.values())[0]
        assert "first" in tags
        assert "second" in tags

    def test_tag_nonexistent_file(self, tmp_path):
        with pytest.raises(ToolError) as exc:
            tag_files(str(tmp_path / "nope.txt"), "tag1", tags_file=str(tmp_path / ".tags.json"))
        assert exc.value.code == ToolErrorCode.NOT_FOUND


class TestSearchByTag:
    def test_search_existing_tag(self, tmp_path):
        tags_file = tmp_path / ".file_tags.json"
        db = {
            str(tmp_path / "a.py"): ["python", "important"],
            str(tmp_path / "b.md"): ["docs", "important"],
            str(tmp_path / "c.txt"): ["notes"],
        }
        tags_file.write_text(json.dumps(db))
        result = search_by_tag("important", tags_file=str(tags_file))
        assert "a.py" in result
        assert "b.md" in result
        assert "c.txt" not in result

    def test_search_no_matches(self, tmp_path):
        tags_file = tmp_path / ".file_tags.json"
        tags_file.write_text(json.dumps({"a.py": ["python"]}))
        result = search_by_tag("nonexistent", tags_file=str(tags_file))
        assert "No files" in result

    def test_search_no_db(self, tmp_path):
        result = search_by_tag("tag", tags_file=str(tmp_path / ".tags.json"))
        assert "No tags database" in result


class TestCompareFiles:
    def test_identical_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content\nline 2")
        f2.write_text("same content\nline 2")
        result = compare_files(str(f1), str(f2))
        assert "identical" in result.lower()

    def test_different_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line 1\nline 2\nline 3")
        f2.write_text("line 1\nchanged\nline 3")
        result = compare_files(str(f1), str(f2))
        assert "changed" in result or "---" in result or "+++" in result

    def test_missing_file(self, tmp_path):
        f1 = tmp_path / "exists.txt"
        f1.write_text("content")
        with pytest.raises(ToolError):
            compare_files(str(f1), str(tmp_path / "nope.txt"))
