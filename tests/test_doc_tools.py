"""文档处理工具测试 —— 预览与摘要（PDF/DOCX 使用 mock）。"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from scaffold.tools.errors import ToolError, ToolErrorCode
from scaffold.safety.sandbox import PathSandbox
import fs_agent.tools.file_tools as ft_module


@pytest.fixture(autouse=True)
def _set_sandbox(tmp_path):
    """为所有测试设置指向 tmp_path 的沙箱。"""
    old = ft_module._sandbox
    ft_module._sandbox = PathSandbox([str(tmp_path)])
    yield
    ft_module._sandbox = old


class TestPreviewFile:
    def test_preview_text_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello world\nThis is line 2\n" * 5)
        from fs_agent.tools.doc_tools import preview_file
        result = preview_file(str(f))
        assert "hello.txt" in result.lower() or "Hello world" in result

    def test_preview_python_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('def main():\n    print("hello")\n\nclass Foo:\n    pass\n')
        from fs_agent.tools.doc_tools import preview_file
        result = preview_file(str(f))
        assert "test.py" in result or "main" in result

    def test_preview_nonexistent(self, tmp_path):
        from fs_agent.tools.doc_tools import preview_file
        with pytest.raises(ToolError) as exc:
            preview_file(str(tmp_path / "nope.txt"))
        assert exc.value.code == ToolErrorCode.NOT_FOUND


class TestSummarizeFile:
    def test_summarize_text_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        from fs_agent.tools.doc_tools import summarize_file
        result = summarize_file(str(f))
        assert "data.csv" in result or "csv" in result.lower()

    def test_summarize_python_file(self, tmp_path):
        f = tmp_path / "module.py"
        f.write_text("class Foo:\n    pass\n\ndef bar():\n    pass\n\ndef baz():\n    pass\n")
        from fs_agent.tools.doc_tools import summarize_file
        result = summarize_file(str(f))
        assert "Foo" in result or "class" in result.lower() or "function" in result.lower()


class TestReadPdf:
    def test_read_pdf_wrong_extension(self, tmp_path):
        f = tmp_path / "not_pdf.txt"
        f.write_text("hello")
        from fs_agent.tools.doc_tools import read_pdf
        with pytest.raises(ToolError) as exc:
            read_pdf(str(f))
        assert exc.value.code == ToolErrorCode.UNSUPPORTED_FORMAT

    def test_read_pdf_not_found(self, tmp_path):
        from fs_agent.tools.doc_tools import read_pdf
        with pytest.raises(ToolError) as exc:
            read_pdf(str(tmp_path / "nope.pdf"))
        assert exc.value.code == ToolErrorCode.NOT_FOUND


class TestReadDocx:
    def test_read_docx_wrong_extension(self, tmp_path):
        f = tmp_path / "not_doc.txt"
        f.write_text("hello")
        from fs_agent.tools.doc_tools import read_docx
        with pytest.raises(ToolError) as exc:
            read_docx(str(f))
        assert exc.value.code == ToolErrorCode.UNSUPPORTED_FORMAT
