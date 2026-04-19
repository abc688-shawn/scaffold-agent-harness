"""Tests for eval runner — case loading, rule evaluation, judge integration."""
from __future__ import annotations

import pytest
import json
from pathlib import Path
from dataclasses import dataclass, field

from evals.runner import EvalCase, EvalResult, load_cases, evaluate_result, save_results


class TestLoadCases:
    def test_load_from_yaml(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("""
cases:
  - id: test1
    description: "Test case 1"
    user_input: "Hello"
    expected_contains: ["hello"]
    category: basic
  - id: test2
    description: "Test case 2"
    user_input: "World"
    category: basic
""")
        cases = load_cases(f)
        assert len(cases) == 2
        assert cases[0].id == "test1"
        assert cases[0].expected_contains == ["hello"]

    def test_load_from_directory(self, tmp_path):
        (tmp_path / "a.yaml").write_text("""
cases:
  - id: a1
    description: "Case A1"
    user_input: "A"
""")
        (tmp_path / "b.yaml").write_text("""
cases:
  - id: b1
    description: "Case B1"
    user_input: "B"
  - id: b2
    description: "Case B2"
    user_input: "B2"
""")
        cases = load_cases(tmp_path)
        assert len(cases) == 3

    def test_load_real_cases(self):
        cases = load_cases("evals/cases/")
        assert len(cases) >= 50


class TestEvaluateResult:
    def _make_result(self, message: str, history=None):
        @dataclass
        class FakeResult:
            final_message: str | None
            steps: int
            total_usage: object
            history: list = field(default_factory=list)

        @dataclass
        class FakeUsage:
            total_tokens: int = 100

        return FakeResult(
            final_message=message, steps=1,
            total_usage=FakeUsage(),
            history=history or [],
        )

    def test_contains_pass(self):
        case = EvalCase(id="t", description="d", user_input="q",
                        expected_contains=["hello", "world"])
        result = self._make_result("Hello World!")
        er = evaluate_result(case, result)
        assert er.passed
        assert er.score == 1.0

    def test_contains_fail(self):
        case = EvalCase(id="t", description="d", user_input="q",
                        expected_contains=["missing_word"])
        result = self._make_result("Hello World!")
        er = evaluate_result(case, result)
        assert not er.passed
        assert er.score == 0.0

    def test_not_contains_pass(self):
        case = EvalCase(id="t", description="d", user_input="q",
                        expected_not_contains=["password", "secret"])
        result = self._make_result("Everything is fine.")
        er = evaluate_result(case, result)
        assert er.passed
        assert er.score == 1.0

    def test_not_contains_fail(self):
        case = EvalCase(id="t", description="d", user_input="q",
                        expected_not_contains=["leaked_password"])
        result = self._make_result("Your leaked_password is here")
        er = evaluate_result(case, result)
        assert not er.passed

    def test_no_checks_passes(self):
        case = EvalCase(id="t", description="d", user_input="q")
        result = self._make_result("Some response")
        er = evaluate_result(case, result)
        assert er.passed
        assert er.score == 1.0

    def test_mixed_checks(self):
        case = EvalCase(id="t", description="d", user_input="q",
                        expected_contains=["yes"],
                        expected_not_contains=["no"])
        result = self._make_result("yes, indeed, no problem")
        er = evaluate_result(case, result)
        # "yes" found → pass, "no" found → fail
        # score = 1/2 = 0.5 → not passed (threshold 0.8)
        assert not er.passed


class TestSaveResults:
    def test_save_and_load(self, tmp_path):
        results = [
            EvalResult(case_id="t1", passed=True, score=0.9,
                       final_message="ok", steps=2, total_tokens=100,
                       latency_ms=50.0, category="basic"),
            EvalResult(case_id="t2", passed=False, score=0.3,
                       final_message="fail", steps=1, total_tokens=50,
                       latency_ms=30.0, category="security", error="timeout"),
        ]
        path = tmp_path / "results.json"
        save_results(results, path)

        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["case_id"] == "t1"
        assert data[0]["passed"] is True
        assert data[1]["error"] == "timeout"
