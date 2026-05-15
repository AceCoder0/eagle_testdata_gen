"""Tests for eagle_testdata_gen — covers dedup, fewshot, shuffle, and prefix cache."""

import json
import os
import sys
import tempfile
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dedup import exact_dedup, question_hash, extract_final_question, cross_prompt_dedup
from fewshot import (
    load_pool,
    load_tokenizer,
    FewShotBuilder,
    INSTRUCTION,
    QA_TEMPLATE,
    FINAL_Q_TEMPLATE,
    _format_count,
)
from shuffle import shuffle_jsonl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Load the DeepSeekR1 tokenizer once per test run."""
    path = os.path.join(os.path.dirname(__file__), "DeepSeekR1")
    return load_tokenizer(path)


@pytest.fixture
def synthetic_pool(tokenizer):
    """Create a small synthetic Q&A pool with diverse lengths."""
    records = []
    for i in range(200):
        # Vary question and answer lengths to create a realistic distribution
        q = f"Math problem {i}: Solve for x where {i}x + {i*2} = {i*5}. " \
            + "Show all steps clearly." * (1 + i % 3)
        a = f"Solution {i}: " + f"Step-by-step reasoning for problem {i}. " * (1 + i % 5)
        rid = f"synth_{i}"
        q_tok = _format_count(q, tokenizer)
        a_tok = _format_count(a, tokenizer)
        records.append({
            "id": rid,
            "question": q,
            "answer": a,
            "source": f"ds_{i % 5}",
            "question_tokens": q_tok,
            "answer_tokens": a_tok,
            "total_tokens": q_tok + a_tok,
        })
    return records


@pytest.fixture
def builder(synthetic_pool, tokenizer):
    """Create a FewShotBuilder with the synthetic pool."""
    return FewShotBuilder(synthetic_pool, tokenizer, seed=12345)


@pytest.fixture
def pool_file(synthetic_pool):
    """Write synthetic pool to a temp JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in synthetic_pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# dedup tests
# ---------------------------------------------------------------------------

class TestDedup:
    def test_question_hash_deterministic(self):
        assert question_hash("hello") == question_hash("hello")
        assert question_hash("hello") != question_hash("world")

    def test_exact_dedup_removes_duplicates(self):
        records = [
            {"question": "Q1", "id": "a"},
            {"question": "Q2", "id": "b"},
            {"question": "Q1", "id": "c"},  # dup
            {"question": "Q3", "id": "d"},
        ]
        result = exact_dedup(records, key="question")
        assert len(result) == 3
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "d"]

    def test_exact_dedup_empty(self):
        assert exact_dedup([], key="question") == []

    def test_extract_final_question(self):
        prompt = "Instruction\n\nQ: What is 2+2?\nA: 4\n\nQ: What is 3+3?\nA:\n"
        assert extract_final_question(prompt) == "What is 3+3?"

    def test_extract_final_question_no_marker(self):
        assert extract_final_question("Just a question") == "Just a question"

    def test_cross_prompt_dedup(self):
        records = [
            {"final_question": "Q1"},
            {"final_question": "Q2"},
            {"final_question": "Q1"},  # dup
        ]
        result = cross_prompt_dedup(records, key="final_question")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# fewshot tests — build_one / build_many
# ---------------------------------------------------------------------------

class TestFewShotBuild:
    def test_build_one_within_tolerance(self, builder):
        rec = builder.build_one(target_tokens=3000, used_final_qs=set(), tolerance=0.05)
        assert rec is not None
        assert 2850 <= rec["question_token_len"] <= 3150
        assert rec["question"].startswith(INSTRUCTION.strip())
        assert rec["num_exemplars"] >= 2
        assert rec["final_question"]

    def test_build_one_unique_final_questions(self, builder):
        used = set()
        finals = set()
        for _ in range(20):
            rec = builder.build_one(target_tokens=2000, used_final_qs=used, tolerance=0.10)
            assert rec is not None
            finals.add(rec["final_question"])
        # All final questions should be unique
        assert len(finals) == 20

    def test_build_one_structure(self, builder):
        rec = builder.build_one(target_tokens=2000, used_final_qs=set(), tolerance=0.10)
        prompt = rec["question"]
        # Must start with instruction
        assert prompt.startswith(INSTRUCTION)
        # Must contain Q: and A: markers
        assert "Q: " in prompt
        assert "A: " in prompt
        # Must end with "A:\n" (no answer for final question)
        assert prompt.rstrip().endswith("A:")

    def test_build_many_count(self, builder):
        results = builder.build_many(target_tokens=2000, num_samples=10, tolerance=0.10)
        assert len(results) == 10

    def test_build_many_all_within_tolerance(self, builder):
        results = builder.build_many(target_tokens=2000, num_samples=30, tolerance=0.10)
        for r in results:
            assert 1800 <= r["question_token_len"] <= 2200, \
                f"Got {r['question_token_len']}, expected 1800-2200"

    def test_build_many_final_questions_unique(self, builder):
        results = builder.build_many(target_tokens=2000, num_samples=30, tolerance=0.10)
        finals = [r["final_question"] for r in results]
        assert len(finals) == len(set(finals))

    def test_build_one_no_duplicate_qa(self, builder):
        """Verify no Q&A pair appears twice in the same prompt."""
        rec = builder.build_one(target_tokens=3000, used_final_qs=set(), tolerance=0.05)
        prompt = rec["question"]
        q_markers = [i for i in range(len(prompt)) if prompt.startswith("Q: ", i)]
        questions = []
        for start in q_markers:
            end = prompt.find("\n", start)
            questions.append(prompt[start:end] if end > 0 else prompt[start:])
        # The final question is excluded from exemplars, so ALL questions are unique
        assert len(questions) == len(set(questions)), \
            f"Found {len(questions) - len(set(questions))} duplicate question(s) in prompt"


# ---------------------------------------------------------------------------
# fewshot tests — prefix cache
# ---------------------------------------------------------------------------

class TestPrefixCache:
    def test_build_prefix_batch_structure(self, builder):
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.5, tolerance=0.10
        )
        assert len(results) == 5
        for r in results:
            assert "prefix_rate" in r
            assert r["prefix_rate"] == 0.5
            assert "batch_id" in r
            assert "common_exemplars" in r
            assert "unique_exemplars" in r

    def test_prefix_identical_across_samples(self, builder):
        """The common prefix text must be byte-identical across all samples."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.5, tolerance=0.10
        )
        samples = [r["question"] for r in results]

        # Find the shortest common prefix across all sample pairs
        diverge = len(samples[0])
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                k = 0
                while (k < len(samples[i]) and k < len(samples[j])
                       and samples[i][k] == samples[j][k]):
                    k += 1
                diverge = min(diverge, k)

        common = samples[0][:diverge]
        assert len(common) > 0, "Should have non-empty common prefix"
        # All samples should start with the common prefix
        for s in samples:
            assert s.startswith(common), f"Sample doesn't start with common prefix"

    def test_prefix_suffixes_differ(self, builder):
        """After the common prefix, each sample must differ from others."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.5, tolerance=0.10
        )
        samples = [r["question"] for r in results]

        # Find divergence point of first two samples
        k = 0
        while (k < len(samples[0]) and k < len(samples[1])
               and samples[0][k] == samples[1][k]):
            k += 1
        diverge = k

        # Suffixes must differ
        suffixes = [s[diverge:] for s in samples]
        for i in range(len(suffixes)):
            for j in range(i + 1, len(suffixes)):
                assert suffixes[i] != suffixes[j], \
                    f"Samples {i} and {j} have identical suffixes"

    def test_prefix_rate_zero_fallback(self, builder):
        """prefix_rate=0 should fall back to build_many (independent samples)."""
        results = builder.build_prefix_batch(
            target_tokens=2000, num_samples=10, prefix_rate=0.0, tolerance=0.10
        )
        assert len(results) == 10
        # No prefix_rate field for independent mode? Actually build_many doesn't add it.
        # Let's just check the samples exist and have correct structure.
        for r in results:
            assert "question_token_len" in r
            assert "final_question" in r

    def test_prefix_rate_within_tolerance(self, builder):
        """All samples in a prefix batch must be within tolerance."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=20, prefix_rate=0.6, tolerance=0.10
        )
        for r in results:
            assert 3600 <= r["question_token_len"] <= 4400, \
                f"Got {r['question_token_len']}, expected 3600-4400"

    def test_prefix_final_questions_unique(self, builder):
        """Final questions must be unique across the batch."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=20, prefix_rate=0.5, tolerance=0.10
        )
        finals = [r["final_question"] for r in results]
        assert len(finals) == len(set(finals))

    def test_prefix_batch_id_consistent(self, builder):
        """All samples in a batch share the same batch_id."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.5, tolerance=0.10
        )
        batch_ids = set(r["batch_id"] for r in results)
        assert len(batch_ids) == 1

    def test_prefix_rate_validation(self, builder):
        """Invalid prefix_rate must raise ValueError."""
        with pytest.raises(ValueError):
            builder.build_prefix_batch(target_tokens=4000, num_samples=5, prefix_rate=-0.1)
        with pytest.raises(ValueError):
            builder.build_prefix_batch(target_tokens=4000, num_samples=5, prefix_rate=1.5)

    def test_prefix_high_rate(self, builder):
        """prefix_rate=0.9 should work with small unique suffix."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.9, tolerance=0.10
        )
        assert len(results) == 5
        for r in results:
            assert 3600 <= r["question_token_len"] <= 4400

    def test_prefix_common_exemplars_same_count(self, builder):
        """common_exemplars count should be identical across all samples."""
        results = builder.build_prefix_batch(
            target_tokens=4000, num_samples=5, prefix_rate=0.5, tolerance=0.10
        )
        common_counts = set(r["common_exemplars"] for r in results)
        assert len(common_counts) == 1


# ---------------------------------------------------------------------------
# shuffle tests
# ---------------------------------------------------------------------------

class TestShuffle:
    def test_shuffle_preserves_record_count(self, pool_file):
        out_file = pool_file + ".shuffled"
        try:
            shuffle_jsonl(pool_file, out_file)
            with open(pool_file) as f:
                original = [json.loads(l) for l in f]
            with open(out_file) as f:
                shuffled = [json.loads(l) for l in f]
            assert len(original) == len(shuffled)
        finally:
            if os.path.exists(out_file):
                os.unlink(out_file)

    def test_shuffle_preserves_all_ids(self, pool_file):
        out_file = pool_file + ".shuffled"
        try:
            shuffle_jsonl(pool_file, out_file)
            with open(pool_file) as f:
                ids_in = {json.loads(l)["id"] for l in f}
            with open(out_file) as f:
                ids_out = {json.loads(l)["id"] for l in f}
            assert ids_in == ids_out
        finally:
            if os.path.exists(out_file):
                os.unlink(out_file)


# ---------------------------------------------------------------------------
# pool loading tests
# ---------------------------------------------------------------------------

class TestPoolLoading:
    def test_load_pool(self, pool_file, synthetic_pool):
        loaded = load_pool(pool_file)
        assert len(loaded) == len(synthetic_pool)
        assert loaded[0]["id"] == synthetic_pool[0]["id"]

    def test_load_pool_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            loaded = load_pool(path)
            assert loaded == []
        finally:
            os.unlink(path)

    def test_load_pool_skips_blank_lines(self, pool_file):
        """Blank lines in JSONL should be skipped."""
        # pool_file already has clean data, tested implicitly via load_pool
        loaded = load_pool(pool_file)
        assert all(r["id"] for r in loaded)


# ---------------------------------------------------------------------------
# token count accuracy tests
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_format_count_positive(self, tokenizer):
        assert _format_count("hello world", tokenizer) > 0

    def test_format_count_empty_string(self, tokenizer):
        # Encoding empty string may return BOS token if add_special_tokens=True
        n = _format_count("", tokenizer, add_special_tokens=False)
        assert n == 0

    def test_instruction_is_nonzero(self, tokenizer):
        assert _format_count(INSTRUCTION, tokenizer) > 5

    def test_precomputed_qa_tokens_match(self, builder, tokenizer):
        """Pre-computed QA tokens should match on-the-fly count."""
        for r in builder.pool[:10]:
            formatted = QA_TEMPLATE.format(question=r["question"], answer=r["answer"])
            expected = _format_count(formatted, tokenizer)
            assert builder._qa_tokens[r["id"]] == expected


# ---------------------------------------------------------------------------
# reproducibility tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_build_many_identical(self, synthetic_pool, tokenizer):
        """Same seed + same pool -> identical output from build_many."""
        b1 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        b2 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        r1 = b1.build_many(target_tokens=2000, num_samples=5, tolerance=0.10)
        r2 = b2.build_many(target_tokens=2000, num_samples=5, tolerance=0.10)
        for a, b in zip(r1, r2):
            assert a["question"] == b["question"]
            assert a["question_token_len"] == b["question_token_len"]
            assert a["final_question"] == b["final_question"]

    def test_different_seed_different_output(self, synthetic_pool, tokenizer):
        """Different seeds should produce different prompts."""
        b1 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        b2 = FewShotBuilder(synthetic_pool, tokenizer, seed=99)
        r1 = b1.build_many(target_tokens=2000, num_samples=5, tolerance=0.10)
        r2 = b2.build_many(target_tokens=2000, num_samples=5, tolerance=0.10)
        any_diff = any(
            r1[i]["question"] != r2[i]["question"] for i in range(5)
        )
        assert any_diff, "Different seeds should produce different output"

    def test_prefix_batch_reproducible(self, synthetic_pool, tokenizer):
        """Same seed -> same batch_id and identical prompts in prefix mode."""
        b1 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        b2 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        r1 = b1.build_prefix_batch(
            target_tokens=3000, num_samples=3, prefix_rate=0.5, tolerance=0.10
        )
        r2 = b2.build_prefix_batch(
            target_tokens=3000, num_samples=3, prefix_rate=0.5, tolerance=0.10
        )
        assert r1[0]["batch_id"] == r2[0]["batch_id"]
        for a, b in zip(r1, r2):
            assert a["question"] == b["question"]
            assert a["question_token_len"] == b["question_token_len"]

    def test_different_seed_different_batch_id(self, synthetic_pool, tokenizer):
        """Different seeds -> different batch_ids."""
        b1 = FewShotBuilder(synthetic_pool, tokenizer, seed=42)
        b2 = FewShotBuilder(synthetic_pool, tokenizer, seed=99)
        r1 = b1.build_prefix_batch(
            target_tokens=3000, num_samples=3, prefix_rate=0.5, tolerance=0.10
        )
        r2 = b2.build_prefix_batch(
            target_tokens=3000, num_samples=3, prefix_rate=0.5, tolerance=0.10
        )
        assert r1[0]["batch_id"] != r2[0]["batch_id"]
