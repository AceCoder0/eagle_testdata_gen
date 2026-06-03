"""Few-shot prompt builder for math performance test cases.

Builds long-context inputs by concatenating multiple Q&A exemplar pairs
plus a final target question, packing to reach a target token length.
"""
import json
import random
from typing import List, Dict, Set, Tuple, Optional
from transformers import AutoTokenizer


# Few-shot prompt template
INSTRUCTION = "Solve the following math problems. Show your step-by-step reasoning.\n\n"
QA_TEMPLATE = "Q: {question}\nA: {answer}\n\n"
FINAL_Q_TEMPLATE = "Q: {question}\nA:\n"


def load_pool(pool_path: str) -> List[Dict]:
    """Load the math Q&A pool from JSONL file."""
    records = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.model_max_length = int(1e9)  # allow arbitrary long sequences for token counting
    return tok


def _format_count(text: str, tokenizer, add_special_tokens: bool = False) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


class FewShotBuilder:
    """Builds few-shot prompts of approximate target token lengths."""

    def __init__(self, pool: List[Dict], tokenizer, seed: int = 42,
                 answer_style: str = "mixed", min_answer_tokens: int = 0):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.answer_style = answer_style

        # Filter pool by minimum answer token length
        if min_answer_tokens > 0:
            self.pool = [r for r in pool if r.get("answer_tokens", 0) >= min_answer_tokens]
            if len(self.pool) == 0:
                raise ValueError(
                    f"No records with answer_tokens >= {min_answer_tokens} "
                    f"(pool has {len(pool)} records)"
                )
        else:
            self.pool = list(pool)

        # Pre-compute formatted token counts for each pool entry
        self._qa_tokens: Dict[str, int] = {}
        self._q_only_tokens: Dict[str, int] = {}
        for r in self.pool:
            rid = r["id"]
            self._qa_tokens[rid] = _format_count(
                QA_TEMPLATE.format(question=r["question"], answer=r["answer"]),
                tokenizer,
            )
            self._q_only_tokens[rid] = _format_count(
                FINAL_Q_TEMPLATE.format(question=r["question"]),
                tokenizer,
            )

        self.instruction_tokens = _format_count(INSTRUCTION, tokenizer)

        # Group pool by source for stratified sampling
        self._by_source: Dict[str, List[Dict]] = {}
        for r in self.pool:
            src = r.get("source", "unknown")
            self._by_source.setdefault(src, []).append(r)

    def _pick_by_style(self, candidates: List[Dict]) -> Dict:
        """Pick a candidate biased by answer_style preference."""
        if self.answer_style == "mixed":
            return self.rng.choice(candidates)
        reverse = self.answer_style == "detailed"
        candidates.sort(key=lambda r: r.get("answer_tokens", 0), reverse=reverse)
        n = max(1, len(candidates) // 2)
        return self.rng.choice(candidates[:n])

    def _pick_stratified(self, used_ids: Set[str], sources_used: Set[str]) -> Optional[Dict]:
        """Pick a random Q&A from a source dataset not yet used in this prompt."""
        available_sources = [s for s in self._by_source if s not in sources_used]
        if not available_sources:
            available_sources = list(self._by_source.keys())

        self.rng.shuffle(available_sources)
        for src in available_sources:
            candidates = [r for r in self._by_source[src] if r["id"] not in used_ids]
            if candidates:
                return self._pick_by_style(candidates)

        # Fallback: any unused record
        candidates = [r for r in self.pool if r["id"] not in used_ids]
        if candidates:
            return self._pick_by_style(candidates)
        return None

    def build_one(
        self,
        target_tokens: int,
        used_final_qs: Set[str],
        tolerance: float = 0.05,
        min_qa_pairs: int = 2,
    ) -> Optional[Dict]:
        """Build a single few-shot prompt of approximately target_tokens length.

        Args:
            target_tokens: Desired total token count for the full prompt.
            used_final_qs: Set of Q&A IDs already used as final questions
                           (modified in-place).
            tolerance: Acceptable fractional deviation from target
                       (e.g., 0.05 = +/-5%).
            min_qa_pairs: Minimum number of exemplar Q&A pairs to include.

        Returns:
            A dict with keys: question, question_token_len, final_question,
            source, num_exemplars, target_tokens.
            Returns None if not enough data.
        """
        lower_bound = int(target_tokens * (1 - tolerance))
        upper_bound = int(target_tokens * (1 + tolerance))

        # Pick a final question not yet used
        available_final = [r for r in self.pool if r["id"] not in used_final_qs]
        if not available_final:
            used_final_qs.clear()
            available_final = list(self.pool)

        final_qa = self._pick_by_style(available_final)
        used_final_qs.add(final_qa["id"])

        final_q_formatted = FINAL_Q_TEMPLATE.format(question=final_qa["question"])
        final_q_tokens = self._q_only_tokens[final_qa["id"]]

        baseline = self.instruction_tokens + final_q_tokens
        remaining = target_tokens - baseline

        exemplars: List[Tuple[Dict, int]] = []  # (record, formatted_qa_tokens)
        used_in_prompt: Set[str] = {final_qa["id"]}
        used_sources: Set[str] = {final_qa.get("source", "unknown")}

        # Phase 1: Greedy packing to fill budget
        budget = remaining
        attempts = 0
        max_attempts = len(self.pool) * 2

        while budget > (target_tokens * tolerance) and attempts < max_attempts:
            candidate = self._pick_stratified(used_in_prompt, set())
            if candidate is None:
                break

            qa_tok = self._qa_tokens[candidate["id"]]
            if qa_tok <= budget + (target_tokens * tolerance * 0.5):
                exemplars.append((candidate, qa_tok))
                budget -= qa_tok
                used_in_prompt.add(candidate["id"])
                used_sources.add(candidate.get("source", "unknown"))
            attempts += 1

        # Phase 2: If below min_qa_pairs, force-add from remaining pool
        while len(exemplars) < min_qa_pairs:
            candidates = [r for r in self.pool if r["id"] not in used_in_prompt]
            if not candidates:
                break
            c = self._pick_by_style(candidates)
            qa_tok = self._qa_tokens[c["id"]]
            exemplars.append((c, qa_tok))
            budget -= qa_tok
            used_in_prompt.add(c["id"])
            used_sources.add(c.get("source", "unknown"))

        # Phase 3: Fill remaining budget with small Q&A pairs
        if budget > 0:
            small_candidates = sorted(
                [r for r in self.pool if r["id"] not in used_in_prompt],
                key=lambda r: self._qa_tokens.get(r["id"], 0),
            )
            for c in small_candidates:
                qa_tok = self._qa_tokens[c["id"]]
                if qa_tok <= budget:
                    exemplars.append((c, qa_tok))
                    budget -= qa_tok
                    used_in_prompt.add(c["id"])
                    if budget <= 0:
                        break

        # Assemble the prompt
        self.rng.shuffle(exemplars)
        parts = [INSTRUCTION]
        for rec, _ in exemplars:
            parts.append(
                QA_TEMPLATE.format(question=rec["question"], answer=rec["answer"])
            )
        parts.append(final_q_formatted)

        full_prompt = "".join(parts)
        actual_tokens = _format_count(full_prompt, self.tokenizer, add_special_tokens=True)

        return {
            "question": full_prompt,
            "answer": "",
            "question_token_len": actual_tokens,
            "final_question": final_qa["question"],
            "source": f"fewshot_{len(exemplars)}_exemplars",
            "num_exemplars": len(exemplars),
            "target_tokens": target_tokens,
        }

    def build_many(
        self,
        target_tokens: int,
        num_samples: int,
        tolerance: float = 0.05,
        min_qa_pairs: int = 2,
    ) -> List[Dict]:
        """Build num_samples few-shot prompts at target_tokens length."""
        results = []
        used_final_qs: Set[str] = set()

        for i in range(num_samples):
            record = self.build_one(
                target_tokens, used_final_qs, tolerance, min_qa_pairs
            )
            if record is None:
                print(f"  Warning: could only generate {i}/{num_samples} samples at "
                      f"target={target_tokens}")
                break
            results.append(record)

        # Cross-prompt dedup of final questions.
        # Skip when filtered pool is smaller than samples — allows
        # cycling through the same records to meet the requested count.
        if len(self.pool) >= num_samples:
            from dedup import cross_prompt_dedup
            results = cross_prompt_dedup(results, key="final_question")

        return results

    # ------------------------------------------------------------------
    # Prefix-cache-aware batch builder
    # ------------------------------------------------------------------

    def _fill_exemplars(
        self,
        budget: int,
        exclude_ids: Set[str],
        tolerance: float,
        target_tokens: int,
    ) -> Tuple[List[Tuple[Dict, int]], int]:
        """Greedy fill exemplar Q&A pairs up to budget tokens.
        Returns (list of (record, qa_tokens), remaining_budget).
        """
        exemplars: List[Tuple[Dict, int]] = []
        used: Set[str] = set(exclude_ids)
        remaining = budget
        attempts = 0
        max_attempts = len(self.pool) * 2

        while remaining > (target_tokens * tolerance) and attempts < max_attempts:
            candidate = self._pick_stratified(used, set())
            if candidate is None:
                break
            qa_tok = self._qa_tokens[candidate["id"]]
            if qa_tok <= remaining + (target_tokens * tolerance * 0.5):
                exemplars.append((candidate, qa_tok))
                remaining -= qa_tok
                used.add(candidate["id"])
            attempts += 1

        # Fill remaining gaps with small Q&A pairs
        if remaining > 0:
            small = sorted(
                [r for r in self.pool if r["id"] not in used],
                key=lambda r: self._qa_tokens.get(r["id"], 0),
            )
            for c in small:
                qa_tok = self._qa_tokens[c["id"]]
                if qa_tok <= remaining:
                    exemplars.append((c, qa_tok))
                    remaining -= qa_tok
                    used.add(c["id"])
                    if remaining <= 0:
                        break

        return exemplars, remaining

    def _exemplars_to_text(self, exemplars: List[Tuple[Dict, int]]) -> str:
        """Render exemplar list to prompt text."""
        parts = []
        for rec, _ in exemplars:
            parts.append(
                QA_TEMPLATE.format(question=rec["question"], answer=rec["answer"])
            )
        return "".join(parts)

    def build_prefix_batch(
        self,
        target_tokens: int,
        num_samples: int,
        prefix_rate: float,
        tolerance: float = 0.05,
        min_qa_pairs: int = 2,
    ) -> List[Dict]:
        """Build a batch of prompts sharing a common prefix.

        The first ``target_tokens * prefix_rate`` tokens are **identical** across
        all samples (the common prefix built from the same Q&A pairs, in the
        same order). The remaining tokens are **unique per sample**, ensuring
        that the non-cached suffix differs across requests.

        Args:
            target_tokens: Total desired token count per sample.
            num_samples: Number of samples in the batch.
            prefix_rate: Fraction of tokens that form the common (cache-hit)
                         prefix. 0.0 = all unique, 0.9 = 90% shared.
            tolerance: Acceptable fractional deviation from target.
            min_qa_pairs: Minimum exemplar Q&A pairs in the unique suffix.

        Returns:
            List of dicts, each with keys: question, question_token_len,
            final_question, source, num_exemplars, target_tokens,
            prefix_rate, batch_id, common_exemplars, unique_exemplars.
        """
        if not 0.0 <= prefix_rate <= 1.0:
            raise ValueError(f"prefix_rate must be in [0, 1], got {prefix_rate}")

        if prefix_rate == 0.0:
            # Degenerate case: no common prefix, fall back to independent samples
            return self.build_many(target_tokens, num_samples, tolerance, min_qa_pairs)

        common_target = int(target_tokens * prefix_rate)
        unique_target = target_tokens - common_target

        # ---- Phase 1: Build the common prefix (shared across all samples) ----
        common_budget = common_target - self.instruction_tokens
        common_exemplars, _ = self._fill_exemplars(
            common_budget, set(), tolerance, target_tokens
        )
        common_used_ids = {rec["id"] for rec, _ in common_exemplars}

        # Render the common prefix text (fixed for all samples)
        common_text = INSTRUCTION + self._exemplars_to_text(common_exemplars)
        common_text_tokens = _format_count(common_text, self.tokenizer)

        # ---- Phase 2: Build unique suffix per sample ----
        batch_id = ''.join(self.rng.choices('0123456789abcdef', k=8))
        results: List[Dict] = []
        used_final_qs: Set[str] = set()

        for i in range(num_samples):
            # Pick a unique final question
            available_final = [r for r in self.pool if r["id"] not in used_final_qs]
            if not available_final:
                used_final_qs.clear()
                available_final = list(self.pool)

            final_qa = self.rng.choice(available_final)
            used_final_qs.add(final_qa["id"])

            final_q_formatted = FINAL_Q_TEMPLATE.format(question=final_qa["question"])
            final_q_tokens = self._q_only_tokens[final_qa["id"]]

            # Build unique exemplars (exclude common prefix IDs + final question)
            exclude = common_used_ids | {final_qa["id"]}
            unique_budget = unique_target - final_q_tokens
            unique_exemplars, _ = self._fill_exemplars(
                max(unique_budget, 1), exclude, tolerance, target_tokens
            )

            # Ensure minimum Q&A pairs in unique suffix
            while len(unique_exemplars) < min_qa_pairs:
                candidates = [
                    r for r in self.pool
                    if r["id"] not in exclude
                    and r["id"] not in {rec["id"] for rec, _ in unique_exemplars}
                ]
                if not candidates:
                    break
                c = self._pick_by_style(candidates)
                qa_tok = self._qa_tokens[c["id"]]
                unique_exemplars.append((c, qa_tok))
                exclude.add(c["id"])

            self.rng.shuffle(unique_exemplars)
            unique_text = self._exemplars_to_text(unique_exemplars)

            # Assemble full prompt
            full_prompt = common_text + unique_text + final_q_formatted
            actual_tokens = _format_count(full_prompt, self.tokenizer, add_special_tokens=True)

            results.append({
                "question": full_prompt,
                "answer": "",
                "question_token_len": actual_tokens,
                "final_question": final_qa["question"],
                "source": f"prefix{prefix_rate}_batch{batch_id}",
                "num_exemplars": len(common_exemplars) + len(unique_exemplars),
                "common_exemplars": len(common_exemplars),
                "unique_exemplars": len(unique_exemplars),
                "target_tokens": target_tokens,
                "prefix_rate": prefix_rate,
                "batch_id": batch_id,
            })

        # Cross-prompt dedup of final questions.
        # Skip when filtered pool is smaller than samples.
        if len(self.pool) >= num_samples:
            from dedup import cross_prompt_dedup
            results = cross_prompt_dedup(results, key="final_question")

        return results
