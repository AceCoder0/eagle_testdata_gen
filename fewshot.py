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

    def __init__(self, pool: List[Dict], tokenizer, seed: int = 42):
        self.pool = pool
        self.tokenizer = tokenizer
        random.seed(seed)

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

    def _pick_stratified(self, used_ids: Set[str], sources_used: Set[str]) -> Optional[Dict]:
        """Pick a random Q&A from a source dataset not yet used in this prompt."""
        available_sources = [s for s in self._by_source if s not in sources_used]
        if not available_sources:
            available_sources = list(self._by_source.keys())

        random.shuffle(available_sources)
        for src in available_sources:
            candidates = [r for r in self._by_source[src] if r["id"] not in used_ids]
            if candidates:
                return random.choice(candidates)

        # Fallback: any unused record
        candidates = [r for r in self.pool if r["id"] not in used_ids]
        if candidates:
            return random.choice(candidates)
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

        final_qa = random.choice(available_final)
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
            c = random.choice(candidates)
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
        random.shuffle(exemplars)
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

        # Cross-prompt dedup of final questions
        from dedup import cross_prompt_dedup
        results = cross_prompt_dedup(results, key="final_question")

        return results
