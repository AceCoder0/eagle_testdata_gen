"""Deduplication utilities for math question pool."""
import hashlib
from typing import List, Dict, Set


def question_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def exact_dedup(records: List[Dict], key: str = "question") -> List[Dict]:
    """Remove records with exact duplicate text (by MD5 hash of stripped text)."""
    seen: Set[str] = set()
    result = []
    for r in records:
        h = question_hash(r.get(key, ""))
        if h not in seen:
            seen.add(h)
            result.append(r)
    return result


def extract_final_question(text: str) -> str:
    """Extract the final question from a few-shot prompt.
    
    The final question is the text after the last 'Q: ' marker,
    before the trailing 'A:' (or end of string).
    """
    marker = "Q: "
    last_q = text.rfind(marker)
    if last_q < 0:
        return text.strip()
    final = text[last_q + len(marker):]
    a_pos = final.rfind("\nA:")
    if a_pos >= 0:
        final = final[:a_pos]
    return final.strip()


def cross_prompt_dedup(records: List[Dict], key: str = "final_question") -> List[Dict]:
    """Ensure uniqueness of a field across records.
    
    For final_question dedup: if two records would generate the same final question,
    skip the duplicate. Uses exact match on the extracted final question field.
    """
    seen: Set[str] = set()
    result = []
    for r in records:
        val = r.get(key, "")
        h = question_hash(val)
        if h not in seen:
            seen.add(h)
            result.append(r)
    return result
