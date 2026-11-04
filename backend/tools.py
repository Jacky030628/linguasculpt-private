"""Deterministic, inspectable editing tools. Scores are not AI probabilities."""
import re
from collections import Counter
from difflib import SequenceMatcher
from langchain_core.tools import tool


@tool
def analyze_style(text: str) -> dict:
    """Measure sentence lengths and repeated words; does not detect AI authorship."""
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    lengths = [len(re.findall(r"\b[\w'-]+\b", s)) for s in re.split(r'[.!?]+', text) if s.strip()]
    return {'words': len(words), 'sentence_lengths': lengths,
            'repeated_words': dict(Counter(words).most_common(12))}


@tool
def extract_protected_spans(text: str, terms: list[str]) -> dict:
    """Extract exact user terms, numbers, quoted spans and parenthetical citations."""
    automatic = re.findall(r'\b\d+(?:[.,]\d+)*(?:%)?|[“"][^“”"\n]+[”"]|\([^()]*\b(?:19|20)\d{2}[^()]*\)', text)
    return {'spans': list(dict.fromkeys([t for t in terms if t.strip()] + automatic))}


@tool
def check_constraints(source: str, draft: str, spans: list[str]) -> dict:
    """Check exact protected-span counts and newly introduced numbers."""
    issues = []
    for span in spans:
        if source.count(span) != draft.count(span):
            issues.append(f'Protected span count changed: {span}')
    nums = lambda s: Counter(re.findall(r'\b\d+(?:[.,]\d+)*(?:%)?', s))
    if nums(source) != nums(draft):
        issues.append('Numbers changed, disappeared, or were introduced')
    if not draft.strip():
        issues.append('Draft is empty')
    return {'passed': not issues, 'issues': issues}


@tool
def compare_drafts(source: str, draft: str) -> dict:
    """Return actual changed text spans rather than model-reported edits."""
    from .change_records import change_records
    return {'edits': change_records(source,draft)}


@tool
def analyze_paper_features(text: str) -> dict:
    """Compute the current paper's 65 features, pruned-fusion probability and local contributions. Applicable to 80–1500 English words. Contributions are associations, not causal editing rules."""
    from .paper import score
    return score(text)


TOOLS = [analyze_style, extract_protected_spans, check_constraints, compare_drafts, analyze_paper_features]
