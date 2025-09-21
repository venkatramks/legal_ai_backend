from typing import List, Dict, Any
import logging

# Support being imported both as a package (from backend.analysis) and when running
# modules directly (python app.py uses absolute imports). Try relative import first
# then fall back to absolute import to avoid "attempted relative import with no known parent package".
try:
    # prefer local package import; will fall back to lazy import at runtime when
    # services are not available in the build environment
    from .llm_service import LLMService
except Exception:
    try:
        from llm_service import LLMService
    except Exception:
        # Provide a lightweight stub LLMService that signals unavailability.
        # This prevents import-time/runtime-time exceptions and allows the
        # analysis function to gracefully fall back to heuristic clause extraction.
        class LLMService:
            def __init__(self, *args, **kwargs):
                pass
            def is_available(self):
                return False
            # Optional methods that might be called by other code can raise
            # or return safe defaults. We keep them absent where not needed.

def extract_and_score_clauses_from_text(document_text: str) -> List[Dict[str, Any]]:
    """Attempt to extract clauses and score them for risk.

    This function uses LLMService when available; otherwise falls back to simple heuristics.
    Returns a list of dicts: {id, clause_text, risk: 'low'|'medium'|'high', highlights: []}
    """
    llm = LLMService()
    try:
        from .text_cleaner import TextCleaner
    except Exception:
        from text_cleaner import TextCleaner
    cleaner = TextCleaner()

    try:
        # Try LLM-based extraction if available
        if llm.is_available():
            clauses = llm.extract_clauses(document_text)
            # clauses expected as list of {text:..., id:...}
            # Score them with the LLM too
            scored = llm.score_clauses(clauses)
            # Ensure highlights present
            for c in scored:
                if 'highlights' not in c or not c.get('highlights'):
                    # derive simple highlights from clause_text
                    c['highlights'] = derive_highlights_from_clause(c.get('clause_text') or c.get('text') or '')
                # Add heuristic metadata to help UI and reduce false positives
                heur = heuristic_metadata(c.get('clause_text') or c.get('text') or '')
                c.update(heur)
                # Compute numeric score and score components
                score, components = compute_clause_score(c)
                c['score'] = score
                c['score_components'] = components
                # Map numeric score to risk label
                c['risk'] = score_to_label(score)
            return scored
    except Exception as e:
        logging.warning(f"LLM clause extraction failed, falling back to heuristics: {e}")

    # Fallback to heuristic extraction
    return heuristic_extract_and_score(document_text)


def heuristic_extract_and_score(document_text: str) -> List[Dict[str, Any]]:
    """Heuristic-only clause extraction used as a robust fallback when LLM is unavailable.

    This function mirrors the heuristic section of the previous implementation and is
    intentionally kept public so callers can directly invoke it when LLM-driven
    extraction fails.
    """
    # Heuristic fallback: split by double newline or by sentences and produce basic scoring
    parts = [p.strip() for p in document_text.split('\n\n') if len(p.strip()) > 40]
    if not parts:
        # fall back to sentence-based groups
        sentences = [s.strip() for s in document_text.split('.') if len(s.strip()) > 40]
        parts = [sentences[i] + '.' + (sentences[i+1] + '.' if i+1 < len(sentences) else '') for i in range(0, min(len(sentences), 10), 2)]

    results = []
    for idx, part in enumerate(parts[:12]):
        text = part if len(part) < 2000 else part[:2000] + '...'
        score = _heuristic_score(text)
        highlights = derive_highlights_from_clause(text)
        heur = heuristic_metadata(text)
        # Build clause dict then compute numeric score
        clause_obj = {
            'id': f'clause-{idx+1}',
            'clause_text': text,
            'highlights': highlights,
            **heur
        }
        sc, comps = compute_clause_score(clause_obj)
        clause_obj['score'] = sc
        clause_obj['score_components'] = comps
        clause_obj['risk'] = score_to_label(sc)
        results.append(clause_obj)

    return results


def _heuristic_score(text: str) -> str:
    import re
    low_keywords = ['notice', 'information', 'background', 'introduction']
    high_keywords = ['termination', 'penalty', 'liability', 'indemn', 'breach', 'fine', 'penal', 'forfeit']
    medium_keywords = ['obligation', 'payment', 'fee', 'term', 'renewal', 'notice period']

    t = text.lower()

    def count_whole_word_matches(keywords):
        cnt = 0
        for k in keywords:
            if re.search(r'\b' + re.escape(k) + r'\b', t):
                cnt += 1
        return cnt

    high_count = count_whole_word_matches(high_keywords)
    medium_count = count_whole_word_matches(medium_keywords)
    low_count = count_whole_word_matches(low_keywords)

    # Better rules: require >=2 high matches or 1 high match + numeric context for HIGH
    if high_count >= 2:
        return 'high'
    if high_count >= 1 and re.search(r"\$?\d+[\d,\.]*", t):
        return 'high'
    if medium_count >= 1:
        return 'medium'
    if low_count >= 1:
        return 'low'
    return 'low'


def heuristic_metadata(text: str) -> Dict[str, Any]:
    """Return heuristic-derived metadata useful to the UI and for adjusting LLM labels.

    Returns keys:
      - matched_keywords: list[str]
      - heuristic_match_count: int
      - heuristic_confidence: float (0..1)
    """
    import re
    if not text:
        return {'matched_keywords': [], 'heuristic_match_count': 0, 'heuristic_confidence': 0.0}

    high_keywords = ['termination', 'penalty', 'liability', 'indemn', 'breach', 'fine', 'penal', 'forfeit']
    medium_keywords = ['obligation', 'payment', 'fee', 'term', 'renewal', 'notice period']
    low_keywords = ['notice', 'information', 'background', 'introduction']

    t = text.lower()
    matched = []
    for k in high_keywords + medium_keywords + low_keywords:
        if re.search(r'\b' + re.escape(k) + r'\b', t):
            matched.append(k)

    match_count = len(matched)

    # heuristic confidence: simple rule-based score
    conf = 0.0
    if match_count >= 2:
        conf = 0.75
    elif match_count == 1:
        conf = 0.35
    else:
        conf = 0.05

    # Boost if numeric/financial amounts present
    if re.search(r"\$?\d+[\d,\.]*", t) and match_count > 0:
        conf = min(1.0, conf + 0.15)

    return {
        'matched_keywords': matched,
        'heuristic_match_count': match_count,
        'heuristic_confidence': conf
    }


def _label_to_numeric(label: str) -> float:
    mapping = {'low': 0.1, 'medium': 0.5, 'high': 0.9}
    return mapping.get((label or '').lower(), 0.1)


from typing import Tuple

def compute_clause_score(clause: Dict[str, Any], weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
    """Compute a numeric score (0..1) for a clause using heuristic_confidence and optional LLM label.

    Returns (score, components) where components is a dict with each component value.
    """
    if weights is None:
        weights = {'heuristic': 0.5, 'llm_label': 0.5}

    heur_conf = float(clause.get('heuristic_confidence', 0.0))
    llm_label = clause.get('risk') or clause.get('llm_risk') or None
    llm_numeric = _label_to_numeric(llm_label) if llm_label else None

    # If no llm label provided, rely solely on heuristic
    if llm_numeric is None:
        score = heur_conf
        comps = {'heuristic': heur_conf, 'llm_label': 0.0}
        return score, comps

    # Combine normalized components
    h = heur_conf
    l = llm_numeric
    w_h = weights.get('heuristic', 0.5)
    w_l = weights.get('llm_label', 0.5)
    score = (h * w_h) + (l * w_l)
    # normalize by weight sum
    denom = (w_h + w_l) if (w_h + w_l) > 0 else 1.0
    score = score / denom

    comps = {'heuristic': h, 'llm_label': l}
    return score, comps


def score_to_label(score: float) -> str:
    """Map numeric score (0..1) to risk label using thresholds."""
    try:
        s = float(score)
    except Exception:
        return 'low'
    if s >= 0.75:
        return 'high'
    if s >= 0.4:
        return 'medium'
    return 'low'


def derive_highlights_from_clause(text: str) -> List[str]:
    """Create simple highlight snippets from a clause text.

    Strategy:
    - Split into sentences and pick sentences containing risk keywords
    - If none, return the first sentence (trimmed)
    """
    import re
    if not text:
        return []

    keywords = ['termination', 'penalty', 'liability', 'indemn', 'breach', 'payment', 'fine', 'terminate', 'renewal', 'obligation']
    sents = [s.strip() for s in re.split(r'[\.\n]+', text) if s.strip()]
    highlights = []
    for s in sents:
        low = s.lower()
        if any(k in low for k in keywords):
            highlights.append(s[:250])
    if not highlights and sents:
        highlights.append(sents[0][:250])
    return highlights
