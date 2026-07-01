"""
Vocabulary Extractor Service

Provides two extraction modes for uploaded documents:
  1. Rule-Based  — stop-word removal, frequency count, deduplication
  2. AI-Assisted — sends text to QwenProvider and parses JSON response
"""
import io
import json
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Common English stop words ─────────────────────────────────────────────────
_STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each","few",
    "for","from","further","get","got","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
    "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've",
    "if","in","into","is","isn't","it","it's","its","itself","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on",
    "once","only","or","other","ought","our","ours","ourselves","out","over",
    "own","same","shan't","she","she'd","she'll","she's","should","shouldn't",
    "so","some","such","than","that","that's","the","their","theirs","them",
    "themselves","then","there","there's","these","they","they'd","they'll",
    "they're","they've","this","those","through","to","too","under","until","up",
    "very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't",
    "what","what's","when","when's","where","where's","which","while","who",
    "who's","whom","why","why's","will","with","won't","would","wouldn't","you",
    "you'd","you'll","you're","you've","your","yours","yourself","yourselves",
    # meeting-speak
    "also","now","then","well","just","like","said","say","know","think",
    "going","yeah","yes","ok","okay","right","good","great","sure","use",
    "used","using","will","can","may","might","shall","would","one","two",
    "three","four","five","six","seven","eight","nine","ten","next","last",
    "first","second","third","new","old","many","much","way","lot","thing",
    "things","make","made","need","needs","needed","take","takes","taken",
    "come","comes","came","see","sees","saw","look","looks","looked","want",
    "wants","wanted","give","gives","gave","back","even","still","already",
    "always","never","every","each","another","others","however","therefore",
    "thus","hence","whereas","whether","although","because","since","unless",
    "until","while","though","ago","per","via","its","our","their","your",
}


def _extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Extract plain text from PDF, DOCX, TXT or MD files."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=data, filetype="pdf")
            pages = [page.get_text() or "" for page in doc]
            return "\n".join(pages)
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
            except ImportError:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(data))
                    return "\n".join(
                        (page.extract_text() or "") for page in reader.pages
                    )
                except ImportError:
                    logger.warning("[VocabExtractor] No PDF library available (fitz / pdfplumber / PyPDF2)")
                    return ""

    elif ext == "docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.warning("[VocabExtractor] python-docx not installed")
            return ""

    else:
        # TXT / MD / any other text format
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")


def extract_rule_based(filename: str, data: bytes) -> List[str]:
    """
    Rule-based extraction pipeline:
      1. Extract text from file
      2. Lowercase → tokenise by non-alpha chars
      3. Remove stop words, single chars, pure numbers
      4. Count frequency
      5. Return tokens sorted by frequency (desc), deduplicated
    """
    text = _extract_text_from_bytes(filename, data)
    if not text.strip():
        return []

    # Tokenise — keep only alphabetic tokens (allows mixed-case for acronyms)
    tokens = re.findall(r"[A-Za-z]{2,}", text)

    freq: Dict[str, int] = {}
    for tok in tokens:
        lower = tok.lower()
        if lower in _STOP_WORDS:
            continue
        if len(lower) < 2:
            continue
        # Keep original casing for first occurrence
        canonical = freq.get(lower)
        if canonical is None:
            freq[lower] = 0
        freq[lower] += 1

    # Rebuild with original case — use most common casing
    # (we tracked lowercase; reconstruct from tokens preserving original)
    casing: Dict[str, str] = {}
    for tok in tokens:
        lower = tok.lower()
        if lower in freq and lower not in casing:
            casing[lower] = tok

    # Sort by frequency desc, then alpha
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    # Return original casing
    result = [casing.get(w, w) for w, _ in sorted_words]
    logger.info(f"[VocabExtractor] Rule-based: {len(result)} terms from '{filename}'")
    return result


_AI_EXTRACT_SYSTEM_PROMPT = """\
You are a technical terminology extractor. Given a document, extract:
1. Technical/domain-specific terms, product names, framework names, company names, APIs, libraries.
2. Abbreviations and acronyms with their full forms.

Return ONLY valid JSON in this exact format — no explanation, no markdown fences:
{
  "technical_words": ["word1", "word2", ...],
  "shortcuts": [{"short": "ABC", "full": "A Big Company"}, ...]
}
"""

_CHUNK_SIZE = 3000  # chars per chunk sent to LLM


async def extract_ai_assisted(filename: str, data: bytes) -> Dict[str, Any]:
    """
    AI-assisted extraction:
      1. Extract text from file
      2. Chunk text
      3. Send each chunk to QwenProvider
      4. Merge and deduplicate results
      5. Return {technical_words, shortcuts} for user review
    """
    from services.ai_provider import QwenProvider

    text = _extract_text_from_bytes(filename, data)
    if not text.strip():
        return {"technical_words": [], "shortcuts": []}

    # Chunk text
    chunks = [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    # Limit to first 5 chunks to avoid excessive processing
    chunks = chunks[:5]

    all_words: List[str] = []
    all_shortcuts: List[Dict[str, str]] = []
    seen_words: set = set()
    seen_shorts: set = set()

    provider = QwenProvider()

    for idx, chunk in enumerate(chunks):
        prompt = (
            f"{_AI_EXTRACT_SYSTEM_PROMPT}\n\n"
            f"Document chunk {idx + 1}/{len(chunks)}:\n{chunk}"
        )
        try:
            raw = provider.generate(prompt, max_new_tokens=512, temperature=0.1)
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            # Find the JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                logger.warning(f"[VocabExtractor] AI chunk {idx+1}: no JSON found in response")
                continue
            parsed = json.loads(m.group())

            for w in parsed.get("technical_words", []):
                w = str(w).strip()
                if w and w.lower() not in seen_words:
                    seen_words.add(w.lower())
                    all_words.append(w)

            for s in parsed.get("shortcuts", []):
                short = str(s.get("short", "")).strip()
                full = str(s.get("full", "")).strip()
                if short and full and short.lower() not in seen_shorts:
                    seen_shorts.add(short.lower())
                    all_shortcuts.append({"short": short, "full": full})

        except Exception as e:
            logger.warning(f"[VocabExtractor] AI chunk {idx+1} failed: {e}")
            continue

    logger.info(
        f"[VocabExtractor] AI: {len(all_words)} terms, {len(all_shortcuts)} shortcuts "
        f"from '{filename}' ({len(chunks)} chunks)"
    )
    return {"technical_words": all_words, "shortcuts": all_shortcuts}
