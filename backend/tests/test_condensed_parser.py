import re
import json


def _parse_condensed_points(raw_resp: str):
    if not raw_resp:
        return []
    text = str(raw_resp).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip() if "</think>" in text else text.split("<think>")[0].strip()

    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidates = fence_matches if fence_matches else [text]

    for cand in candidates:
        cand_str = cand.strip()
        start = cand_str.find("[")
        end = cand_str.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_substr = cand_str[start:end + 1]
            try:
                data = json.loads(json_substr)
                if isinstance(data, list):
                    pts = []
                    for item in data:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("polished_text") or item.get("point") or item.get("discussion") or str(item)
                            if t and str(t).strip():
                                pts.append(str(t).strip())
                        elif item and str(item).strip():
                            pts.append(str(item).strip())
                    if pts:
                        return pts
            except Exception:
                try:
                    fixed = re.sub(r",\s*([\]\}])", r"\1", json_substr)
                    data = json.loads(fixed)
                    if isinstance(data, list):
                        pts = [str(x).strip() for x in data if str(x).strip()]
                        if pts:
                            return pts
                except Exception:
                    pass

    bullet_pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        m = re.match(r"^(?:\d+[\.\)]|[-*•])\s*(.+)$", line)
        if m:
            item = m.group(1).strip()
            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                item = item[1:-1].strip()
            if item:
                bullet_pts.append(item)
    return bullet_pts


def test_parse_think_and_fence():
    test1 = '<think>\nI should shorten [Point 1] and [Point 2]\n</think>\n```json\n[\n  "Alice agreed to finish the draft.",\n  "Bob will review."\n]\n```'
    pts = _parse_condensed_points(test1)
    assert pts == ["Alice agreed to finish the draft.", "Bob will review."]


def test_parse_numbered_list():
    test2 = "Here are the points:\n1. Alice agreed to finish the draft.\n2. Bob will review."
    pts = _parse_condensed_points(test2)
    assert pts == ["Alice agreed to finish the draft.", "Bob will review."]


def test_parse_objects():
    test3 = '[{"text": "Alice agreed."}, {"text": "Bob reviewed."}]'
    pts = _parse_condensed_points(test3)
    assert pts == ["Alice agreed.", "Bob reviewed."]


def test_parse_trailing_comma():
    test4 = '[\n  "Alice agreed.",\n  "Bob reviewed.",\n]'
    pts = _parse_condensed_points(test4)
    assert pts == ["Alice agreed.", "Bob reviewed."]
