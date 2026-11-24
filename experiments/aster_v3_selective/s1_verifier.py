"""Frozen deterministic verifier. Inputs are condition and current patient text only."""
import re
from functools import lru_cache
from experiments.aster_cr.repair.core import direct_rule, evaluate

HB = "Is preoperative Hb <13.0 g/dL (in either men or women)?"
SBP = "收缩压是否在90~120 mmHg范围内？"
NUMBER = r"(?P<value>\d+(?:\.\d+)?)"
VALUE_UNIT = {"hb": re.compile(NUMBER + r"\s*g/dL\b", re.I),
              "sbp": re.compile(NUMBER + r"\s*mmHg\b", re.I)}
SBP_VALUE = re.compile(r"(?<!动脉)(?<!肺)(?<!左室)(?<!右室)收缩压(?:测得|为|约为|是|约|[:：])?\s*" + NUMBER + r"\s*mmHg\b", re.I)
HB_OBJECT = re.compile(r"\b(?:hemoglobin|Hb)\b", re.I)
PREOP = re.compile(r"pre[- ]?operativ\w*|before surgery|prior to (?:the |planned )*(?:surgery|operation)", re.I)
NEGATIVE = re.compile(r"\b(?:no|never|missing|absent|unknown|not obtained|not measured|postoperative|hypothetical)\b|若|如果|假设|目标|应维持|未测|未知", re.I)
EN_SENTENCE = re.compile(r"(?:\.(?=\s+[A-Z])|[。；;!?\n])")
SEP = re.compile(r"(?<!\d)[<>≤≥]\s*$|\b(?:below|above|under|over|less than|threshold of|target of)\s*$|[–~～-]\s*$", re.I)
HB_TAIL = re.compile(r"(?:level|measurement|concentration|was|is|measured|of|had|been|obtained|and|with|an?|the|which|low|reduced|at|before|surgery|prior|to|operation|showed|demonstrated|anemia|,|\s)*$",re.I)


@lru_cache(None)
def definition(question):
    if question == HB:
        return "hb", {"op":"<", "field":"Hb", "value":13.0}
    if question == SBP:
        return "sbp", direct_rule(question)["predicate"]
    return None, None


def verify(question, patient, simple=False):
    """Return decision, reason, and half-open character offsets; never receives a label."""
    kind, predicate = definition(question)
    if kind is None:
        return None, "outside_scope", None
    field = "Hb" if kind == "hb" else "收缩压"
    if simple:
        match = VALUE_UNIT[kind].search(patient)
        if not match:
            return None, "no_value", None
        decision = evaluate(predicate, {field:float(match["value"])})
        return int(decision), "first_number_with_unit", match.span("value")

    facts = []
    if kind == "hb":
        for interval in re.finditer(r".+?(?=\.(?:\s+[A-Z]|$)|[。；;!?\n]|$)", patient):
            sentence = interval.group()
            if not PREOP.search(sentence) or NEGATIVE.search(sentence):
                continue
            for match in VALUE_UNIT[kind].finditer(sentence):
                before = sentence[:match.start()]
                if SEP.search(before):
                    continue
                names = list(HB_OBJECT.finditer(before))
                if not names or not HB_TAIL.fullmatch(before[names[-1].end():]):
                    continue
                facts.append((float(match["value"]), (interval.start()+match.start("value"), interval.start()+match.end("value"))))
    else:
        for match in SBP_VALUE.finditer(patient):
            prefix = re.split(r"[。，,；;\n]",patient[:match.start()])[-1]
            if NEGATIVE.search(prefix):
                continue
            facts.append((float(match["value"]),match.span("value")))

    if len({x[0] for x in facts}) != 1:
        return None, "missing_or_multiple_values", None
    value, span = facts[0]
    decision = int(evaluate(predicate, {field:value}))
    if kind == "hb":
        contrary = r"(?:not below|not less than|at least|≥|>=)" if decision else r"(?<!not )\b(?:below|less than)\b|(?<![≥>])<"
        conflict = re.search(r"(?:"+contrary+r")\s*13(?:\.0)?\s*g/dL",patient,re.I)
    else:
        contrary = r"(?:未处于|未达到|不在|不处于|超出|不符合)" if decision else r"(?<!未)(?<!不)(?:处于|落在|符合)"
        conflict = re.search(contrary+r".{0,5}90\s*[~～–—-]\s*120\s*mmHg",patient,re.I)
    if conflict:
        return None, "conflicting_comparison", span
    return decision, "observed", span
