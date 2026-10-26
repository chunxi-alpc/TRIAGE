"""Development revision: compact exact anchors, with source-side passage binding."""
INSTRUCTION = '''Extract patient-directed conditional diagnostic or management rules from this guideline only.
Return only a JSON array. Each object has condition_quote, action_quote, exception_quotes, rule_quote.
Copy condition_quote, action_quote, and every exception_quote EXACTLY from the guideline, preserving spelling and punctuation.
condition_quote must identify patient eligibility or a factual predicate. It must NOT be the entire recommendation or include the proposed action. For "For X, recommend Y", quote X as condition and Y as action.
action_quote must name the corresponding recommended action, not a definition, heading, citation or table-of-contents item.
exception_quotes is a list of exact exceptions, or []. Never invent an unstated condition, action or exception.
Set rule_quote to the empty string "". Software will recover the original passage containing the quoted anchors, retaining all intervening text and qualifiers. Do not concatenate or paraphrase source fragments into a new passage.
Keep conditional rules you can identify in source order; omit nonclinical boilerplate. This is extraction, not a patient decision.
Synthetic format example, NOT part of the guideline:
Source: "For adults with marker Q, offer test Z unless exception E applies."
Output: [{"condition_quote":"adults with marker Q","action_quote":"offer test Z","exception_quotes":["unless exception E applies"],"rule_quote":""}]
Synthetic Chinese example, NOT part of the guideline:
Source: "对具有标记Q的成年人，建议进行检查Z；出现情况E时除外。"
Output: [{"condition_quote":"具有标记Q的成年人","action_quote":"进行检查Z","exception_quotes":["出现情况E时除外"],"rule_quote":""}]
Neither example supplies medical knowledge. Extract only from the actual guideline below:
'''
