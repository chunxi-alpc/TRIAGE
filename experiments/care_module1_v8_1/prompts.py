"""V8.1 semantic prompts; frozen V8 A1 demonstrations and schema are reused."""
from experiments.care_module1_semantic_v8.prompts import (
    prediction_input, parse_output, messages_for as original_messages_for,
)

CHECK = """Assess the complete applicability of each rule. Check every required condition and every exception against the patient narrative, preserving the meaning of AND, OR, and negation. An established failed required condition can make an AND rule F even if another condition is unknown; an established satisfied alternative can make an OR rule T even if another alternative is unknown. An established applicable exception makes the rule F. Otherwise distinguish missing or unresolved information from an established negative. Do not infer the patient's status from relatives, historical findings, or neighboring branches. Before assigning T, verify that the complete rule including its exceptions is established. Keep the output in the same short JSON schema; do not output a checklist or intermediate reasoning."""

def messages_for(inputs, arm, examples, draft=None):
    messages = original_messages_for(inputs, 'A1', examples)
    if arm == 'B1':
        messages[0] = dict(messages[0], content=messages[0]['content'] + '\n' + CHECK)
    elif arm == 'B2' and draft is not None:
        messages += [{'role':'assistant', 'content': draft},
                     {'role':'user', 'content': 'Review the preceding applicability judgments against the original narrative and all the complete rules. Correct any wrong T/F/U state and retain any correct state. ' + CHECK + '\nReturn the complete JSON object for every supplied rule.'}]
    return messages
