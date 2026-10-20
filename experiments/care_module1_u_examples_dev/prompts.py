"""Unchanged A1 prompt builder; only the supplied training demonstrations differ."""
from experiments.care_module1_semantic_v8.prompts import prediction_input, parse_output
from experiments.care_module1_semantic_v8.prompts import messages_for as original_messages_for

def messages_for(inputs, arm, examples, draft=None):
    return original_messages_for(inputs, "A1", examples)
