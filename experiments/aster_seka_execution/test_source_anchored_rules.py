import json
import unittest
from source_anchored_rules import enclosing_source_span, parse_source_anchored_rules


class SourceAnchors(unittest.TestCase):
    def item(self, c="if C", a="do A", passage="invented join", exceptions=None):
        return {"condition_quote": c, "action_quote": a, "rule_quote": passage,
                "exception_quotes": exceptions or []}

    def test_original_intervening_exception_and_newline_preserved(self):
        text = "if C\nexcept E; then do A."
        rows, audit = parse_source_anchored_rules(json.dumps([self.item(exceptions=["except E"])]), text)
        self.assertEqual(rows[0].rule_quote, "if C\nexcept E; then do A")
        self.assertEqual(audit[0]["status"], "original_source_span_reconstructed")

    def test_missing_or_changed_anchor_is_never_filled(self):
        for item in [self.item(a="do B"), self.item(c="if  C")]:
            rows, _ = parse_source_anchored_rules(json.dumps([item]), "if C then do A")
            self.assertFalse(rows)

    def test_action_leak_and_truncated_tail_remain_rejected(self):
        rows, _ = parse_source_anchored_rules(json.dumps([self.item(c="if C then do A")]), "if C then do A")
        self.assertFalse(rows)
        rows, audit = parse_source_anchored_rules('[{"condition_quote":"if C"', "if C then do A")
        self.assertFalse(rows)
        self.assertTrue(audit[0]["truncated"])

    def test_shortest_span_then_earliest_tie(self):
        text = "if C ------------ do A; if C do A; if C do A"
        start, end = enclosing_source_span(text, ["if C", "do A"])
        self.assertEqual(text[start:end], "if C do A")
        self.assertEqual(start, text.index("if C do A"))

    def test_existing_valid_rule_is_unchanged(self):
        text = "if C then do A"
        rows, audit = parse_source_anchored_rules(json.dumps([self.item(passage=text)]), text)
        self.assertEqual(rows[0].rule_quote, text)
        self.assertEqual(audit[0]["status"], "quotes_validated")

    def test_explicit_empty_passage_is_bound_to_original_source(self):
        text = "if C then do A"
        rows, audit = parse_source_anchored_rules(json.dumps([self.item(passage="")]), text)
        self.assertEqual(rows[0].rule_quote, text)
        self.assertEqual(audit[0]["status"], "original_source_span_reconstructed")


if __name__ == "__main__":
    unittest.main()
