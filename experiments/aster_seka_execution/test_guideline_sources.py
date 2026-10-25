"""Synthetic CPU tests, to be run only on the server."""
import json
import unittest

from audit_guideline_sources import MetadataIndex, compare_indexes, project_object, source_doi, source_url


class SourceMetadataTests(unittest.TestCase):
    def test_projection_never_returns_reference_or_patient_values(self):
        raw = json.dumps({"guideline_id": 12, "filename": "synthetic.md", "language": "en",
                          "primary_clinical_focus": "synthetic", "paths": [{"gold": "not exported } , \\\""}],
                          "path": {"hidden": "not decoded"}, "answer": "not exported"})
        fields, types = project_object(raw)
        self.assertEqual(fields, {"guideline_id": 12, "filename": "synthetic.md"})
        self.assertEqual(types["paths"], "array")
        self.assertEqual(types["path"], "object")
        self.assertNotIn("not exported", json.dumps([fields, types]))

    def test_url_identity_preserves_query_case_and_scheme(self):
        self.assertEqual(source_url("https://EXAMPLE.org:443/A?id=1#page=4"), "https://example.org/A?id=1")
        self.assertNotEqual(source_url("https://example.org/A?id=1"), source_url("https://example.org/A?id=2"))
        self.assertNotEqual(source_url("https://example.org/A"), source_url("https://example.org/a"))
        self.assertNotEqual(source_url("https://example.org/A"), source_url("http://example.org/A"))
        self.assertIsNone(source_url("a decision path -> an outcome"))
        self.assertIsNone(source_url("prefix https://example.org/A"))
        self.assertEqual(source_doi("https://doi.org/10.1234/ABC"), "10.1234/abc")

    def test_generic_path_url_is_candidate_not_confirmed_identity(self):
        left, right = MetadataIndex(), MetadataIndex()
        left.add({"source_url": "https://example.org/doc"})
        right.add({"path": "https://example.org/doc"})
        result = compare_indexes(left, right)
        self.assertEqual(result["declared_identity_matches"]["declared_url"]["shared_identity_count"], 0)
        self.assertEqual(result["unverified_url_candidate_matches"][0]["shared_candidate_count"], 1)
        self.assertEqual(result["source_independence"], "unknown")
        self.assertEqual(right.report()["path_kind_counts"], {"whole_http_url": 1})

    def test_local_ids_do_not_cross_match_without_namespace(self):
        left, right = MetadataIndex(), MetadataIndex()
        left.add({"guideline_id": 7})
        right.add({"guideline_id": 7})
        first = compare_indexes(left, right)
        self.assertNotIn("declared_namespaced_id", first["declared_identity_matches"])
        left.add({"guideline_id": 7, "source_namespace": "synthetic-registry"})
        right.add({"guideline_id": 7, "source_namespace": "different-registry"})
        self.assertEqual(compare_indexes(left, right)["declared_identity_matches"]["declared_namespaced_id"]["shared_identity_count"], 0)

    def test_unknown_path_is_aggregate_only_and_records_retained(self):
        left, right = MetadataIndex(), MetadataIndex()
        left.add({"source_url": "https://example.org/doc"})
        right.add({"path": {"synthetic secret question": "synthetic secret answer"}})
        right.add({"path": ["synthetic secret decision"]})
        right.add({})
        report = right.report()
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["rows_without_declared_source_identity"], 3)
        self.assertNotIn("synthetic secret", json.dumps(report))
        result = compare_indexes(left, right)
        self.assertTrue(result["all_records_retained"])
        self.assertEqual(result["test_sample_exclusions"], 0)
        self.assertEqual(result["source_independence"], "unknown")


if __name__ == "__main__":
    unittest.main()
