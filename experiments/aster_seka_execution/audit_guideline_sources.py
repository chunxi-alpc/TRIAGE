#!/usr/bin/env python3
"""Server-only metadata projection; no model input/output or labels are inspected.

MedGUIDE: project path and explicit source metadata columns only from Parquet.
MEGA: project top-level provenance/identity fields from case_paths.jsonl. Nested
case/path/reference values are skipped lexically, never decoded or exported.
Only counts, schema/type names, file fingerprints and source hashes are written.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


URL_FIELDS = {"source_url", "guideline_url", "document_url", "original_url"}
ID_FIELDS = {"source_id", "source_document_id", "source_guideline_id", "guideline_id", "document_id"}
NAMESPACE_FIELDS = {"source_namespace", "guideline_namespace", "document_namespace"}
METADATA_FIELDS = URL_FIELDS | ID_FIELDS | NAMESPACE_FIELDS | {"doi", "filename", "path"}
EXPECTED_REVISIONS = {
    "MEGA-CDP": "8f058c88ce1ed43693785b435ad0ff99907a2890",
    "MedGUIDE": "1460ec478280a7295040b92f7eb985327f77dcbe",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashed(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def source_url(value):
    """Conservative identity: preserve scheme, path case, query and slash.

    Fragment is not sent to the origin. A shared URL is location evidence, not
    proof of identical guideline version/content. No network access occurs.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or re.search(r"\s", value):
        return None
    try:
        url = urlsplit(value)
        if url.scheme.lower() not in {"http", "https"} or not url.hostname:
            return None
        if url.username is not None or url.password is not None:
            return None
        port = url.port
        scheme, host = url.scheme.lower(), url.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        if port is not None and not ((scheme == "https" and port == 443) or
                                     (scheme == "http" and port == 80)):
            host += ":" + str(port)
        return urlunsplit((scheme, host, url.path or "/", url.query, ""))
    except ValueError:
        return None


def source_doi(value):
    if not isinstance(value, str):
        return None
    value = re.sub(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip())
    return value.casefold() if re.fullmatch(r"10\.[0-9]{4,9}/[^\s]+", value) else None


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, (int, float)):
        return "number"
    return "other"


def _value_end(text, start):
    """Find a JSON value boundary without decoding nested nonmetadata values."""
    quoted, escaped, depth = False, False, 0
    i = start
    while i < len(text):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                return i
            depth -= 1
        elif char == "," and depth == 0:
            return i
        i += 1
    raise ValueError("Unterminated JSON object; no record contents emitted")


def project_object(text):
    """Return only selected top-level fields, plus top-level type metadata."""
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    if i == len(text) or text[i] != "{":
        raise ValueError("Expected JSON object; no record contents emitted")
    i += 1
    selected, types = {}, {}
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "}":
            if text[i + 1:].strip():
                raise ValueError("Unexpected trailing JSON; no record contents emitted")
            return selected, types
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in types:
            raise ValueError("Invalid or duplicate top-level key")
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text) or text[i] != ":":
            raise ValueError("Expected JSON colon")
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text):
            raise ValueError("Missing JSON value")
        end = _value_end(text, i)
        first = text[i]
        types[key] = ({'"': "string", "{": "object", "[": "array",
                       "n": "null", "t": "boolean", "f": "boolean"}.get(first, "number"))
        if key in METADATA_FIELDS - {"path"}:
            selected[key] = json.loads(text[i:end])
        i = end
        if text[i] == ",":
            i += 1
        elif text[i] != "}":
            raise ValueError("Expected JSON object separator")


class MetadataIndex:
    def __init__(self):
        self.rows = 0
        self.rows_with_declared_identity = 0
        self.types = defaultdict(Counter)
        self.path_kinds = Counter()
        self.path_nested_types = Counter()
        self.path_known_source_keys = Counter()
        self.declared = defaultdict(Counter)
        self.candidate = defaultdict(Counter)
        self.local_ids = defaultdict(set)

    def add(self, row, types=None):
        self.rows += 1
        for field, kind in (types or {key: json_type(value) for key, value in row.items()}).items():
            self.types[field][kind] += 1
        keys = set()
        for field in URL_FIELDS:
            url = source_url(row.get(field))
            if url:
                keys.add(("declared_url", hashed(["url", url])))
        doi = source_doi(row.get("doi"))
        if doi:
            keys.add(("declared_doi", hashed(["doi", doi])))
        namespaces = {str(row[field]).strip() for field in NAMESPACE_FIELDS
                      if isinstance(row.get(field), str) and row[field].strip()}
        for field in ID_FIELDS:
            value = row.get(field)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                self.local_ids[field].add(hashed(value))
                # Differently named ID fields do not automatically share semantics.
                if len(namespaces) == 1:
                    keys.add(("declared_namespaced_id", hashed([next(iter(namespaces)), field, value])))
        for kind, value in keys:
            self.declared[kind][value] += 1
        self.rows_with_declared_identity += bool(keys)
        for field in ("filename", "path"):
            value = row.get(field)
            url = source_url(value)
            if url:
                self.candidate[field + "_whole_url"][hashed(["url", url])] += 1
        if "path" not in row:
            self.path_kinds["missing"] += 1
            return
        value = row["path"]
        kind = json_type(value)
        if isinstance(value, str):
            kind = "whole_http_url" if source_url(value) else "non_url_string"
        self.path_kinds[kind] += 1
        # Nested generic path fields may contain decision text; report only types
        # and recognized metadata key names, never arbitrary keys or values.
        if isinstance(value, list):
            self.path_nested_types.update(json_type(item) for item in value)
        elif isinstance(value, dict):
            self.path_nested_types.update(json_type(item) for item in value.values())
            self.path_known_source_keys.update(set(value) & METADATA_FIELDS)

    def report(self):
        return {
            "rows": self.rows,
            "rows_with_declared_source_identity": self.rows_with_declared_identity,
            "rows_without_declared_source_identity": self.rows - self.rows_with_declared_identity,
            "field_type_counts": {key: dict(value) for key, value in sorted(self.types.items())},
            "path_kind_counts": dict(self.path_kinds),
            "path_nested_value_type_counts": dict(self.path_nested_types),
            "path_recognized_source_key_counts": dict(self.path_known_source_keys),
            "dataset_local_id_unique_counts": {key: len(value) for key, value in self.local_ids.items()},
            "declared_identity_unique_counts": {key: len(value) for key, value in self.declared.items()},
            "unverified_url_candidate_unique_counts": {key: len(value) for key, value in self.candidate.items()},
        }


def compare_indexes(mega, medguide):
    matches = {}
    for kind in sorted(set(mega.declared) | set(medguide.declared)):
        left, right = mega.declared[kind], medguide.declared[kind]
        shared = set(left) & set(right)
        matches[kind] = {"shared_identity_count": len(shared),
                         "identities": [{"identity_sha256": key, "mega_records": left[key],
                                          "medguide_records": right[key]} for key in sorted(shared)]}
    candidates = []
    for left_kind, left in list(mega.candidate.items()) + [("declared_url", mega.declared["declared_url"])]:
        for right_kind, right in list(medguide.candidate.items()) + [("declared_url", medguide.declared["declared_url"])]:
            if left_kind == right_kind == "declared_url":
                continue
            shared = set(left) & set(right)
            if shared:
                candidates.append({"mega_field_kind": left_kind, "medguide_field_kind": right_kind,
                                   "shared_candidate_count": len(shared), "sha256": sorted(shared)})
    return {
        "declared_identity_matches": matches,
        "unverified_url_candidate_matches": candidates,
        "source_independence": "unknown",
        "identical_guideline_version_verified": False,
        "interpretation": "Matching declared IDs/URLs is source-identity/location evidence only. Generic path/filename URL matches require provenance semantics. Missing or zero matches never establish independence; no inferred IDs, titles or disease matching.",
        "all_records_retained": True,
        "test_sample_exclusions": 0,
    }


def audit(config):
    import pyarrow.parquet as pq
    indices, files = {}, {}
    for name in ("MEGA-CDP", "MedGUIDE"):
        root = Path(config[name]["path"])
        if not root.is_dir():
            raise FileNotFoundError("Configured dataset directory unavailable: " + name)
        files[name], indices[name] = [], {}
        if name == "MEGA-CDP":
            for split in ("train_val", "test"):
                path, index = root / split / "case_paths.jsonl", MetadataIndex()
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            row, types = project_object(line)
                            # MEGA path is a reference-path container, not a
                            # documented external source field. Do not decode it.
                            index.add(row, types)
                indices[name][split] = index
                files[name].append({"relative_path": str(path.relative_to(root)),
                                    "sha256": sha256_file(path), "rows": index.rows,
                                    "projected_fields": sorted(set(index.types) & (METADATA_FIELDS - {"path"}))})
        else:
            index = MetadataIndex()
            for path in sorted(root.rglob("*.parquet")):
                pf = pq.ParquetFile(path)
                columns = sorted(set(pf.schema_arrow.names) & METADATA_FIELDS)
                file_rows = 0
                if columns:
                    for batch in pf.iter_batches(columns=columns):
                        for row in batch.to_pylist():
                            index.add(row)
                            file_rows += 1
                else:
                    for _ in range(pf.metadata.num_rows):
                        index.add({})
                    file_rows = pf.metadata.num_rows
                files[name].append({"relative_path": str(path.relative_to(root)),
                                    "sha256": sha256_file(path), "rows": file_rows,
                                    "all_field_names": pf.schema_arrow.names, "projected_fields": columns,
                                    "path_arrow_type": str(pf.schema_arrow.field("path").type) if "path" in pf.schema_arrow.names else None})
            indices[name]["all"] = index
    return {
        "schema_version": "aster.guideline_source_overlap.v1",
        "execution_status": "complete_verified",
        "execution_scope": "metadata audit only; no inference, new restriction or source-independence claim",
        "registered_revisions": {name: config[name].get("revision") for name in indices},
        "expected_revisions": EXPECTED_REVISIONS,
        "registered_revision_matches_expected": {name: config[name].get("revision") == EXPECTED_REVISIONS[name] for name in indices},
        "files": files,
        "datasets": {name: {split: index.report() for split, index in splits.items()} for name, splits in indices.items()},
        "comparisons": {split: compare_indexes(index, indices["MedGUIDE"]["all"])
                        for split, index in indices["MEGA-CDP"].items()},
        "read_scope": {"MedGUIDE": "Parquet footer + explicitly projected path/provenance columns; no prompt/profile/options/answer",
                       "MEGA-CDP": "case_paths.jsonl bytes streamed; nested reference values skipped, only provenance/filename/guideline identifiers decoded; no vignette/gt_paths/guideline text"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["execution_status"], "source_independence": "unknown",
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
