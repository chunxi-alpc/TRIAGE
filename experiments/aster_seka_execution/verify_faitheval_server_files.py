#!/usr/bin/env python3
"""Server-only byte hash + parquet footer schema/count verification; emits no rows."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_type(value):
    import pyarrow as pa
    if pa.types.is_string(value):
        return "string"
    if pa.types.is_int64(value):
        return "int64"
    if pa.types.is_list(value) or pa.types.is_large_list(value):
        return {"sequence": describe_type(value.value_type)}
    if pa.types.is_struct(value):
        return {field.name: describe_type(field.type) for field in value}
    return str(value)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for task in ("counterfactual", "inconsistent", "unanswerable"):
        parser.add_argument("--" + task, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = []
    for entry in manifest["datasets"]:
        path = getattr(args, entry["task"])
        if path is None:
            continue
        record = {"task": entry["task"], "revision": entry["revision"], "file": str(path), "status": "failed"}
        try:
            actual_size, actual_sha = path.stat().st_size, sha256(path)
            record.update(bytes=actual_size, sha256=actual_sha)
            if actual_size != entry["bytes"] or actual_sha != entry["sha256"]:
                raise ValueError("File bytes differ from the official fixed-revision LFS object")
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)  # Footer only; never read_table/read_row_group.
            schema = {field.name: describe_type(field.type) for field in parquet.schema_arrow}
            record.update(rows=parquet.metadata.num_rows, schema=schema, row_values_read=False)
            if record["rows"] != entry["rows_card"] or schema != entry["schema_card"]:
                raise ValueError("Actual footer differs from the official card schema/count")
            record["status"] = "complete_verified"
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
        results.append(record)
    if not results:
        raise ValueError("Provide at least one downloaded server file")
    outcome = {"status": "complete_verified" if all(x["status"] == "complete_verified" for x in results) else "failed",
               "scope": "bytes and footer metadata only; no test questions/answers emitted", "files": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": outcome["status"], "files": len(results), "output": str(args.output)}))
    return 0 if outcome["status"] == "complete_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
