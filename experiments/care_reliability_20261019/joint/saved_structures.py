"""Validate fixed structure artifacts before using the original Qwen runtime."""
from __future__ import annotations
import json
from pathlib import Path
from .context import prediction_input, prompt_sha256
from .structures import normalize_structure
from .run_structures import file_digest, implementation_hashes, digest


class SavedStructureProvider:
    def __init__(self, source, run_paths, data_sha256, split):
        if source not in {"v6", "v7"}:
            raise ValueError("Only predicted structures may be loaded from model runs")
        self.source = source
        self.instance = None
        self.entries = {}
        self.provenance = []
        current = implementation_hashes()
        for directory in run_paths:
            directory = Path(directory).resolve()
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            complete = json.loads((directory / "COMPLETE.json").read_text())
            if complete["manifest_sha256"] != file_digest(manifest_path) or manifest["status"] != "completed":
                raise ValueError("Structure run is incomplete or changed")
            contract = manifest["contract"]
            provenance_path = directory / "immutable_provenance.json"
            if manifest["immutable_provenance_sha256"] != file_digest(provenance_path):
                raise ValueError("Immutable structure provenance changed")
            immutable = json.loads(provenance_path.read_text())
            if immutable != {"contract": contract, "versions": manifest["versions"], "models": manifest["models"]}:
                raise ValueError("Structure runtime or model provenance differs from immutable record")
            if digest(json.loads((directory / "request_manifest.json").read_text())) != contract["request_manifest_sha256"]:
                raise ValueError("Structure request manifest changed")
            if contract["data_sha256"] != data_sha256 or contract["split"] != split:
                raise ValueError("Structure data or split differs from requested run")
            if contract["implementation_sha256"] != current:
                raise ValueError("Structure implementation changed after inference")
            if manifest["generation_journal_sha256"] != file_digest(directory / "trialmatch_generations.jsonl"):
                raise ValueError("Structure model generation journal changed")
            if not manifest.get("active_adapters") and not manifest.get("reused_completed_requests"):
                raise ValueError("No active adapter provenance in structure run")
            for case_id in contract["case_ids"]:
                if case_id in self.entries:
                    raise ValueError("Case occurs in more than one structure shard")
                relative = Path("structures") / source / (case_id + ".json")
                expected = manifest["structure_sha256"].get(str(relative))
                if not expected:
                    raise ValueError("Missing structure file digest")
                self.entries[case_id] = (directory / relative, expected)
            self.provenance.append({"run": str(directory), "manifest_sha256": file_digest(manifest_path),
                                    "python_executable": manifest["python_executable"],
                                    "versions": manifest["versions"], "models": manifest["models"]})

    def load(self, shared_adapters=None):
        return None

    def predict(self, case):
        if case["case_id"] not in self.entries:
            raise KeyError("No actual model structure for requested case")
        path, expected = self.entries[case["case_id"]]
        if file_digest(path) != expected:
            raise ValueError("Saved structure file changed")
        row = json.loads(path.read_text())
        if row["case_id"] != case["case_id"]:
            raise ValueError("Saved structure case ID mismatch")
        if row["input_sha256"] != prompt_sha256(json.dumps(prediction_input(case), sort_keys=True)):
            raise ValueError("Saved structure input mismatch")
        normalized = normalize_structure(row["raw"], case, self.source)
        if normalized != row["normalized"]:
            raise ValueError("Saved normalized structure differs from raw model graph")
        return normalized, row["raw"]

