"""Dataset helpers for the SkyRL sandbox."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from skyrl_sandbox.models import (
    DatasetDefaults,
    DatasetEntry,
    DatasetExportResponse,
    DatasetSpec,
)


class DatasetManager:
    """Manage dataset specs plus export artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

        self.exports_dir = self.root / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def list_specs(self) -> List[str]:
        """Return all available dataset spec names."""
        return sorted(path.stem for path in self.root.glob("*.json"))

    def load_spec(self, name: str) -> DatasetSpec:
        """Load a dataset spec by name."""
        path = self._spec_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")
        data = json.loads(path.read_text())
        return DatasetSpec(**data)

    def save_spec(self, spec: DatasetSpec) -> Path:
        """Persist a dataset spec to disk."""
        path = self._spec_path(spec.dataset_name)
        path.write_text(spec.model_dump_json(indent=2))
        return path

    def delete_spec(self, name: str) -> None:
        """Delete a stored dataset spec."""
        path = self._spec_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")
        path.unlink()

    def export(self, spec: DatasetSpec, fmt: str = "json") -> DatasetExportResponse:
        """Export dataset splits to json or parquet artifacts."""
        if fmt not in {"json", "parquet"}:
            raise ValueError("format must be either 'json' or 'parquet'")

        export_dir = self.exports_dir / spec.dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)

        files: List[Dict[str, Any]] = []
        if not spec.splits:
            raise ValueError("Dataset spec must include at least one split")

        for split in spec.splits:
            if not split.entries:
                continue
            payload = [
                self._apply_defaults(entry, spec.defaults) for entry in split.entries
            ]
            if fmt == "json":
                file_path = export_dir / f"{split.name}.json"
                file_path.write_text(json.dumps(payload, indent=2))
            else:
                file_path = export_dir / f"{split.name}.parquet"
                dataset = Dataset.from_list(payload)
                dataset.to_parquet(str(file_path))

            files.append(
                {
                    "split": split.name,
                    "format": fmt,
                    "path": str(file_path.resolve()),
                }
            )

        return DatasetExportResponse(
            artifact_dir=str(export_dir.resolve()), files=files
        )

    def list_exports(self) -> List[Dict[str, Any]]:
        """Enumerate exported datasets and available files."""
        exports: List[Dict[str, Any]] = []
        for dataset_dir in sorted(self.exports_dir.glob("*")):
            if not dataset_dir.is_dir():
                continue
            files = [
                {
                    "dataset": dataset_dir.name,
                    "split": file.stem,
                    "format": file.suffix.lstrip("."),
                    "path": str(file.resolve()),
                }
                for file in sorted(dataset_dir.glob("*.*"))
                if file.is_file()
            ]
            if files:
                exports.append({"dataset": dataset_dir.name, "files": files})
        return exports

    def _spec_path(self, name: str) -> Path:
        clean = name.replace(" ", "_")
        return self.root / f"{clean}.json"

    def _apply_defaults(
        self, entry: DatasetEntry, defaults: DatasetDefaults
    ) -> Dict[str, Any]:
        """Merge per-entry overrides with dataset defaults."""
        payload = entry.model_dump()

        payload["data_source"] = payload.get("data_source") or defaults.data_source
        payload["env_class"] = payload.get("env_class") or defaults.env_class

        reward_spec = defaults.reward_spec.model_copy(deep=True).model_dump()
        entry_reward = payload.get("reward_spec") or {}
        for key, value in entry_reward.items():
            if value is not None:
                reward_spec[key] = value
        payload["reward_spec"] = reward_spec

        merged_extra = defaults.extra_info.copy()
        merged_extra.update(payload.get("extra_info") or {})
        payload["extra_info"] = merged_extra

        if not payload.get("prompt"):
            raise ValueError("Each dataset entry must include at least one prompt message")

        return payload

