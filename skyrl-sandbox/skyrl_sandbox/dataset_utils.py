"""Dataset helpers for the SkyRL sandbox."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, load_dataset

from skyrl_sandbox.models import (
    ChatMessage,
    DatasetDefaults,
    DatasetEntry,
    DatasetExportResponse,
    DatasetSpec,
    DatasetSplit,
    RewardSpecConfig,
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
            payload = [self._apply_defaults(entry, spec.defaults) for entry in split.entries]
            if fmt == "json":
                file_path = export_dir / f"{split.name}.json"
                file_path.write_text(json.dumps(payload, indent=2))
            else:
                file_path = export_dir / f"{split.name}.parquet"
                # Fix parquet export - ensure no empty dicts that cause struct errors
                cleaned_payload = []
                for item in payload:
                    cleaned_item = self._clean_for_parquet(item)
                    cleaned_payload.append(cleaned_item)
                dataset = Dataset.from_list(cleaned_payload)
                dataset.to_parquet(str(file_path))

            files.append(
                {
                    "split": split.name,
                    "format": fmt,
                    "path": str(file_path.resolve()),
                }
            )

        return DatasetExportResponse(artifact_dir=str(export_dir.resolve()), files=files)

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

    def _apply_defaults(self, entry: DatasetEntry, defaults: DatasetDefaults) -> Dict[str, Any]:
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
        # Ensure extra_info is never completely empty for parquet compatibility
        if not merged_extra:
            merged_extra = {"_created_by": "skyrl_sandbox"}
        payload["extra_info"] = merged_extra

        if not payload.get("prompt"):
            raise ValueError("Each dataset entry must include at least one prompt message")

        return payload

    def _clean_for_parquet(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Clean an item for parquet export by ensuring no empty structs."""
        import copy

        cleaned = copy.deepcopy(item)

        # Recursively ensure no empty dicts
        def fix_empty_dicts(obj):
            if isinstance(obj, dict):
                if not obj:  # Empty dict
                    return {"_placeholder": True}
                else:
                    return {k: fix_empty_dicts(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_empty_dicts(item) for item in obj]
            else:
                return obj

        return fix_empty_dicts(cleaned)

    def import_from_source(self, source: str, path: str, sample_size: int = None, split: str = None) -> DatasetSpec:
        """Import dataset from external source."""
        if source == "file":
            return self._import_from_file(path, sample_size)
        elif source == "huggingface":
            return self._import_from_huggingface(path, sample_size, split)
        elif source == "skyrl":
            return self._import_from_skyrl_examples(path, sample_size)
        else:
            raise ValueError(f"Unsupported import source: {source}")

    def _import_from_file(self, file_path: str, sample_size: int = None) -> DatasetSpec:
        """Import from local JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        # Try to detect format and convert
        if isinstance(data, list):
            # List of entries - convert to SkyRL format
            entries = []
            for item in data:
                entry = self._convert_to_skyrl_entry(item)
                entries.append(entry)

            # Sample if requested
            if sample_size and len(entries) > sample_size:
                entries = random.sample(entries, sample_size)

            # Create dataset spec
            dataset_name = f"imported_{path.stem}"
            return DatasetSpec(
                dataset_name=dataset_name,
                description=f"Imported from {file_path}",
                defaults=DatasetDefaults(),
                splits=[DatasetSplit(name="train", entries=entries)],
            )
        else:
            raise ValueError("Expected JSON file to contain a list of entries")

    def _import_from_huggingface(self, dataset_path: str, sample_size: int = None, split: str = None) -> DatasetSpec:
        """Import from Hugging Face dataset."""
        try:
            # Parse dataset path (format: "owner/dataset" or "owner/dataset:subset")
            if ":" in dataset_path:
                repo_id, subset = dataset_path.split(":", 1)
            else:
                repo_id = dataset_path
                subset = None

            # Use provided split or default to train
            split_to_load = split or "train"

            # Load dataset with explicit parameters
            if subset:
                dataset = load_dataset(repo_id, name=subset, split=split_to_load)
            else:
                dataset = load_dataset(repo_id, split=split_to_load)

            # Convert to list and sample if needed
            data_list = list(dataset)
            if sample_size and len(data_list) > sample_size:
                data_list = random.sample(data_list, sample_size)

            # Store raw entries for transformation (instead of converting immediately)
            entries = []
            for item in data_list:
                # Keep raw structure but ensure it's JSON serializable
                raw_entry = {
                    "data_source": "imported_raw",
                    "prompt": [{"role": "system", "content": f"Raw data: {json.dumps(item)[:500]}..."}],
                    "env_class": "raw_data",
                    "reward_spec": {"method": "rule", "ground_truth": ""},
                    "extra_info": {"raw_data": item},  # Store the original data here
                }
                entries.append(raw_entry)

            dataset_name = f"hf_{repo_id.replace('/', '_')}"
            if subset:
                dataset_name += f"_{subset}"
            if split and split != "train":
                dataset_name += f"_{split}"

            # Create split name based on the imported split
            split_name = split_to_load

            return DatasetSpec(
                dataset_name=dataset_name,
                description=f"Imported from Hugging Face: {dataset_path} (subset: {subset or 'default'}, split: {split_to_load})",
                defaults=DatasetDefaults(),
                splits=[DatasetSplit(name=split_name, entries=entries)],
            )
        except Exception as e:
            raise ValueError(f"Failed to import from Hugging Face: {e}")

    def _import_from_skyrl_examples(self, example_name: str, sample_size: int = None) -> DatasetSpec:
        """Import from SkyRL example datasets."""
        # Define some example datasets that could be available in SkyRL
        examples = {
            "simple_qa": {
                "description": "Simple Q&A pairs for testing",
                "entries": [
                    {
                        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                        "ground_truth": "4",
                    },
                    {
                        "prompt": [{"role": "user", "content": "What is the capital of France?"}],
                        "ground_truth": "Paris",
                    },
                    {
                        "prompt": [{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
                        "ground_truth": "William Shakespeare",
                    },
                ],
            },
            "math_word_problems": {
                "description": "Basic math word problems",
                "entries": [
                    {
                        "prompt": [
                            {
                                "role": "user",
                                "content": "Sarah has 5 apples. She gives 2 to her friend. How many apples does she have left?",
                            }
                        ],
                        "ground_truth": "3",
                    },
                    {
                        "prompt": [
                            {
                                "role": "user",
                                "content": "A train travels 60 miles in 1 hour. How far will it travel in 3 hours?",
                            }
                        ],
                        "ground_truth": "180",
                    },
                ],
            },
            "coding_tasks": {
                "description": "Simple coding challenges",
                "entries": [
                    {
                        "prompt": [
                            {"role": "user", "content": "Write a function that returns the sum of two numbers."}
                        ],
                        "ground_truth": "def add(a, b):\n    return a + b",
                    }
                ],
            },
        }

        if example_name not in examples:
            available = ", ".join(examples.keys())
            raise ValueError(f"Unknown SkyRL example: {example_name}. Available: {available}")

        example_data = examples[example_name]
        raw_entries = example_data["entries"]

        # Sample if requested
        if sample_size and len(raw_entries) > sample_size:
            raw_entries = random.sample(raw_entries, sample_size)

        # Convert to DatasetEntry objects
        entries = []
        for item in raw_entries:
            prompt_messages = []
            for msg in item["prompt"]:
                prompt_messages.append(ChatMessage(**msg))

            entry = DatasetEntry(
                data_source="skyrl_examples",
                prompt=prompt_messages,
                env_class="custom_env",
                reward_spec=RewardSpecConfig(method="rule", ground_truth=item["ground_truth"]),
                extra_info={"ground_truth": item["ground_truth"]},
            )
            entries.append(entry)

        return DatasetSpec(
            dataset_name=f"skyrl_{example_name}",
            description=example_data["description"],
            defaults=DatasetDefaults(),
            splits=[DatasetSplit(name="train", entries=entries)],
        )

    def _convert_to_skyrl_entry(self, item: Dict[str, Any]) -> DatasetEntry:
        """Convert various formats to SkyRL DatasetEntry."""
        # Try to detect common formats and convert

        # Format 1: Direct SkyRL format
        if "prompt" in item and isinstance(item["prompt"], list):
            prompt_messages = []
            for msg in item["prompt"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    prompt_messages.append(ChatMessage(**msg))
                else:
                    # Fallback: treat as user message
                    prompt_messages.append(ChatMessage(role="user", content=str(msg)))

            return DatasetEntry(
                data_source=item.get("data_source", "imported"),
                prompt=prompt_messages,
                env_class=item.get("env_class", "custom_env"),
                reward_spec=RewardSpecConfig(**(item.get("reward_spec", {}))),
                extra_info=item.get("extra_info", {}),
            )

        # Format 2: Question-answer pairs
        elif "question" in item and "answer" in item:
            prompt = [ChatMessage(role="user", content=item["question"])]
            return DatasetEntry(
                data_source="imported",
                prompt=prompt,
                env_class="custom_env",
                reward_spec=RewardSpecConfig(method="rule", ground_truth=item["answer"]),
                extra_info={"ground_truth": item["answer"]},
            )

        # Format 3: Input-output pairs
        elif "input" in item and "output" in item:
            prompt = [ChatMessage(role="user", content=item["input"])]
            return DatasetEntry(
                data_source="imported",
                prompt=prompt,
                env_class="custom_env",
                reward_spec=RewardSpecConfig(method="rule", ground_truth=item["output"]),
                extra_info={"ground_truth": item["output"]},
            )

        # Format 4: Instruction-response pairs (common in HF datasets)
        elif "instruction" in item and "response" in item:
            prompt = [ChatMessage(role="user", content=item["instruction"])]
            return DatasetEntry(
                data_source="imported",
                prompt=prompt,
                env_class="custom_env",
                reward_spec=RewardSpecConfig(method="rule", ground_truth=item["response"]),
                extra_info={"ground_truth": item["response"]},
            )

        # Format 5: Text field (assume it's a prompt, no ground truth)
        elif "text" in item:
            prompt = [ChatMessage(role="user", content=item["text"])]
            return DatasetEntry(
                data_source="imported",
                prompt=prompt,
                env_class="custom_env",
                reward_spec=RewardSpecConfig(method="rule"),
                extra_info={},
            )

        else:
            raise ValueError(f"Cannot convert item to SkyRL format: {item}")

    def get_dataset_statistics(self, spec: DatasetSpec) -> Dict[str, Any]:
        """Calculate statistics for a dataset spec."""
        stats = {
            "total_splits": len(spec.splits),
            "split_details": [],
            "total_entries": 0,
            "avg_prompt_length": 0,
            "avg_messages_per_entry": 0,
            "data_sources": set(),
            "env_classes": set(),
            "reward_methods": set(),
        }

        all_prompt_lengths = []
        all_message_counts = []

        for split in spec.splits:
            split_stats = {"name": split.name, "entries": len(split.entries), "avg_prompt_length": 0, "avg_messages": 0}

            split_prompt_lengths = []
            split_message_counts = []

            for entry in split.entries:
                # Apply defaults to get complete entry
                complete_entry = self._apply_defaults(entry, spec.defaults)

                # Count messages and calculate prompt length
                message_count = len(entry.prompt)
                prompt_length = sum(len(msg.content) for msg in entry.prompt)

                split_prompt_lengths.append(prompt_length)
                split_message_counts.append(message_count)
                all_prompt_lengths.append(prompt_length)
                all_message_counts.append(message_count)

                # Collect metadata
                stats["data_sources"].add(complete_entry.get("data_source", "unknown"))
                stats["env_classes"].add(complete_entry.get("env_class", "unknown"))

                reward_spec = complete_entry.get("reward_spec", {})
                if isinstance(reward_spec, dict):
                    stats["reward_methods"].add(reward_spec.get("method", "unknown"))

            if split_prompt_lengths:
                split_stats["avg_prompt_length"] = sum(split_prompt_lengths) / len(split_prompt_lengths)
                split_stats["avg_messages"] = sum(split_message_counts) / len(split_message_counts)

            stats["split_details"].append(split_stats)
            stats["total_entries"] += len(split.entries)

        if all_prompt_lengths:
            stats["avg_prompt_length"] = sum(all_prompt_lengths) / len(all_prompt_lengths)
            stats["avg_messages_per_entry"] = sum(all_message_counts) / len(all_message_counts)

        # Convert sets to lists for JSON serialization
        stats["data_sources"] = list(stats["data_sources"])
        stats["env_classes"] = list(stats["env_classes"])
        stats["reward_methods"] = list(stats["reward_methods"])

        return stats

    def get_dataset_preview(self, spec: DatasetSpec, max_entries: int = 3) -> List[Dict[str, Any]]:
        """Get a preview of dataset entries."""
        preview_entries = []

        for split in spec.splits:
            for i, entry in enumerate(split.entries[:max_entries]):
                complete_entry = self._apply_defaults(entry, spec.defaults)

                preview_entry = {
                    "split": split.name,
                    "index": i,
                    "prompt": [
                        {"role": msg.role, "content": msg.content[:100] + ("..." if len(msg.content) > 100 else "")}
                        for msg in entry.prompt
                    ],
                    "data_source": complete_entry.get("data_source"),
                    "env_class": complete_entry.get("env_class"),
                    "reward_spec": complete_entry.get("reward_spec"),
                    "has_extra_info": bool(complete_entry.get("extra_info")),
                }
                preview_entries.append(preview_entry)

                if len(preview_entries) >= max_entries:
                    return preview_entries

        return preview_entries
