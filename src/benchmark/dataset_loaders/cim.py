"""CIM (Cross-domain Information Memory) dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from benchmark.config import resolve_entry_configuration
from benchmark.dataset_loaders import Sample
from benchmark.utils import generate_hash_id
from benchmark.work_planner import load_input_file

DEFAULT_INPUT_FILE = Path("benchmark_samples/CIM/partitioned/llama3p3_70b/cim_partitioned_normalized.jsonl")


def _render_prompt(query: str, memories: list[str], task: str, recipient: str) -> str:
    return (
        query
        .replace("{memories}", "\n".join(memories))
        .replace("{task}", task)
        .replace("{recipient}", recipient.lower())
    )


class CIMDataset:
    """Loads CIM entries from the normalized JSONL file and yields Samples.

    required_attributes and forbidden_attributes are read directly from each
    JSONL row (pre-populated by cim_normalize.py).  No labels file is needed
    at runtime.
    """

    def __init__(self, input_file: Path | None = None) -> None:
        self.input_file = input_file or DEFAULT_INPUT_FILE
        if not self.input_file.exists():
            raise FileNotFoundError(f"CIM input file not found: {self.input_file}")

        self._entries = load_input_file(self.input_file)
        if not self._entries:
            raise ValueError(f"No CIM entries found in {self.input_file}")

    def __iter__(self) -> Iterator[Sample]:
        for row in self._entries:
            name = row["name"]
            memories = row["memories"]
            task = row["task"]
            recipient = row["recipient"]
            query_template = row["query"]

            prompt = _render_prompt(query_template, memories, task, recipient)
            sample_id = generate_hash_id(memories, prompt)

            yield Sample(
                sample_id=sample_id,
                prompt=prompt,
                memories=memories,
                required_attributes=row.get("required_attributes", []),
                forbidden_attributes=row.get("forbidden_attributes", []),
                metadata={
                    "failure_type": resolve_entry_configuration(row),
                    "name": name,
                    "cim_task": task,
                    "cim_recipient": recipient.lower(),
                },
            )
