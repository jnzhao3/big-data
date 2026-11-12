"""Scripted user turns for E3/E4 (BDI-Kit agent test plan, Stage 1). Bump TASK_VERSION when any text changes:
the version is part of the run condition, and the texts' hashes are in every run header."""

from __future__ import annotations

import json

TASK_VERSION = "task_v1"

FOLLOWUP = "Please continue autonomously and finish the task. End with the final table."

# Task B1: small categorical attribute pairs, given (oracle pairs), so value matching is scored apart from SM.
VM_PAIRS = {
    "clark": [("Gender", "gender"), ("Race", "race"), ("Tumor/Normal", "tissue_type"),
              ("Tumor_Focality", "tumor_focality"), ("Grade", "tumor_grade")],
}

_AUTONOMY = ("Work autonomously: do not ask me questions or for confirmation; when something is ambiguous, "
             "decide yourself. Do not materialize any data.")


def task_text(task: str, src: str, study: str) -> str:
    """The first user message. src is the absolute path of the source CSV (no spaces)."""
    if task == "smoke":
        return "List the available schema matching algorithms."
    if task == "sm":
        return (f"Perform schema matching between the source dataset at {src} and the GDC standard "
                '(use target_dataset_path "gdc"). Follow your schema matching workflow and use the tools. '
                f"{_AUTONOMY} Finish with the final matches table.")
    if task == "vm_pairs":
        pairs = [{"source_attribute": s, "target_attribute": t} for s, t in VM_PAIRS[study]]
        return (f"Perform value matching for the source dataset at {src} against the GDC standard "
                '(use target_dataset_path "gdc") for exactly these attribute pairs, as given: '
                f"{json.dumps(pairs)}. Do not run schema matching. Follow your value matching workflow and use "
                f"the tools. {_AUTONOMY} Finish with the final value matches table.")
    if task == "sm_vm":
        return (f"Harmonize the source dataset at {src} to the GDC standard (use target_dataset_path \"gdc\"): "
                "first perform schema matching and follow your schema matching workflow; then perform value "
                "matching for the matched categorical attributes and follow your value matching workflow. "
                f"Use the tools. {_AUTONOMY} Finish with the final schema matches table and the final value "
                "matches table.")
    raise ValueError(f"unknown task {task!r}")


TASKS = ("smoke", "sm", "vm_pairs", "sm_vm")
TASK_KIND = {"smoke": "none", "sm": "sm", "vm_pairs": "vm", "sm_vm": "sm+vm"}
VM_MODE = {"vm_pairs": "oracle", "sm_vm": "end_to_end"}
