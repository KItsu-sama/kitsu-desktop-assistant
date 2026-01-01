"""
Trainer module - the ONLY component that may trigger LoRA fine-tuning.

This module requires sanitized, curator-approved datasets. To avoid
accidental training on raw or unsanitized data, the trainer will only
accept dataset paths that have an accompanying '.sanitized' marker file
or that explicitly provide a 'sanitized=True' metadata file next to the
dataset.

Training itself is executed by external tools (finetune scripts). The
trainer only orchestrates and verifies preconditions.
"""
from pathlib import Path
import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, finetune_cmd: Optional[str] = None):
        # finetune_cmd may be a path to a script or a callable; we default
        # to the CPU-friendly script if present.
        self.finetune_cmd = finetune_cmd or "scripts/finetune_lora.py"

    def _is_sanitized(self, dataset_path: Path) -> bool:
        """A dataset is considered sanitized only if a marker file exists:
        e.g., <dataset>.sanitized (zero-length marker) or a <dataset>.meta.json
        with {"sanitized": true}
        """
        marker = dataset_path.with_suffix(dataset_path.suffix + ".sanitized")
        if marker.exists():
            return True
        meta = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
        if meta.exists():
            try:
                import json
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("sanitized") is True:
                    return True
            except Exception:
                pass
        return False

    def train_lora(self, dataset_path: Path, style: Optional[str] = None, dry_run: bool = False) -> bool:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            log.error("Dataset not found: %s", dataset_path)
            return False

        if not self._is_sanitized(dataset_path):
            log.error("Dataset not sanitized: %s - trainer refuses to train", dataset_path)
            return False

        cmd = ["python", self.finetune_cmd, "--data-path", str(dataset_path)]
        if style:
            cmd.extend(["--style", style])

        log.info("Starting training: %s", cmd)
        if dry_run:
            log.info("Dry run - not executing training")
            return True

        try:
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                log.error("Training process failed (exit %s)", res.returncode)
                return False
            log.info("Training completed successfully")
            return True
        except Exception as e:
            log.exception("Training execution failed: %s", e)
            return False
