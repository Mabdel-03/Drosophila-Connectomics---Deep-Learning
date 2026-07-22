"""Shared helpers for the Stage-5 campaign wrapper scripts.

The consortium runner invokes a stage's ``launcher_script`` with ONLY
``--workspace <ws> --planning-config <json>`` (runner.py:299-304); stage parameters
arrive via the subprocess environment (``stage.env``). These wrappers therefore parse
``--workspace`` (and ignore ``--planning-config``), read parameters from ``os.environ``,
read upstream artifacts that the runner copied into ``<ws>``, and write their
``success_artifacts`` back into ``<ws>``.

LLM stages (propose/verify/report) shell out to the Claude Code CLI; the extract stage
submits the SLURM extract job (or runs it inline). Keeping this glue thin and dependency-
free (only stdlib) means the wrappers run under the consortium env without importing
flyconn.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

CONNECTOMICS_REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")

# Stage-5 data root (CAVE cache + MCNS intermediates) on the GROUP DATA volume: personal
# scratch is over quota. Stage 5 only WRITES here (it queries CAVE live; it does not need
# the 48G offline FAFB feather, which stays on scratch for stages 1-4). Overridable via env.
STAGE5_DATA_ROOT = "/orcd/data/tpoggio/001/mabdel03/connectome_data"


def stage5_data_root() -> str:
    return os.environ.get("FLYCONN_DATA_ROOT", STAGE5_DATA_ROOT)


def parse_workspace_args(argv=None) -> argparse.Namespace:
    """Standard wrapper arg parse: --workspace (used) + --planning-config (ignored)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--planning-config", default="{}")  # accepted + ignored
    return ap.parse_known_args(argv)[0]


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def stage_artifacts() -> Path:
    """The canonical Stage-5 output dir in the repo (source of truth the wrappers mirror)."""
    d = CONNECTOMICS_REPO / "5 - Muscular Projection"
    d.mkdir(parents=True, exist_ok=True)
    return d


def copy_into_workspace(ws: Path, names: list[str]) -> list[str]:
    """Copy named Stage-5 artifacts from the repo dir into the stage workspace.

    Returns the names actually copied. Lets a stage emit its success_artifacts into the
    workspace (which the runner validates) while the canonical files live in the repo.
    """
    src = stage_artifacts()
    copied = []
    for n in names:
        s = src / n
        if s.exists():
            shutil.copy2(s, ws / n)
            copied.append(n)
    return copied


DEFAULT_CAVE_ENV = "/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave"


def cave_python() -> str:
    """The interpreter for stage-5 flyconn work: the dedicated flyconn_cave env.

    Resolved from FLYCONN_CAVE_ENV (set by the campaign env / common.sh), falling back to
    the default prefix, then to whatever `python` is on PATH. The campaign-overseer drives
    these wrappers from its own conda context, so we must NOT rely on the ambient `python`.
    """
    prefix = os.environ.get("FLYCONN_CAVE_ENV", DEFAULT_CAVE_ENV)
    cand = Path(prefix) / "bin" / "python"
    if cand.exists():
        return str(cand)
    return shutil.which("python") or "python"


def run_flyconn(subcmd: list[str]) -> int:
    """Run `python -m flyconn.<subcmd>` in the Connectomics repo with src on PYTHONPATH."""
    e = dict(os.environ)
    e["PYTHONPATH"] = str(CONNECTOMICS_REPO / "src") + os.pathsep + e.get("PYTHONPATH", "")
    # Pin the Stage-5 data root to the group volume unless explicitly overridden, so the
    # CAVE cache + MCNS intermediates land somewhere writable (scratch is over quota).
    e.setdefault("FLYCONN_DATA_ROOT", stage5_data_root())
    return subprocess.call(
        [cave_python(), "-u", "-m", *subcmd], cwd=str(CONNECTOMICS_REPO), env=e
    )


def claude_code(task_text: str, *, workspace: Path, allow_edits: bool, model: str,
                effort: str = "high", max_turns: int = 30, budget_cents: int = 1000) -> int:
    """Invoke the Claude Code CLI on a task, scoped to the workspace.

    propose/report run with edits allowed (they author artifacts in the workspace);
    verify runs the flyconn verifier directly (no LLM verdict authoring) so we do NOT
    expose edit permissions there. Returns the CLI exit code; falls back to a
    deterministic flyconn call if the CLI is unavailable.
    """
    binary = shutil.which("claude")
    if not binary:
        return 127  # caller decides on a deterministic fallback
    perm = "bypassPermissions" if allow_edits else "plan"
    cmd = [
        binary, "-p", "--output-format", "text", "--model", model,
        "--effort", effort, "--permission-mode", perm, "--max-turns", str(max_turns),
    ]
    e = dict(os.environ)
    e["CLAUDE_CODE_MAX_COST_CENTS"] = str(budget_cents)
    proc = subprocess.run(cmd, cwd=str(workspace), env=e,
                          input=task_text, text=True)
    return proc.returncode


def task_file_text(name: str) -> str:
    """Read a campaign task file from campaigns/automation_tasks/."""
    p = CONNECTOMICS_REPO / "campaigns" / "automation_tasks" / name
    return p.read_text() if p.exists() else ""
