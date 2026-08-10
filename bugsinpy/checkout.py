"""
BugsInPy checkout and environment management.

Wraps the BugsInPy Bash tools by delegating to Ubuntu WSL.
Every operation returns a structured result — nothing is silently discarded.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# ─── Status enum ───────────────────────────────────────────────────────────────

class BugStatus(str, Enum):
    """Processing status for each BugsInPy bug. Used in dataset_status.jsonl."""
    AVAILABLE = "AVAILABLE"
    CHECKOUT_FAILED = "CHECKOUT_FAILED"
    INSTALL_FAILED = "INSTALL_FAILED"
    TEST_FAILED_INFRASTRUCTURE = "TEST_FAILED_INFRASTRUCTURE"
    COVERAGE_FAILED = "COVERAGE_FAILED"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    SUCCESSFULLY_PROCESSED = "SUCCESSFULLY_PROCESSED"


@dataclass
class CommandResult:
    """Result of a single WSL shell command."""
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    command: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# ─── Configuration ─────────────────────────────────────────────────────────────

# The WSL distro to use
WSL_DISTRO = "Ubuntu"

# Linux workspace root — repos are checked out here for fast I/O
LINUX_WORKSPACE = "/home/akshay/bugsinpy_workspace"

# Linux path to this project (via /mnt/c/...)
LINUX_PROJECT_ROOT = "/mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY"

# BugsInPy clone location inside WSL
LINUX_BUGSINPY = f"{LINUX_WORKSPACE}/BugsInPy"


# ─── WSL runner ────────────────────────────────────────────────────────────────

def _is_inside_wsl() -> bool:
    """Return True if this process is already running inside WSL/Linux."""
    # WSL sets this env variable; also check /proc/version on Linux
    import os
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# Cache the result — doesn't change within a process lifetime
_INSIDE_WSL: bool = _is_inside_wsl()


def wsl(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 300,
) -> CommandResult:
    """
    Run a bash command in Ubuntu WSL.

    Automatically detects whether it is already running inside WSL. If so,
    commands are executed directly via bash (no nested wsl call). If running
    on Windows, commands are dispatched via ``wsl -d Ubuntu``.

    Args:
        command: Shell command string to run inside bash.
        cwd: Working directory inside WSL (Linux path). If None, uses default.
        timeout: Maximum seconds to wait.

    Returns:
        CommandResult with stdout, stderr, returncode, and elapsed time.
    """
    # Clean Linux PATH — avoids inheriting multi-line Windows PATH via WSL interop
    clean_path = (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ":/home/akshay/.local/bin:/home/akshay/.pyenv/bin"
        f":{LINUX_BUGSINPY}/framework/bin"
    )
    prefix = f"export PATH='{clean_path}' HOME=/home/akshay && "

    if cwd:
        full_command = f"{prefix}cd '{cwd}' && {command}"
    else:
        full_command = f"{prefix}{command}"

    if _INSIDE_WSL:
        # Already inside WSL — run bash directly, no wsl.exe needed
        args = ["bash", "-c", full_command]
    else:
        # Running on Windows — dispatch into WSL
        args = ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", full_command]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        return CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
            command=command,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"TIMEOUT after {timeout}s",
            elapsed_seconds=elapsed,
            command=command,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return CommandResult(
            returncode=-2,
            stdout="",
            stderr=str(exc),
            elapsed_seconds=elapsed,
            command=command,
        )


# ─── BugsInPy operations ───────────────────────────────────────────────────────

def ensure_workspace() -> CommandResult:
    """Create the Linux workspace directory if it doesn't exist."""
    return wsl(f"mkdir -p {LINUX_WORKSPACE}")


def clone_bugsinpy() -> CommandResult:
    """
    Clone the BugsInPy repository into the WSL filesystem if not already present.
    Uses LINUX_BUGSINPY as the target path.
    """
    check = wsl(f"test -d {LINUX_BUGSINPY}/.git && echo EXISTS || echo MISSING")
    if check.ok and "EXISTS" in check.stdout:
        return CommandResult(0, "Already cloned", "", 0.0, "clone_check")

    return wsl(
        f"git clone --depth=1 https://github.com/soarsmu/BugsInPy.git {LINUX_BUGSINPY}",
        timeout=180,
    )


def checkout_bug(
    project: str,
    bug_id: str,
    version: int,
    workspace: str,
) -> CommandResult:
    """
    Run bugsinpy-checkout for a specific bug version.

    Args:
        project: BugsInPy project name (e.g. "thefuck").
        bug_id: Bug identifier (e.g. "1").
        version: 0 for buggy version, 1 for fixed version.
        workspace: Linux directory path to check out into.

    Returns:
        CommandResult.
    """
    bin_path = f"{LINUX_BUGSINPY}/framework/bin"
    cmd = (
        f"export PATH=\"$PATH:{bin_path}\" && "
        f"bugsinpy-checkout -p {project} -v {version} -i {bug_id} -w {workspace}"
    )
    return wsl(cmd, timeout=180)


def install_bug(project_workspace: str, python_version: str = "3.7") -> CommandResult:
    """
    Run bugsinpy-compile (installs dependencies) for a checked-out bug.

    Args:
        project_workspace: The directory containing the checked-out project
                           (i.e. where bugsinpy-compile should be run).
        python_version: The Python version to use for this project (e.g., '3.7' or '3.7.0').

    Returns:
        CommandResult.
    """
    # Many BugsInPy projects specify versions like 3.7.0, we just need the major.minor (3.7)
    major_minor = ".".join(python_version.split(".")[:2])

    bin_path = f"{LINUX_BUGSINPY}/framework/bin"
    # Create a persistent bin folder within the project workspace so the venv's python3 symlink survives
    pybin_dir = f"{project_workspace}/bugsinpy_pybin"

    # Resolve the Python executable — check system paths first, then uv-managed Pythons.
    # uv stores its Pythons at: ~/.local/share/uv/python/cpython-<ver>-linux-x86_64-gnu/bin/python<ver>
    resolve_cmd = (
        f"python{major_minor} --version >/dev/null 2>&1 && echo \"python{major_minor}\" || "
        f"~/.local/bin/uv python find {major_minor} 2>/dev/null || echo ''"
    )
    resolve_result = wsl(resolve_cmd, timeout=15)
    python_exe_resolved = resolve_result.stdout.strip()

    if not python_exe_resolved:
        return CommandResult(
            returncode=1,
            stdout="",
            stderr=(
                f"Python {major_minor} is not available. "
                f"Install it with: uv python install {major_minor} (run inside WSL Ubuntu). "
                f"Bug requires Python {python_version}."
            ),
            elapsed_seconds=0.0,
            command=resolve_cmd,
        )

    # If uv returned a full path, use that directly as the symlink target
    # If it's just "python3.X", resolve to full path
    if python_exe_resolved.startswith("/"):
        symlink_target = python_exe_resolved
    else:
        which_result = wsl(f"which {python_exe_resolved}", timeout=5)
        symlink_target = which_result.stdout.strip() if which_result.ok else f"/usr/bin/{python_exe_resolved}"

    cmd = (
        f"mkdir -p {pybin_dir} && "
        f"ln -sf '{symlink_target}' {pybin_dir}/python3 && "
        f"export PATH=\"{pybin_dir}:$PATH:{bin_path}\" && "
        f"bugsinpy-compile"
    )
    return wsl(cmd, cwd=project_workspace, timeout=600)


def get_installed_python(
    project_workspace: str,
    python_version: str = "3.7",
) -> Optional[str]:
    """
    Detect which Python executable is available in the project's venv/env.

    If no venv is found (e.g. for tox-based projects like cookiecutter where
    bugsinpy-compile does not create a persistent venv), we create one ourselves
    using the appropriate Python version and install bugsinpy_requirements.txt.

    Returns:
        Path string to the Python executable, or None if not found.
    """
    # BugsInPy uses virtualenv under the project directory
    for candidate in [
        f"{project_workspace}/.venv/bin/python",
        f"{project_workspace}/env/bin/python",
        f"{project_workspace}/venv/bin/python",
    ]:
        result = wsl(f"test -x {candidate} && echo YES || echo NO")
        if result.ok and "YES" in result.stdout:
            return candidate

    # No standard venv found — create one using the correct Python version.
    # This handles tox-based projects (e.g. cookiecutter) that don't produce
    # a persistent venv from bugsinpy-compile.
    major_minor = ".".join(python_version.split(".")[:2])
    env_path = f"{project_workspace}/env"
    req_file = f"{project_workspace}/bugsinpy_requirements.txt"

    # Resolve the Python interpreter to use
    resolve_cmd = (
        f"python{major_minor} --version >/dev/null 2>&1 && which python{major_minor} || "
        f"~/.local/bin/uv python find {major_minor} 2>/dev/null || echo ''"
    )
    resolve_result = wsl(resolve_cmd, timeout=15)
    python_exe_for_venv = resolve_result.stdout.strip() or "python3"

    # Create the venv and install requirements
    setup_cmd = (
        f"{python_exe_for_venv} -m venv {env_path} && "
        f"~/.local/bin/uv pip install --python {env_path}/bin/python --upgrade pip -q && "
        f"([ -f {req_file} ] && "
        f"  ~/.local/bin/uv pip install --python {env_path}/bin/python -r {req_file} -q 2>/dev/null || true)"
    )
    setup_result = wsl(setup_cmd, cwd=project_workspace, timeout=300)

    candidate = f"{env_path}/bin/python"
    check = wsl(f"test -x {candidate} && echo YES || echo NO")
    if check.ok and "YES" in check.stdout:
        return candidate

    return None

