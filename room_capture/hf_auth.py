"""HuggingFace authentication helpers for the room_capture pipeline.

Resolves an HF token in this priority order (first hit wins):

    1. Explicit argument (e.g. from --hf-token CLI flag)
    2. Environment: HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN
    3. .env file at the package root (loaded only if python-dotenv is installed)
    4. HuggingFace CLI cached login at ~/.cache/huggingface/token

If a token is found from sources 1-3, we export it as HF_TOKEN *and* call
``huggingface_hub.login()`` so every downstream caller (DA3, safetensors,
snapshot_download) picks it up automatically.

Source 4 needs no action on our part: ``huggingface_hub`` already reads it
transparently. We merely detect and report it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ENV_VAR_CANDIDATES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


@dataclass
class HFAuthResult:
    authenticated: bool
    source: str  # "cli-arg" | "env:HF_TOKEN" | ".env" | "cli-cache" | "anonymous"
    username: Optional[str]  # filled in only when we successfully whoami()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _token_from_env() -> tuple[str | None, str | None]:
    for var in _ENV_VAR_CANDIDATES:
        v = os.environ.get(var)
        if v:
            return v, f"env:{var}"
    return None, None


def _token_from_dotenv(dotenv_path: Path) -> tuple[str | None, str | None]:
    if not dotenv_path.is_file():
        return None, None
    try:
        from dotenv import dotenv_values  # type: ignore
    except ImportError:
        # Silent minimal parser fallback so we don't require python-dotenv.
        values: dict[str, str] = {}
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    else:
        values = {k: v for k, v in dotenv_values(dotenv_path).items() if v is not None}

    for var in _ENV_VAR_CANDIDATES:
        v = values.get(var)
        if v:
            return v, ".env"
    return None, None


def _token_from_cli_cache() -> tuple[str | None, str | None]:
    """Read the HF CLI cached token (~/.cache/huggingface/token)."""
    try:
        from huggingface_hub import HfFolder  # type: ignore

        tok = HfFolder.get_token()
        if tok:
            return tok, "cli-cache"
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def setup_hf_token(
    explicit_token: str | None = None,
    dotenv_path: Path | None = None,
    verbose: bool = True,
) -> HFAuthResult:
    """Resolve and install an HF token for the current process.

    Side effects when a token is found from sources 1-3:
        - sets os.environ["HF_TOKEN"]
        - calls huggingface_hub.login(token=..., add_to_git_credential=False)
    """
    token: str | None = None
    source: str = "anonymous"

    if explicit_token:
        token, source = explicit_token, "cli-arg"

    if token is None:
        token, maybe_source = _token_from_env()
        if token:
            source = maybe_source or "env"

    if token is None and dotenv_path is not None:
        token, maybe_source = _token_from_dotenv(dotenv_path)
        if token:
            source = maybe_source or ".env"

    # If we still don't have one, fall through to the CLI cache (HF reads it
    # automatically, we just detect it for the status line).
    if token is None:
        cache_token, maybe_source = _token_from_cli_cache()
        if cache_token:
            token = cache_token
            source = maybe_source or "cli-cache"

    if token is None:
        if verbose:
            print("[hf_auth] No HF token detected. Model will be downloaded "
                  "anonymously (rate-limited, gated models will fail).")
        return HFAuthResult(authenticated=False, source="anonymous", username=None)

    # Only call login() when the token came from us. If it came from the CLI
    # cache, huggingface_hub will reuse it automatically - no need to duplicate.
    if source != "cli-cache":
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        try:
            from huggingface_hub import login  # type: ignore

            login(token=token, add_to_git_credential=False)
        except Exception as e:
            if verbose:
                print(f"[hf_auth] huggingface_hub.login() failed ({e}); "
                      "continuing with env-var only.")

    # Try a whoami() for a friendly confirmation line.
    username: str | None = None
    try:
        from huggingface_hub import whoami  # type: ignore

        who = whoami(token=token if source != "cli-cache" else None)
        if isinstance(who, dict):
            username = who.get("name") or who.get("fullname")
    except Exception:
        pass

    if verbose:
        who_str = f" as {username}" if username else ""
        redacted = (token[:4] + "..." + token[-4:]) if len(token) >= 12 else "***"
        print(f"[hf_auth] Authenticated{who_str} via {source} (token={redacted}).")

    return HFAuthResult(authenticated=True, source=source, username=username)
