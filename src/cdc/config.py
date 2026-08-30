"""Configuration loading.

Every experimental parameter lives in config/*.yaml so the pre-registered
analysis plan can cite exact values and reviewers can diff them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# repo root = three levels up from this file (src/cdc/config.py)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

# `.env` is the documented name; `.env.local` is the convention many people
# reach for by habit. Support both, local winning, so a misnamed file is a
# non-event rather than a confusing "key missing" an hour into debugging.
for _envfile in (".env", ".env.local"):
    load_dotenv(ROOT / _envfile, override=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "settings.yaml")


@lru_cache(maxsize=1)
def channels() -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "channels.yaml")


def path_for(key: str) -> Path:
    """Resolve a storage path from settings, relative to the repo root."""
    rel = settings()["storage"][key]
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Secrets:
    youtube_api_key: str | None
    scrapecreators_api_key: str | None
    scrapecreators_api_keys: tuple[str, ...] = ()

    def require_youtube(self) -> str:
        if not self.youtube_api_key:
            raise RuntimeError(
                "YOUTUBE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set it as a GitHub Actions secret for CI runs."
            )
        return self.youtube_api_key

    def require_scrapecreators(self) -> str:
        if not self.scrapecreators_api_key:
            raise RuntimeError(
                "SCRAPECREATORS_API_KEY is not set. Instagram collection cannot run."
            )
        return self.scrapecreators_api_key

    def require_scrapecreators_keys(self) -> tuple[str, ...]:
        """All configured keys, in priority order.

        Credits are per-key and non-transferable, so a second key is a second
        wallet rather than a bigger one. Holding them all up front lets the
        client roll over the moment one is spent, instead of the collection
        stopping until a human notices and swaps a secret.
        """
        if not self.scrapecreators_api_keys:
            raise RuntimeError(
                "No SCRAPECREATORS_API_KEY set. Instagram collection cannot run."
            )
        return self.scrapecreators_api_keys


def secrets() -> Secrets:
    # SCRAPECREATORS_API_KEY, then _2, _3, ... in that order. Numbered keys are
    # spare wallets used only once the ones before them are spent.
    keys: list[str] = []
    primary = os.environ.get("SCRAPECREATORS_API_KEY")
    if primary:
        keys.append(primary.strip())
    for n in range(2, 10):
        k = os.environ.get(f"SCRAPECREATORS_API_KEY_{n}")
        if k and k.strip():
            keys.append(k.strip())
    # Deduplicate while preserving order: pasting the same key twice would
    # otherwise look like extra budget that does not exist.
    keys = list(dict.fromkeys(keys))
    return Secrets(
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
        scrapecreators_api_key=keys[0] if keys else None,
        scrapecreators_api_keys=tuple(keys),
    )
