"""Loads providers.json and exposes provider/model resolution."""

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv is absent

    def load_dotenv(dotenv_path=None, **_kwargs):
        """Load KEY=VALUE pairs from a .env file without overriding os.environ."""
        if not dotenv_path:
            return False
        path = Path(dotenv_path)
        if not path.is_file():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        return True


# Global development mode flag. Preserved here because score.py and github.py
# import it from this module.
DEVELOPMENT_MODE = True

# Load .env before any os.getenv below, so values apply regardless of import order.
load_dotenv(Path(__file__).parent / ".env")

_CONFIG_PATH = Path(__file__).parent / "providers.json"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _config = json.load(_f)

# Default model, overridable by env.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", _config["default_model"])

# Flat model -> request parameters map. Preserves the contract that
# prompt.MODEL_PARAMETERS exposes to evaluator.py / pdf.py / github.py / score.py.
MODEL_PARAMETERS = {
    model: {k: v for k, v in params.items() if k != "extra_body"}
    for provider in _config["providers"].values()
    for model, params in provider["models"].items()
}


def provider_for(model_name: str) -> dict:
    """Resolve provider config for a model.

    Returns {base_url, api_key, structured_output, extra_body}.
    Raises ValueError if the model is unknown or its required key is unset.
    """
    for name, prov in _config["providers"].items():
        if model_name not in prov["models"]:
            continue
        api_key_env = prov.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key_env and not (api_key and api_key.strip()):
            raise ValueError(
                f"Model '{model_name}' uses provider '{name}', which requires "
                f"env var '{api_key_env}', but it is unset."
            )
        if api_key is not None:
            api_key = api_key.strip()
        extra_body = {
            **prov.get("extra_body", {}),
            **prov["models"][model_name].get("extra_body", {}),
        }
        return {
            "base_url": prov["base_url"].rstrip("/"),
            "api_key": api_key,
            "structured_output": prov.get("structured_output", "json_schema"),
            "extra_body": extra_body,
        }

    available = ", ".join(sorted(MODEL_PARAMETERS))
    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {available}"
    )
