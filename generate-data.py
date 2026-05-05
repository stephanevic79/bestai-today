import json
import sys
from datetime import datetime, timezone

import requests

CONFIG_PATH = "models-config.json"
OUTPUT_PATH = "data.json"

ENDPOINTS = {
    "general": "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=text",
    "code": "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=code",
}


def fetch_leaderboard(url: str) -> list[dict]:
    """Fetch leaderboard API and return list of model entries."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return []

    # The API may wrap data in a dict with key 'data' or return a list directly
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("data") or data.get("leaderboard") or []
        if isinstance(items, list):
            return items
    return []


def deduce_vendor(name: str, org: str | None = None) -> str:
    """Deduce vendor from model name prefix or optional org field."""
    if org:
        return org
    name_lower = name.lower()
    if "gpt-" in name_lower or name_lower.startswith("o1") or name_lower.startswith("o3"):
        return "OpenAI"
    if "claude" in name_lower:
        return "Anthropic"
    if "gemini" in name_lower:
        return "Google"
    if "llama" in name_lower:
        return "Meta"
    if "mistral" in name_lower:
        return "Mistral"
    if "qwen" in name_lower:
        return "Alibaba"
    if "deepseek" in name_lower:
        return "DeepSeek"
    if "yi-" in name_lower or "yi " in name_lower:
        return "01.AI"
    if "command-r" in name_lower:
        return "Cohere"
    if "reka" in name_lower:
        return "Reka"
    if "dbrx" in name_lower:
        return "Databricks"
    # Add more as needed
    return "Unknown"


def main():
    # Load config
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_PATH} not found. Proceeding with empty config.", file=sys.stderr)
        config = []
    config_map = {m["arena_name"]: m for m in config}
    print(f"Loaded {len(config)} models from config.")

    # Fetch leaderboards
    general_data = fetch_leaderboard(ENDPOINTS["general"])
    code_data = fetch_leaderboard(ENDPOINTS["code"])
    print(f"Fetched {len(general_data)} models from General leaderboard.")
    print(f"Fetched {len(code_data)} models from Code leaderboard.")

    # Build code score lookup
    code_scores = {}
    for entry in code_data:
        name = entry.get("name") or entry.get("model_name") or entry.get("arena_name")
        if name and "elo" in entry:
            code_scores[name] = entry["elo"]

    # Process models
    result = []
    found_in_config = 0
    auto_added = 0
    auto_added_list = []

    for entry in general_data:
        arena_name = entry.get("name") or entry.get("model_name") or entry.get("arena_name")
        if not arena_name:
            continue

        score_general = entry.get("elo")
        license_val = entry.get("license", "").lower()

        # Look up in config
        config_entry = config_map.get(arena_name)
        if config_entry:
            # Use metadata from config
            model_name = config_entry["name"]
            vendor = config_entry["vendor"]
            type_val = config_entry["type"]
            url = config_entry["url"]
            found_in_config += 1
        else:
            # Auto-generate metadata
            model_name = arena_name
            org = entry.get("organization")
            vendor = deduce_vendor(arena_name, org)
            type_val = "unlimited" if license_val == "open" else "paid"
            url = "https://lmarena.ai"
            auto_added += 1
            auto_added_list.append(model_name)

        # Lookup code score
        score_code = code_scores.get(arena_name, None)

        entry_out = {
            "name": model_name,
            "vendor": vendor,
            "type": type_val,
            "url": url,
            "score_general": score_general,
            "score_code": score_code,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        }
        result.append(entry_out)

    # Sort by score_general descending (None treated as lowest)
    result.sort(key=lambda x: x["score_general"] if x["score_general"] is not None else float("-inf"), reverse=True)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_arena = len(result)
    print("\n=== Summary ===")
    print(f"Total models from Arena (General): {total_arena}")
    print(f"Found in config (with custom metadata): {found_in_config}")
    print(f"Auto-added (fallback metadata): {auto_added}")
    if auto_added_list:
        print("Auto-added models for review:")
        for m in sorted(auto_added_list):
            print(f"  - {m}")


if __name__ == "__main__":
    main()