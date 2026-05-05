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
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return []
    # Structure réelle : {"meta": {...}, "models": [...]}
    if isinstance(data, dict):
        return data.get("models", [])
    if isinstance(data, list):
        return data
    return []


def deduce_vendor(arena_name: str, org: str | None = None) -> str:
    if org:
        return org
    n = arena_name.lower()
    if "gpt" in n or n.startswith("o1") or n.startswith("o3") or n.startswith("o4"):
        return "OpenAI"
    if "claude" in n:
        return "Anthropic"
    if "gemini" in n:
        return "Google"
    if "llama" in n or "muse" in n:
        return "Meta"
    if "mistral" in n:
        return "Mistral"
    if "qwen" in n:
        return "Alibaba"
    if "deepseek" in n:
        return "DeepSeek"
    if "grok" in n:
        return "xAI"
    if "glm" in n:
        return "Z.ai"
    if "ernie" in n:
        return "Baidu"
    return "Unknown"


def main():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_PATH} not found. Proceeding with empty config.", file=sys.stderr)
        config = []

    # Clé de lookup : arena_name -> config entry
    config_map = {m["arena_name"]: m for m in config}
    print(f"Loaded {len(config)} models from config.")

    # Fetch leaderboards
    general_data = fetch_leaderboard(ENDPOINTS["general"])
    code_data = fetch_leaderboard(ENDPOINTS["code"])
    print(f"Fetched {len(general_data)} models from General leaderboard.")
    print(f"Fetched {len(code_data)} models from Code leaderboard.")

    # Build code score lookup — champ "model" = arena_name, "score" = elo
    code_scores = {entry["model"]: entry["score"] for entry in code_data if "model" in entry and "score" in entry}

    result = []
    found_in_config = 0
    auto_added = 0
    auto_added_list = []

    for entry in general_data:
        arena_name = entry.get("model")
        if not arena_name:
            continue

        score_general = entry.get("score")
        license_val = entry.get("license", "").lower()
        vendor_api = entry.get("vendor")

        config_entry = config_map.get(arena_name)
        if config_entry:
            model_name = config_entry["name"]
            vendor = config_entry["vendor"]
            type_val = config_entry["type"]
            url = config_entry["url"]
            found_in_config += 1
        else:
            model_name = arena_name
            vendor = vendor_api or deduce_vendor(arena_name)
            type_val = "unlimited" if license_val == "open" else "paid"
            url = "https://lmarena.ai"
            auto_added += 1
            auto_added_list.append(model_name)

        score_code = code_scores.get(arena_name, None)

        result.append({
            "name": model_name,
            "vendor": vendor,
            "type": type_val,
            "url": url,
            "score_general": score_general,
            "score_code": score_code,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        })

    result.sort(key=lambda x: x["score_general"] if x["score_general"] is not None else float("-inf"), reverse=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Total models written: {len(result)}")
    print(f"Found in config: {found_in_config}")
    print(f"Auto-added: {auto_added}")
    if auto_added_list:
        print("Auto-added models for review:")
        for m in sorted(auto_added_list):
            print(f"  - {m}")


if __name__ == "__main__":
    main()
