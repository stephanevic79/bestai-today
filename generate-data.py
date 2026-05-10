import json
import sys
from datetime import datetime, timezone
from datasets import load_dataset

CONFIG_PATH = "models-config.json"
OUTPUT_PATH = "data.json"


def load_hf_leaderboard(subset: str) -> dict[str, dict]:
    """Charge un subset HuggingFace et retourne un dict indexé par model_name."""
    try:
        ds = load_dataset(
            "lmarena-ai/leaderboard-dataset",
            subset,
            split="latest",
            trust_remote_code=True
        )
        result = {}
        ranking_date = None
        for row in ds:
            if row.get("category") == "overall":
                name = row["model_name"]
                result[name] = {
                    "rating": row["rating"],
                    "rank": row["rank"],
                    "organization": row.get("organization", ""),
                    "license": row.get("license", ""),
                }
                if ranking_date is None:
                    ranking_date = row.get("leaderboard_publish_date")
        return result, ranking_date
    except Exception as e:
        print(f"Error loading HuggingFace subset '{subset}': {e}", file=sys.stderr)
        return {}, None


def deduce_vendor(model_name: str, organization: str) -> str:
    if organization:
        return organization.title()
    n = model_name.lower()
    if "gpt" in n or n.startswith("o1") or n.startswith("o3") or n.startswith("o4"):
        return "OpenAI"
    if "claude" in n:
        return "Anthropic"
    if "gemini" in n or "gemma" in n:
        return "Google"
    if "llama" in n or "muse" in n:
        return "Meta"
    if "mistral" in n or "mixtral" in n or "magistral" in n:
        return "Mistral"
    if "qwen" in n or "qwq" in n:
        return "Alibaba"
    if "deepseek" in n:
        return "DeepSeek"
    if "grok" in n:
        return "xAI"
    if "glm" in n:
        return "Z.ai"
    if "ernie" in n:
        return "Baidu"
    if "kimi" in n:
        return "Moonshot"
    return "Unknown"


def deduce_type(license_str: str) -> str:
    """Proprietary → paid, tout le reste → free (open source / gratuit)."""
    if not license_str or license_str.lower() == "proprietary":
        return "paid"
    return "free"


def deduce_api_access(model_name: str, vendor: str, license_str: str) -> str:
    """
    Déduit l'accès API (free, freemium, paid) en fonction du fournisseur ou du nom du modèle.
    Règles heuristiques (à compléter selon tes connaissances) :
    - Google, OpenAI, Anthropic, xAI : paid (API généralement payante)
    - DeepSeek, Alibaba (Qwen), Z.ai (GLM), Moonshot (Kimi) : freemium (proposent souvent un accès gratuit limité)
    - Autres : paid par défaut
    """
    if vendor in ("Google", "OpenAI", "Anthropic", "xAI"):
        return "paid"
    if vendor in ("DeepSeek", "Alibaba", "Z.ai", "Moonshot"):
        return "freemium"
    # Quelques cas particuliers par nom
    if "mistral" in model_name.lower():
        return "freemium"
    if "llama" in model_name.lower():
        return "free"  # modèles open source souvent disponibles gratuitement via HF
    return "paid"


def main():
    # Chargement du config
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_PATH} not found. Proceeding with empty config.", file=sys.stderr)
        config = []
    config_map = {m["arena_name"]: m for m in config}
    print(f"Loaded {len(config)} models from config.")

    # Chargement des deux leaderboards HuggingFace
    print("Loading General leaderboard (text/overall)...")
    general_data, ranking_date = load_hf_leaderboard("text")
    print(f"Fetched {len(general_data)} models from General leaderboard.")

    print("Loading Code leaderboard (webdev/overall)...")
    code_data, _ = load_hf_leaderboard("webdev")
    print(f"Fetched {len(code_data)} models from Code leaderboard.")

    # Construction de la liste des modèles
    result = []
    found_in_config = 0
    auto_added = 0
    auto_added_list = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for arena_name, general_entry in general_data.items():
        config_entry = config_map.get(arena_name)
        code_entry = code_data.get(arena_name)

        if config_entry:
            model_name = config_entry["name"]
            vendor = config_entry["vendor"]
            type_val = config_entry["type"]
            url = config_entry["url"]
            # api_access depuis la config si présent, sinon on déduit
            api_access = config_entry.get("api_access", deduce_api_access(arena_name, vendor, general_entry["license"]))
            found_in_config += 1
        else:
            model_name = arena_name
            vendor = deduce_vendor(arena_name, general_entry["organization"])
            type_val = deduce_type(general_entry["license"])
            url = "https://lmarena.ai"
            api_access = deduce_api_access(arena_name, vendor, general_entry["license"])
            auto_added += 1
            auto_added_list.append(arena_name)

        result.append({
            "name": model_name,
            "vendor": vendor,
            "type": type_val,
            "url": url,
            "score_general": general_entry["rating"],
            "score_code": code_entry["rating"] if code_entry else None,
            "api_access": api_access
        })

    # Tri par score_general décroissant
    result.sort(
        key=lambda x: x["score_general"] if x["score_general"] is not None else float("-inf"),
        reverse=True
    )

    # Nouvelle structure de sortie avec métadonnées globales
    output = {
        "ranking_date": str(ranking_date) if ranking_date else now[:10],
        "retrieved_at": now,
        "models": result
    }

    # Sauvegarde
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Ranking date   : {output['ranking_date']}")
    print(f"Retrieved at   : {output['retrieved_at']}")
    print(f"Total models   : {len(result)}")
    print(f"Found in config: {found_in_config}")
    print(f"Auto-added     : {auto_added}")
    if auto_added_list:
        print("Auto-added models for review:")
        for m in sorted(auto_added_list):
            print(f"  - {m}")


if __name__ == "__main__":
    main()
