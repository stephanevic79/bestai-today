import json
import sys
import re
from datetime import datetime, timezone
from datasets import load_dataset
import requests

CONFIG_PATH = "models-config.json"
OUTPUT_PATH = "data.json"
FREE_API_README_URL = "https://raw.githubusercontent.com/mnfst/awesome-free-llm-apis/main/README.md"


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


def normalize_name(name: str) -> str:
    """Normalise un nom pour la comparaison : minuscules, sans ponctuation ni espaces multiples."""
    name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)
    name = re.sub(r'[^a-z0-9\s]', '', name.lower())
    name = re.sub(r'\s+', '', name)
    return name


def clean_provider_name(raw: str) -> str:
    """Nettoie le nom d'un fournisseur (issu du titre de section markdown)."""
    raw = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw)
    raw = raw.replace('[', '').replace(']', '')
    raw = re.sub(r'\([^)]*\)', '', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def fetch_free_api_data():
    """
    Récupère la liste des modèles avec API gratuite, leurs fournisseurs et les liens
    d'obtention de clé API depuis le README. Les liens sont extraits des titres de section.
    Retourne un dict : { normalized_model_name: [ { "provider": "...", "url": "..." } ] }
    """
    print(f"Fetching free API data from {FREE_API_README_URL}...")
    try:
        response = requests.get(FREE_API_README_URL, timeout=30)
        response.raise_for_status()
        content = response.text
    except Exception as e:
        print(f"Warning: could not fetch free API list: {e}", file=sys.stderr)
        return {}

    providers = {}
    current_section = None
    current_link = None
    # Pattern pour extraire [texte](url) depuis un titre de section ou une ligne **Link**
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

    for line in content.splitlines():
        line = line.strip()

        # Détection d'une section de fournisseur (ex: "### [Google Gemini](https://aistudio.google.com/app/apikey)")
        if line.startswith("### "):
            raw_section = line[4:].strip()
            match = link_pattern.search(raw_section)
            if match:
                current_section = clean_provider_name(match.group(1))
                current_link = match.group(2)        # URL de la page d'obtention de clé
            else:
                current_section = clean_provider_name(raw_section)
                current_link = None
            continue

        # Détection des tableaux de modèles
        if line.startswith('|') and line.endswith('|'):
            if re.match(r'^\|[\s\-:]+\|', line):
                continue
            if '**Model**' in line or 'Model' == line.split('|')[1].strip():
                continue

            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:
                model_name = cells[0]
                model_name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', model_name)
                if model_name and model_name not in ('', 'Model', '**Model**'):
                    normalized = normalize_name(model_name)
                    if normalized:
                        if normalized not in providers:
                            providers[normalized] = []
                        url = current_link if current_link else ""
                        provider_data = {
                            "provider": current_section or "Unknown",
                            "url": url
                        }
                        if provider_data not in providers[normalized]:
                            providers[normalized].append(provider_data)

    print(f"Found {len(providers)} unique models with free APIs from {FREE_API_README_URL}")
    return providers


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

    # Récupération de la liste des API gratuites
    free_api_data = fetch_free_api_data()

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
            found_in_config += 1
        else:
            model_name = arena_name
            vendor = deduce_vendor(arena_name, general_entry["organization"])
            type_val = deduce_type(general_entry["license"])
            url = "https://lmarena.ai"
            auto_added += 1
            auto_added_list.append(arena_name)

        # Déterminer has_free_api et les fournisseurs
        has_free = False
        providers = []
        names_to_test = [model_name, arena_name]
        if config_entry and "free_api_names" in config_entry:
            names_to_test.extend(config_entry["free_api_names"])

        for name in names_to_test:
            normalized = normalize_name(name)
            if normalized in free_api_data:
                has_free = True
                for p in free_api_data[normalized]:
                    if p not in providers:
                        providers.append(p)

        if config_entry and "has_free_api" in config_entry:
            has_free = config_entry["has_free_api"]

        result.append({
            "name": model_name,
            "vendor": vendor,
            "type": type_val,
            "url": url,
            "score_general": general_entry["rating"],
            "score_code": code_entry["rating"] if code_entry else None,
            "has_free_api": has_free,
            "free_api_providers": providers
        })

    # Tri par score_general décroissant
    result.sort(
        key=lambda x: x["score_general"] if x["score_general"] is not None else float("-inf"),
        reverse=True
    )

    output = {
        "ranking_date": str(ranking_date) if ranking_date else now[:10],
        "retrieved_at": now,
        "models": result
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Ranking date        : {output['ranking_date']}")
    print(f"Retrieved at        : {output['retrieved_at']}")
    print(f"Total models        : {len(result)}")
    print(f"Models with free API: {sum(1 for m in result if m['has_free_api'])}")
    print(f"Found in config     : {found_in_config}")
    print(f"Auto-added          : {auto_added}")
    if auto_added_list:
        print("Auto-added models for review:")
        for m in sorted(auto_added_list):
            print(f"  - {m}")


if __name__ == "__main__":
    main()
