import json
import sys
import re
import os
import hashlib
from datetime import datetime, timezone
from datasets import load_dataset
import requests
from google import genai
from google.genai import types

# ---- Configuration ----
OUTPUT_PATH = "data.json"
CACHE_PATH = "gemini-cache.json"
FREE_API_README_URL = "https://raw.githubusercontent.com/mnfst/awesome-free-llm-apis/main/README.md"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"


def load_hf_leaderboard(subset: str):
    """Charge un subset HuggingFace et retourne (data dict, ranking_date)."""
    try:
        ds = load_dataset(
            "lmarena-ai/leaderboard-dataset",
            subset,
            split="latest",
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
        return result, str(ranking_date) if ranking_date else None
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
    if not license_str or license_str.lower() == "proprietary":
        return "paid"
    return "free"


def normalize_name(name: str) -> str:
    name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)
    name = re.sub(r'[^a-z0-9\s]', '', name.lower())
    name = re.sub(r'\s+', '', name)
    return name


def clean_provider_name(raw: str) -> str:
    raw = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw)
    raw = raw.replace('[', '').replace(']', '')
    raw = re.sub(r'\([^)]*\)', '', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def fetch_free_api_data():
    print(f"Fetching free API data from {FREE_API_README_URL}...")
    try:
        resp = requests.get(FREE_API_README_URL, timeout=30)
        resp.raise_for_status()
        content = resp.text
    except Exception as e:
        print(f"Warning: could not fetch free API list: {e}", file=sys.stderr)
        return {}

    providers = {}
    current_section = None
    current_link = None
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("### "):
            raw = line[4:].strip()
            m = link_pattern.search(raw)
            if m:
                current_section = clean_provider_name(m.group(1))
                current_link = m.group(2)
            else:
                current_section = clean_provider_name(raw)
                current_link = None
            continue
        if line.startswith('|') and line.endswith('|'):
            if re.match(r'^\|[\s\-:]+\|', line): continue
            if '**Model**' in line or 'Model' == line.split('|')[1].strip(): continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells:
                model_name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cells[0])
                if model_name and model_name not in ('', 'Model'):
                    n = normalize_name(model_name)
                    if n:
                        if n not in providers: providers[n] = []
                        url = current_link if current_link else ""
                        entry = {"provider": current_section or "Unknown", "url": url}
                        if entry not in providers[n]: providers[n].append(entry)
    print(f"Found {len(providers)} unique models with free APIs.")
    return providers


def load_gemini_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_gemini_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def call_gemini_for_new_models(new_models):
    if not GEMINI_API_KEY:
        print("No GEMINI_API_KEY, using fallback heuristics.", file=sys.stderr)
        return {}

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        prompt = f"""
Tu es un assistant spécialisé dans les modèles d'IA. Pour chaque modèle ci-dessous, tu dois :
1. Normaliser le nom du fournisseur (vendor) : utilise le nom officiel complet (ex. "OpenAI", "Anthropic", "Google", "Meta", "Mistral", "Alibaba", "DeepSeek", "xAI", "Z.ai", "Baidu", "Moonshot", etc.).
2. Déterminer le type d'accès : "free" si le modèle est open source ou accessible gratuitement, "paid" s'il nécessite un abonnement ou un paiement.
3. Trouver l'URL officielle du modèle (page d'accueil du produit). Utilise la recherche web si nécessaire. Si tu ne trouves pas, utilise "https://lmarena.ai".

Retourne UNIQUEMENT un tableau JSON valide (sans texte autour). Chaque élément doit avoir exactement ces champs :
{{
  "arena_name": "...",
  "name": "...",
  "vendor": "...",
  "type": "free ou paid",
  "url": "..."
}}

Modèles à traiter :
{json.dumps(new_models, indent=2)}
"""
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        corrected_list = json.loads(text)
        result = {item["arena_name"]: item for item in corrected_list}
        print(f"Gemini corrected {len(result)} new models.")
        return result
    except Exception as e:
        print(f"Gemini error: {e}. Falling back to heuristics.", file=sys.stderr)
        return {}


def main():
    cache = load_gemini_cache()
    print(f"Loaded {len(cache)} models from Gemini cache.")

    print("Loading General leaderboard (text/overall)...")
    general_data, arena_date = load_hf_leaderboard("text")
    print(f"Fetched {len(general_data)} models, date={arena_date}")

    print("Loading Code leaderboard (webdev/overall)...")
    code_data, _ = load_hf_leaderboard("webdev")
    print(f"Fetched {len(code_data)} models from Code leaderboard.")

    # Identifier les nouveaux modèles (absents du cache)
    new_models = []
    for arena_name, entry in general_data.items():
        if arena_name not in cache:
            new_models.append({
                "arena_name": arena_name,
                "organization": entry["organization"],
                "license": entry["license"]
            })

    if new_models:
        print(f"Found {len(new_models)} new models. Calling Gemini...")
        corrections = call_gemini_for_new_models(new_models)
        for entry in new_models:
            arena_name = entry["arena_name"]
            if arena_name in corrections:
                cache[arena_name] = corrections[arena_name]
            else:
                e = general_data[arena_name]
                cache[arena_name] = {
                    "arena_name": arena_name,
                    "name": arena_name,
                    "vendor": deduce_vendor(arena_name, e["organization"]),
                    "type": deduce_type(e["license"]),
                    "url": "https://lmarena.ai"
                }
        save_gemini_cache(cache)
    else:
        print("No new models, skipping Gemini call.")

    free_api_data = fetch_free_api_data()

    result = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for arena_name, general_entry in general_data.items():
        meta = cache.get(arena_name)
        if not meta:
            meta = {
                "name": arena_name,
                "vendor": deduce_vendor(arena_name, general_entry["organization"]),
                "type": deduce_type(general_entry["license"]),
                "url": "https://lmarena.ai"
            }

        has_free = False
        providers = []
        for test_name in (meta["name"], arena_name):
            n = normalize_name(test_name)
            if n in free_api_data:
                has_free = True
                for p in free_api_data[n]:
                    if p not in providers:
                        providers.append(p)

        code_entry = code_data.get(arena_name)
        result.append({
            "name": meta["name"],
            "vendor": meta["vendor"],
            "type": meta["type"],
            "url": meta["url"],
            "score_general": general_entry["rating"],
            "score_code": code_entry["rating"] if code_entry else None,
            "has_free_api": has_free,
            "free_api_providers": providers
        })

    result.sort(key=lambda x: x["score_general"] if x["score_general"] is not None else float("-inf"),
                reverse=True)

    output = {
        "ranking_date": arena_date or now[:10],
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


if __name__ == "__main__":
    main()
