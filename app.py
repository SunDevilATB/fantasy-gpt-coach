from flask import render_template
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

import os
import json
import requests

# Load environment variables
load_dotenv()

# OpenAI client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Load player map from Sleeper API and cache locally
def load_player_map():
    path = "sleeper_players.json"
    if not os.path.exists(path):
        print("Downloading player map from Sleeper...")
        r = requests.get("https://api.sleeper.app/v1/players/nfl")
        if r.status_code == 200:
            with open(path, "w") as f:
                f.write(json.dumps(r.json()))
        else:
            raise Exception("Could not download player map")
    with open(path, "r") as f:
        return json.load(f)

player_map = load_player_map()

# Match player name to Sleeper player_id
def match_player_id(name, player_map):
    name = name.lower().strip()
    for player_id, data in player_map.items():
        full_name = data.get("full_name", "").lower()
        if name in full_name:
            return player_id
    return None

# Get player stats from Sleeper API
def get_player_stats(week=1):
    url = f"https://api.sleeper.app/v1/stats/nfl/2023/regular/{week}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

# Build the GPT prompt using structured JSON
def build_prompt(data, stats=None):
    roster = data.get("roster", {})
    scoring = data.get("scoring_format", "unknown format")
    notes = data.get("notes", "None")

    prompt = f"""
You are a fantasy football coach and analyst.

The user is playing in a {scoring} league.
Here is their roster:

{json.dumps(roster, indent=2)}

Additional notes:
{notes}
"""

    if stats:
        prompt += "\nHere are some recent stats for their players:\n"
        for position in roster:
            for name in roster[position]:
                player_id = match_player_id(name, player_map)
                player_stats = next((s for s in stats if s.get("player_id") == player_id), None)
                if player_stats:
                    prompt += f"\n{name} ({player_id}): {json.dumps(player_stats, indent=2)}"

    prompt += """
Please respond in **valid JSON format** with the following structure:

{
  "recommended_starters": {
    "QB": [],
    "RB": [],
    "WR": [],
    "TE": [],
    "FLEX": []
  },
  "bench": [],
  "waiver_watchlist": [],
  "strategy_summary": ""
}
"""
    return prompt

# POST endpoint for GPT recommendations
@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        stats = get_player_stats(week=1)
        prompt = build_prompt(data, stats)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()

        # Try to parse the JSON response from GPT
        try:
            if raw.startswith("```json"):
                raw = raw.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(raw)
            return jsonify(parsed)
        except Exception:
            return jsonify({"raw_output": raw, "error": "Failed to parse JSON"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        previous_messages = data.get("history", [])

        messages = [{"role": "system", "content": "You are a helpful fantasy football coach and analyst."}]
        messages += previous_messages  # previous_messages is a list of {role, content}
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=200  # limit the length of the response
        )

        reply = response.choices[0].message.content
        return jsonify({ "reply": reply })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port)

