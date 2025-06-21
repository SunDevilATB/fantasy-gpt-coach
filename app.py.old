import os
import json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

app = Flask(__name__, static_folder="static", static_url_path="")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_ui(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.route("/challenge", methods=["POST"])
def challenge():
    data = request.get_json()
    advice_json = data.get("advice")
    format = data.get("scoring_format", "PPR")
    user_challenge = data.get("user_challenge", "").strip()

    if not advice_json:
        return jsonify({"error": "No advice provided to challenge."}), 400
    if not user_challenge:
        return jsonify({"error": "No challenge provided."}), 400

    prompt = f"""
You are a no-nonsense fantasy football coach who has just given a lineup recommendation.

League Format: {format}

Player Challenge:
"{user_challenge}"

Here’s the lineup advice you gave:
{json.dumps(advice_json, indent=2)}

Respond angrily and defend your picking logic—explain your baseline, reason through matchups, and address the specific challenge.

Return only the coach's verbal response (plain text, no JSON).
"""

    messages = [
        {"role": "system", "content": "You are a bold, opinionated fantasy coach ready to defend your choices."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.9
        )
        rebuttal = response.choices[0].message.content.strip()
        return jsonify({"rebuttal": rebuttal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        scoring_format = data.get("scoring_format", "PPR")
        notes = data.get("notes", "")
        roster = data.get("roster", {})

        prompt = f"""
You are a sharp, opinionated fantasy football coach. Your job is to select the optimal starting lineup
based on the user's current roster, league format, and strategic considerations.

League Format: {scoring_format}
Custom Notes: {notes or "None provided"}

Here is the user's full roster:
{json.dumps(roster, indent=2)}

Instructions:
- Identify the strongest possible starting lineup for this week.
- Clearly separate players into: recommended_starters, bench, and waiver_watchlist.
- Prioritize upside for FLEX positions and matchups for borderline players.
- Include a brief strategy_summary that explains your reasoning, especially any risky calls.

Respond ONLY in valid JSON like this:
{{
  "recommended_starters": {{
    "QB": ["..."],
    "RB": ["..."],
    "WR": ["..."],
    "TE": ["..."],
    "FLEX": ["..."]
  }},
  "bench": ["..."],
  "waiver_watchlist": ["..."],
  "strategy_summary": "..."
}}
"""

        messages = [
            {"role": "system", "content": "You are a fantasy football expert. Only return valid JSON with no extra explanation."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()
        advice = json.loads(raw)
        return jsonify(advice)

    except json.JSONDecodeError:
        return jsonify({"error": "GPT returned invalid JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
