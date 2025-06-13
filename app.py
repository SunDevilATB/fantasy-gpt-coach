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

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        scoring_format = data.get("scoring_format", "PPR")
        notes = data.get("notes", "")
        roster = data.get("roster", {})

        prompt = (
            f"You are a fantasy football assistant. Based on this scoring format: '{scoring_format}', "
            f"and these notes: '{notes}', give me lineup advice. "
            f"Here is the user's roster:\n{json.dumps(roster, indent=2)}\n\n"
            "Respond ONLY in valid JSON like this:\n"
            "{\n"
            "  \"recommended_starters\": { \"QB\": [\"...\"], \"RB\": [\"...\"], ... },\n"
            "  \"bench\": [\"...\"],\n"
            "  \"waiver_watchlist\": [\"...\"],\n"
            "  \"strategy_summary\": \"...\"\n"
            "}"
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful fantasy football expert. Respond only in valid JSON."},
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
