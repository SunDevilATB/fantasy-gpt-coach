import os
import json
from flask import Flask, request, jsonify, send_from_directory
import openai

app = Flask(__name__, static_folder="static", static_url_path="")
openai.api_key = os.getenv("OPENAI_API_KEY")

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

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fantasy football expert. Respond in JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()
        advice = json.loads(raw)
        return jsonify(advice)

    except json.JSONDecodeError:
        print("⚠️ GPT returned invalid JSON.")
        return jsonify({"error": "Invalid JSON returned from GPT"}), 500
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)