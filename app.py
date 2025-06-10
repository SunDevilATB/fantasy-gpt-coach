import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# Load your OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()

        # Build the prompt
        format = data.get("scoring_format", "PPR")
        notes = data.get("notes", "")
        roster = data.get("roster", {})

        prompt = (
            f"You are a fantasy football assistant. Based on this scoring format: '{format}', "
            f"and these notes: '{notes}', give me lineup advice. "
            f"Here is the user's roster:\n{json.dumps(roster, indent=2)}\n\n"
            "Respond in valid JSON like this:\n"
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
                {"role": "system", "content": "You are a helpful fantasy football assistant. Reply in JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Parse the JSON string returned by GPT
        raw_text = response.choices[0].message.content.strip()

        try:
            advice = json.loads(raw_text)
        except json.JSONDecodeError:
            print("❌ GPT returned invalid JSON:")
            print(raw_text)
            return jsonify({"error": "Invalid JSON returned from GPT"}), 500

        return jsonify(advice)

    except Exception as e:
        print("❌ Server error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

