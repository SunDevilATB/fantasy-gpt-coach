import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")

# Validate environment variables on startup
def validate_env():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return api_key

try:
    client = OpenAI(api_key=validate_env())
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    # In production, you might want to exit here
    client = None

def validate_roster(roster: Dict[str, Any]) -> bool:
    """Validate roster structure and content."""
    if not isinstance(roster, dict):
        raise ValueError("Roster must be a dictionary")
    
    required_positions = ['QB', 'RB', 'WR', 'TE']
    missing_positions = [pos for pos in required_positions if pos not in roster]
    
    if missing_positions:
        raise ValueError(f"Missing required positions: {', '.join(missing_positions)}")
    
    # Validate each position has valid player names
    for position, players in roster.items():
        if not isinstance(players, list):
            raise ValueError(f"Position {position} must contain a list of players")
        
        for player in players:
            if not isinstance(player, str) or not player.strip():
                raise ValueError(f"Invalid player name in {position}: {player}")
    
    return True

def get_scoring_format(data: Dict[str, Any]) -> str:
    """Extract and format the scoring format."""
    format_type = data.get("scoring_format", "PPR")
    custom_format = data.get("custom_format", "")
    
    if format_type == "Custom" and custom_format:
        return f"Custom: {custom_format}"
    return format_type

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_ui(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/challenge", methods=["POST"])
def challenge():
    if not client:
        return jsonify({"error": "OpenAI client not configured"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        advice_json = data.get("advice")
        user_challenge = data.get("user_challenge", "").strip()
        scoring_format = get_scoring_format(data)

        if not advice_json:
            return jsonify({"error": "No advice provided to challenge"}), 400
        if not user_challenge:
            return jsonify({"error": "No challenge provided"}), 400

        prompt = f"""
You are a no-nonsense fantasy football coach who has just given a lineup recommendation.

League Format: {scoring_format}

Player Challenge:
"{user_challenge}"

Here's the lineup advice you gave:
{json.dumps(advice_json, indent=2)}

Respond angrily and defend your picking logic—explain your baseline, reason through matchups, and address the specific challenge.

Return only the coach's verbal response (plain text, no JSON).
"""

        messages = [
            {"role": "system", "content": "You are a bold, opinionated fantasy coach ready to defend your choices."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to latest model
            messages=messages,
            temperature=0.9,
            max_tokens=500  # Add token limit
        )
        
        rebuttal = response.choices[0].message.content.strip()
        return jsonify({"rebuttal": rebuttal})
        
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Challenge endpoint error: {e}")
        return jsonify({"error": "Failed to generate coach response"}), 500

@app.route("/recommend", methods=["POST"])
def recommend():
    if not client:
        return jsonify({"error": "OpenAI client not configured"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        scoring_format = get_scoring_format(data)
        notes = data.get("notes", "")
        roster = data.get("roster", {})

        # Validate roster
        try:
            validate_roster(roster)
        except ValueError as e:
            return jsonify({"error": f"Invalid roster: {str(e)}"}), 400

        # Enhanced prompt with better context
        current_week = "Week 1"  # You could make this dynamic
        prompt = f"""
You are a sharp, opinionated fantasy football coach analyzing lineups for {current_week} of the 2024 NFL season.

League Format: {scoring_format}
Custom Notes: {notes or "None provided"}

Here is the user's full roster:
{json.dumps(roster, indent=2)}

Instructions:
- Select the optimal starting lineup considering matchups, injury status, and recent performance
- Clearly separate players into: recommended_starters, bench, and waiver_watchlist
- For FLEX positions, prioritize high-upside players with favorable matchups
- Include team defenses (D/ST) and kickers (K) in your recommendations
- Include 4-6 players to watch on waivers based on potential opportunity, injuries, or breakout potential
- Provide a brief strategy_summary explaining your key decisions and any risky calls

Respond ONLY in valid JSON format:
{{
  "recommended_starters": {{
    "QB": ["..."],
    "RB": ["...", "..."],
    "WR": ["...", "..."],
    "TE": ["..."],
    "FLEX": ["..."],
    "DST": ["..."],
    "K": ["..."]
  }},
  "bench": ["...", "..."],
  "waiver_watchlist": ["Player1", "Player2", "Player3", "Player4", "Player5"],
  "strategy_summary": "Brief explanation of key decisions and reasoning"
}}
"""

        messages = [
            {"role": "system", "content": "You are a fantasy football expert. Return only valid JSON with lineup recommendations."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to latest model
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        raw_response = response.choices[0].message.content.strip()
        
        # Clean up response (remove any markdown formatting)
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        
        try:
            advice = json.loads(raw_response)
            
            # Ensure waiver_watchlist always exists
            if 'waiver_watchlist' not in advice or not advice['waiver_watchlist']:
                # Fallback waiver recommendations
                fallback_waivers = [
                    'Tank Dell', 'Tyler Boyd', 'Roschon Johnson', 'Ty Chandler',
                    'Demarcus Robinson', 'Noah Brown', 'Jordan Mason', 'Rico Dowdle',
                    'Isaiah Likely', 'Tyler Conklin', 'Deon Jackson', 'Zay Jones'
                ]
                advice['waiver_watchlist'] = fallback_waivers[:5]  # Take first 5
            
            return jsonify(advice)
        except json.JSONDecodeError as e:
            logger.error(f"GPT returned invalid JSON: {raw_response}")
            return jsonify({"error": "AI returned invalid response format"}), 500

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Recommend endpoint error: {e}")
        return jsonify({"error": "Failed to generate lineup recommendations"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    status = {
        "status": "healthy",
        "openai_configured": client is not None
    }
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)