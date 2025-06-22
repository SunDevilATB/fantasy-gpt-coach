import os
import json
import logging
import time
import hashlib
import threading
import requests
import difflib
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")

# =============================================================================
# CONFIGURATION & GLOBALS
# =============================================================================

# Caching configuration
cache = {}
CACHE_TTL = 3600  # 1 hour in seconds

# Rate limiting and cost controls
request_counts = defaultdict(list)  # IP -> [timestamps]
daily_api_calls = {'count': 0, 'date': datetime.now().date()}
monthly_budget = 100  # $100 monthly limit
api_call_cost = 0.03  # Rough cost per call
counter_lock = threading.Lock()

# Player data configuration
SLEEPER_API_BASE = "https://api.sleeper.app/v1"
PLAYER_DATA_TTL = 24 * 3600  # 24 hours
player_data_cache = {'data': None, 'timestamp': 0}

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# =============================================================================
# OPENAI CLIENT SETUP
# =============================================================================

def validate_env():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return api_key

try:
    client = OpenAI(api_key=validate_env())
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    client = None

# =============================================================================
# RETRY LOGIC & ERROR HANDLING
# =============================================================================

def retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator for retrying failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    
            # All retries failed
            logger.error(f"All {max_retries} attempts failed: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=2)
def call_openai_api(messages, model="gpt-4o", temperature=0.7, max_tokens=1000):
    """Robust OpenAI API call with retries and timeouts"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30  # 30 second timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

# =============================================================================
# RATE LIMITING & COST CONTROLS
# =============================================================================

def check_rate_limit(ip_address, max_requests=10, window_minutes=10):
    """Check if IP has exceeded rate limit"""
    with counter_lock:
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)
        
        # Clean old requests
        request_counts[ip_address] = [
            timestamp for timestamp in request_counts[ip_address] 
            if timestamp > cutoff
        ]
        
        # Check current count
        if len(request_counts[ip_address]) >= max_requests:
            return False
        
        # Add current request
        request_counts[ip_address].append(now)
        return True

def check_daily_budget():
    """Check if we've hit daily spending limits"""
    with counter_lock:
        today = datetime.now().date()
        
        # Reset counter if new day
        if daily_api_calls['date'] != today:
            daily_api_calls['count'] = 0
            daily_api_calls['date'] = today
        
        # Check limits
        daily_cost = daily_api_calls['count'] * api_call_cost
        daily_limit = monthly_budget / 30  # Rough daily limit
        
        if daily_cost >= daily_limit:
            logger.warning(f"Daily budget exceeded: ${daily_cost:.2f}")
            return False
            
        return True

def increment_api_usage():
    """Track API usage for cost monitoring"""
    with counter_lock:
        daily_api_calls['count'] += 1
        logger.info(f"API calls today: {daily_api_calls['count']}, "
                   f"estimated cost: ${daily_api_calls['count'] * api_call_cost:.2f}")

# =============================================================================
# PLAYER DATA MANAGEMENT
# =============================================================================

def fetch_fresh_player_data():
    """Fetch current player data from Sleeper API"""
    try:
        # Get all NFL players from Sleeper
        response = requests.get(f"{SLEEPER_API_BASE}/players/nfl", timeout=10)
        response.raise_for_status()
        
        players_data = response.json()
        
        # Transform to our format
        fresh_database = {
            'QB': [],
            'RB': [],
            'WR': [],
            'TE': []
        }
        
        fresh_teams = {}
        
        for player_id, player_info in players_data.items():
            if not player_info.get('active', False):
                continue
                
            name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
            position = player_info.get('position', '')
            team = player_info.get('team', '')
            
            if position in fresh_database and name:
                fresh_database[position].append(name)
                fresh_teams[name] = team
        
        # Sort players by name
        for position in fresh_database:
            fresh_database[position].sort()
        
        logger.info(f"Fetched fresh player data: {sum(len(players) for players in fresh_database.values())} players")
        return fresh_database, fresh_teams
        
    except Exception as e:
        logger.error(f"Failed to fetch fresh player data: {e}")
        return None, None

def get_current_player_data():
    """Get current player data with caching"""
    current_time = time.time()
    
    # Check if we have fresh data
    if (player_data_cache['data'] and 
        current_time - player_data_cache['timestamp'] < PLAYER_DATA_TTL):
        return player_data_cache['data']
    
    # Try to fetch fresh data
    fresh_db, fresh_teams = fetch_fresh_player_data()
    
    if fresh_db and fresh_teams:
        # Update cache
        player_data_cache['data'] = (fresh_db, fresh_teams)
        player_data_cache['timestamp'] = current_time
        
        logger.info("Updated player database with fresh data")
        return fresh_db, fresh_teams
    else:
        # Fall back to static data (your original database)
        logger.warning("Using static player data - API fetch failed")
        return get_fallback_player_data()

def get_fallback_player_data():
    """Fallback to your original static player database"""
    # Your original player database as fallback
    playerDatabase = {
        'QB': [
            'Josh Allen', 'Lamar Jackson', 'Patrick Mahomes', 'Joe Burrow', 'Jalen Hurts',
            'Justin Herbert', 'Dak Prescott', 'Tua Tagovailoa', 'Kyler Murray', 'Russell Wilson',
            'Aaron Rodgers', 'Geno Smith', 'Kirk Cousins', 'Trevor Lawrence', 'Anthony Richardson',
            'Brock Purdy', 'Jared Goff', 'Daniel Jones', 'Derek Carr', 'Justin Fields',
            'C.J. Stroud', 'Kenny Pickett', 'Deshaun Watson', 'Ryan Tannehill', 'Mac Jones',
            'Bryce Young', 'Will Levis', 'Sam Howell', 'Baker Mayfield', 'Jordan Love',
            'Aidan O\'Connell', 'Tommy DeVito', 'Mason Rudolph', 'Gardner Minshew', 'Jacoby Brissett',
            'Jayden Daniels', 'Caleb Williams', 'Drake Maye', 'Bo Nix', 'J.J. McCarthy', 'Michael Penix Jr.',
            'Cam Ward', 'Dillon Gabriel', 'Shedeur Sanders', 'Jalen Milroe', 'Quinn Ewers'
        ],
        'RB': [
            'Christian McCaffrey', 'Austin Ekeler', 'Derrick Henry', 'Josh Jacobs', 'Nick Chubb',
            'Saquon Barkley', 'Tony Pollard', 'Aaron Jones', 'Joe Mixon', 'Kenneth Walker III',
            'Najee Harris', 'Alvin Kamara', 'James Conner', 'Travis Etienne', 'D\'Andre Swift',
            'Javonte Williams', 'Rachaad White', 'Jerome Ford', 'Isiah Pacheco', 'Breece Hall',
            'Jonathan Taylor', 'James Cook', 'Brian Robinson Jr.', 'Gus Edwards', 'Chuba Hubbard',
            'Roschon Johnson', 'Tank Bigsby', 'Jaylen Warren', 'Justice Hill', 'Rico Dowdle',
            'Kenneth Gainwell', 'Devin Singletary', 'Jordan Mason', 'Ty Chandler', 'AJ Dillon',
            'Zamir White', 'Tyler Allgeier', 'Khalil Herbert', 'Zack Moss', 'Cam Akers',
            'Ashton Jeanty', 'Omarion Hampton', 'Blake Corum', 'Trey Benson', 'Braelon Allen'
        ],
        'WR': [
            'Tyreek Hill', 'Stefon Diggs', 'Davante Adams', 'Cooper Kupp', 'DeAndre Hopkins',
            'A.J. Brown', 'Mike Evans', 'Keenan Allen', 'Amari Cooper', 'Tyler Lockett',
            'DK Metcalf', 'CeeDee Lamb', 'Ja\'Marr Chase', 'Justin Jefferson', 'Amon-Ra St. Brown',
            'Puka Nacua', 'Chris Olave', 'Garrett Wilson', 'Jaylen Waddle', 'Terry McLaurin',
            'Calvin Ridley', 'DJ Moore', 'Michael Pittman Jr.', 'Courtland Sutton', 'Tee Higgins',
            'Jerry Jeudy', 'Drake London', 'Chris Godwin', 'Deebo Samuel', 'DeVonta Smith',
            'Zay Flowers', 'George Pickens', 'Tank Dell', 'Nico Collins', 'Christian Watson',
            'Romeo Doubs', 'Jordan Addison', 'Rashee Rice', 'Josh Palmer', 'Marquise Goodwin',
            'Malik Nabers', 'Rome Odunze', 'Marvin Harrison Jr.', 'Brian Thomas Jr.', 'Xavier Worthy',
            'Ladd McConkey', 'Keon Coleman', 'Xavier Legette', 'Ricky Pearsall', 'Ja\'Lynn Polk',
            'Travis Hunter', 'Emeka Egbuka', 'Matthew Golden'
        ],
        'TE': [
            'Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'George Kittle', 'Kyle Pitts',
            'Dallas Goedert', 'Evan Engram', 'Sam LaPorta', 'David Njoku', 'Pat Freiermuth',
            'Jake Ferguson', 'Dawson Knox', 'Cole Kmet', 'Trey McBride', 'Tyler Higbee',
            'Gerald Everett', 'Noah Fant', 'Hunter Henry', 'Mike Gesicki', 'Isaiah Likely',
            'Luke Musgrave', 'Tucker Kraft', 'Will Dissly', 'Tyler Conklin', 'Cade Otton',
            'Brock Bowers', 'Dalton Kincaid', 'Michael Mayer', 'Tyler Warren', 'Colston Loveland'
        ]
    }
    
    playerTeams = {
        'Josh Allen': 'BUF', 'Lamar Jackson': 'BAL', 'Patrick Mahomes': 'KC', 'Joe Burrow': 'CIN',
        'Jalen Hurts': 'PHI', 'Justin Herbert': 'LAC', 'Dak Prescott': 'DAL', 'Tua Tagovailoa': 'MIA',
        'Jayden Daniels': 'WSH', 'Caleb Williams': 'CHI', 'Drake Maye': 'NE', 'Bo Nix': 'DEN',
        'Christian McCaffrey': 'SF', 'Saquon Barkley': 'PHI', 'Josh Jacobs': 'GB', 'Derrick Henry': 'BAL',
        'Austin Ekeler': 'WSH', 'Tony Pollard': 'TEN', 'Aaron Jones': 'MIN', 'Joe Mixon': 'HOU',
        'Tyreek Hill': 'MIA', 'Stefon Diggs': 'HOU', 'CeeDee Lamb': 'DAL', 'Ja\'Marr Chase': 'CIN',
        'Justin Jefferson': 'MIN', 'A.J. Brown': 'PHI', 'Mike Evans': 'TB', 'Davante Adams': 'LV',
        'Travis Kelce': 'KC', 'Mark Andrews': 'BAL', 'George Kittle': 'SF', 'Sam LaPorta': 'DET',
        'T.J. Hockenson': 'MIN', 'Kyle Pitts': 'ATL', 'Dallas Goedert': 'PHI', 'Brock Bowers': 'LV'
    }
    
    return playerDatabase, playerTeams

# =============================================================================
# PLAYER NAME VALIDATION & AUTO-CORRECTION
# =============================================================================

def find_closest_player_match(input_name, position=None):
    """Find the closest matching player name with fuzzy matching"""
    try:
        player_db, _ = get_current_player_data()
        
        # Get all players or just from specific position
        if position and position in player_db:
            candidates = player_db[position]
        else:
            candidates = []
            for pos_players in player_db.values():
                candidates.extend(pos_players)
        
        if not candidates:
            return None, 0
        
        # Find best match using difflib
        matches = difflib.get_close_matches(
            input_name, 
            candidates, 
            n=1, 
            cutoff=0.6  # 60% similarity threshold
        )
        
        if matches:
            best_match = matches[0]
            similarity = difflib.SequenceMatcher(None, input_name.lower(), best_match.lower()).ratio()
            return best_match, similarity
        
        return None, 0
        
    except Exception as e:
        logger.error(f"Player matching error: {e}")
        return None, 0

def validate_and_correct_roster(roster):
    """Validate player names and suggest corrections"""
    corrections = {}
    validated_roster = {}
    
    for position, players in roster.items():
        validated_roster[position] = []
        
        for player_name in players:
            if not player_name or not player_name.strip():
                continue
                
            # Try exact match first
            player_db, _ = get_current_player_data()
            exact_match = False
            
            for pos_players in player_db.values():
                if player_name in pos_players:
                    validated_roster[position].append(player_name)
                    exact_match = True
                    break
            
            if not exact_match:
                # Try fuzzy matching
                suggested_name, similarity = find_closest_player_match(player_name, position)
                
                if suggested_name and similarity > 0.7:  # 70% confidence
                    corrections[player_name] = {
                        'suggested': suggested_name,
                        'similarity': similarity,
                        'position': position
                    }
                    validated_roster[position].append(suggested_name)
                    logger.info(f"Auto-corrected '{player_name}' → '{suggested_name}' (similarity: {similarity:.2f})")
                else:
                    # Keep original if no good match found
                    validated_roster[position].append(player_name)
    
    return validated_roster, corrections

# =============================================================================
# CACHING SYSTEM
# =============================================================================

def create_request_hash(roster, scoring_format, notes):
    """Create a hash of the request parameters for caching"""
    cache_key = {
        'roster': {k: sorted(v) for k, v in roster.items()},  # Sort player lists
        'format': scoring_format,
        'notes': notes.strip().lower() if notes else ""
    }
    cache_string = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

def get_cached_recommendation(roster, scoring_format, notes=""):
    """Get recommendation with caching and error handling"""
    request_hash = create_request_hash(roster, scoring_format, notes)
    
    # Check cache first
    if request_hash in cache:
        cached_result, timestamp = cache[request_hash]
        if time.time() - timestamp < CACHE_TTL:
            logger.info(f"Cache HIT: {request_hash[:8]}...")
            return cached_result
        else:
            del cache[request_hash]
    
    try:
        # Build prompt
        current_week = "Week 1"
        prompt = f"""
You are a sharp, opinionated fantasy football coach analyzing lineups for {current_week} of the 2025 NFL season.

League Format: {scoring_format}
Custom Notes: {notes or "None provided"}

Here is the user's full roster:
{json.dumps(roster, indent=2)}

Instructions:
- Select the optimal starting lineup considering matchups, injury status, and recent performance
- Clearly separate players into: recommended_starters, bench, and waiver_watchlist
- For FLEX positions, prioritize high-upside players with favorable matchups
- Include players to watch on waivers based on potential opportunity
- Provide a brief strategy_summary explaining your key decisions and any risky calls

Respond ONLY in valid JSON format:
{{
  "recommended_starters": {{
    "QB": ["..."],
    "RB": ["...", "..."],
    "WR": ["...", "..."],
    "TE": ["..."],
    "FLEX": ["..."]
  }},
  "bench": ["...", "..."],
  "waiver_watchlist": ["...", "..."],
  "strategy_summary": "Brief explanation of key decisions and reasoning"
}}
"""

        messages = [
            {"role": "system", "content": "You are a fantasy football expert. Return only valid JSON with lineup recommendations."},
            {"role": "user", "content": prompt}
        ]

        # Call OpenAI with retries
        raw_response = call_openai_api(messages)
        
        # Clean and parse response
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from OpenAI: {raw_response}")
            raise ValueError("AI returned invalid response format")
        
        # Validate response structure
        required_keys = ['recommended_starters', 'bench', 'waiver_watchlist', 'strategy_summary']
        if not all(key in result for key in required_keys):
            raise ValueError("AI response missing required fields")
        
        # Cache successful result
        cache[request_hash] = (result, time.time())
        return result
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        # Return a fallback response instead of crashing
        return {
            "recommended_starters": {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": []},
            "bench": [],
            "waiver_watchlist": [],
            "strategy_summary": "Unable to generate recommendations at this time. Please try again in a few moments.",
            "error": True
        }

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

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

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_ui(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    if not client:
        return jsonify({"error": "OpenAI client not configured"}), 500
    
    # Get client IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    
    # Check rate limiting
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({
            "error": "Too many requests. Please wait a few minutes before trying again."
        }), 429
    
    # Check daily budget
    if not check_daily_budget():
        return jsonify({
            "error": "Service temporarily unavailable. Please try again tomorrow."
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        scoring_format = get_scoring_format(data)
        notes = data.get("notes", "")
        roster = data.get("roster", {})

        # Validate roster structure
        try:
            validate_roster(roster)
        except ValueError as e:
            return jsonify({"error": f"Invalid roster: {str(e)}"}), 400

        # NEW: Validate and auto-correct player names
        validated_roster, corrections = validate_and_correct_roster(roster)
        
        # If we made corrections, log them
        if corrections:
            logger.info(f"Auto-corrected player names: {corrections}")

        # Increment usage counter before API call
        increment_api_usage()
        
        # Get recommendation using validated roster
        advice = get_cached_recommendation(validated_roster, scoring_format, notes)
        
        # Add corrections to response if any were made
        if corrections:
            advice['player_corrections'] = corrections
            advice['message'] = f"Auto-corrected {len(corrections)} player name(s)"
        
        # Log successful request
        logger.info(f"Successful recommendation for IP: {client_ip}")
        return jsonify(advice)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Recommend endpoint error: {e}")
        return jsonify({"error": "Failed to generate lineup recommendations"}), 500

@app.route("/challenge", methods=["POST"])
def challenge():
    if not client:
        return jsonify({"error": "OpenAI client not configured"}), 500
    
    # Get client IP and check rate limits
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if not check_rate_limit(client_ip, max_requests=5, window_minutes=10):  # Lower limit for challenges
        return jsonify({"error": "Too many challenge requests. Please wait before trying again."}), 429
    
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

        # Track API usage for challenges too
        increment_api_usage()
        
        # Use the robust API call
        rebuttal = call_openai_api(messages, temperature=0.9, max_tokens=500)
        
        return jsonify({"rebuttal": rebuttal})
        
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Challenge endpoint error: {e}")
        return jsonify({"error": "Failed to generate coach response"}), 500

# =============================================================================
# MONITORING ENDPOINTS
# =============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    status = {
        "status": "healthy",
        "openai_configured": client is not None,
        "cache_entries": len(cache),
        "player_data_fresh": time.time() - player_data_cache.get('timestamp', 0) < PLAYER_DATA_TTL
    }
    return jsonify(status)

@app.route("/cache-stats", methods=["GET"])
def cache_stats():
    """Debug endpoint to see cache performance"""
    if not cache:
        return jsonify({"message": "Cache is empty"})
    
    current_time = time.time()
    active_entries = sum(1 for _, (_, timestamp) in cache.items() 
                        if current_time - timestamp < CACHE_TTL)
    
    return jsonify({
        "total_entries": len(cache),
        "active_entries": active_entries,
        "expired_entries": len(cache) - active_entries,
        "cache_ttl_hours": CACHE_TTL / 3600
    })

@app.route("/usage-stats", methods=["GET"])
def usage_stats():
    """Monitor API usage and costs"""
    with counter_lock:
        today_cost = daily_api_calls['count'] * api_call_cost
        monthly_estimate = today_cost * 30
        
        return jsonify({
            "daily_api_calls": daily_api_calls['count'],
            "estimated_daily_cost": f"${today_cost:.2f}",
            "estimated_monthly_cost": f"${monthly_estimate:.2f}",
            "budget_status": "OK" if monthly_estimate < monthly_budget else "WARNING",
            "active_ips": len(request_counts)
        })

@app.route("/refresh-players", methods=["POST"])
def refresh_players():
    """Manually refresh player data"""
    try:
        fresh_db, fresh_teams = fetch_fresh_player_data()
        if fresh_db:
            # Force update cache
            player_data_cache['data'] = (fresh_db, fresh_teams)
            player_data_cache['timestamp'] = time.time()
            
            return jsonify({
                "status": "success",
                "message": f"Updated {sum(len(players) for players in fresh_db.values())} players",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to fetch fresh data"}), 500
    except Exception as e:
        logger.error(f"Manual refresh failed: {e}")
        return jsonify({"error": str(e)}), 500

# =============================================================================
# APP INITIALIZATION
# =============================================================================

def initialize_app():
    """Initialize app with fresh player data"""
    logger.info("Initializing app with fresh player data...")
    try:
        get_current_player_data()
        logger.info("App initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize player data: {e}")
        logger.info("App will use fallback static data")

if __name__ == "__main__":
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)