import hashlib
import json
import time
from functools import wraps
from datetime import datetime, timedelta

# Add this near the top of app.py after imports
cache = {}
CACHE_TTL = 3600  # 1 hour in seconds

def create_request_hash(roster, scoring_format, notes):
    """Create a hash of the request parameters for caching"""
    # Normalize the request for consistent hashing
    cache_key = {
        'roster': {k: sorted(v) for k, v in roster.items()},  # Sort player lists
        'format': scoring_format,
        'notes': notes.strip().lower() if notes else ""
    }
    cache_string = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

def with_cache(ttl=CACHE_TTL):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
            
            # Check if we have a cached result
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.info(f"Cache HIT for {func.__name__}")
                    return cached_result
                else:
                    # Cache expired, remove it
                    del cache[cache_key]
                    logger.info(f"Cache EXPIRED for {func.__name__}")
            
            # No cache or expired - call the function
            logger.info(f"Cache MISS for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache with timestamp
            cache[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

def get_cached_recommendation(roster, scoring_format, notes=""):
    """Get recommendation with caching"""
    request_hash = create_request_hash(roster, scoring_format, notes)
    
    # Check cache first
    if request_hash in cache:
        cached_result, timestamp = cache[request_hash]
        if time.time() - timestamp < CACHE_TTL:
            logger.info(f"Cache HIT: {request_hash[:8]}... (saved OpenAI call)")
            return cached_result
        else:
            # Expired
            del cache[request_hash]
            logger.info(f"Cache EXPIRED: {request_hash[:8]}...")
    
    # Cache miss - make OpenAI call
    logger.info(f"Cache MISS: {request_hash[:8]}... (calling OpenAI)")
    
    try:
        # Your existing OpenAI call logic here
        prompt = f"""
You are a sharp, opinionated fantasy football coach analyzing lineups for Week 1 of the 2024 NFL season.

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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        raw_response = response.choices[0].message.content.strip()
        
        # Clean up response
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        
        result = json.loads(raw_response)
        
        # Cache the successful result
        cache[request_hash] = (result, time.time())
        logger.info(f"Cached result: {request_hash[:8]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        raise

# Update your /recommend endpoint
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

        # Use cached recommendation
        advice = get_cached_recommendation(roster, scoring_format, notes)
        return jsonify(advice)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Recommend endpoint error: {e}")
        return jsonify({"error": "Failed to generate lineup recommendations"}), 500

# Optional: Add cache stats endpoint for monitoring
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