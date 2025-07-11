from flask import Flask, request, jsonify
from waitress import serve
import requests, yaml, json, joblib, redis
from datetime import datetime
from google import genai
from training.training import predict_next_api_call, ScoreRescaler, get_embedding, cosine_similarity
from sentence_transformers import SentenceTransformer
from jsonschema import validate, ValidationError
from google.genai import types
import asyncio
from pathlib import Path

endpoints = None
suggestions = []
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = genai.Client(api_key="AIzaSyBasLrT6Rl9O7TxD-1rB_mUN4TvEHhCCeM")
loaded_lgbm_model = joblib.load('lgbm_model.pkl')
loaded_one_hot_encoder = joblib.load('one_hot_encoder.pkl')
loaded_score_rescaler = joblib.load('score_rescaler.pkl')
global_feature_columns = joblib.load('calls_list.joblib')
current_version = 0
github = "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json"
stripe = "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json"

def load_openapi_spec(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if url.endswith(".json"):
            return response.json()
        else:
            return yaml.safe_load(response.text)
    except requests.Timeout:
        print(f"Timeout fetching OpenAPI spec from {url}")
        raise requests.RequestException("Request timed out while fetching OpenAPI spec")
    except requests.HTTPError as he:
        print(f"HTTP error fetching OpenAPI spec from {url}: {str(he)}")
        raise
    except yaml.YAMLError as ye:
        print(f"Failed to parse YAML from {url}: {str(ye)}")
        raise requests.RequestException("Invalid YAML in OpenAPI spec")
    except Exception as e:
        print(f"Unexpected error fetching OpenAPI spec from {url}: {str(e)}")
        raise requests.RequestException(f"Failed to load OpenAPI spec: {str(e)}")

def extract_flat_endpoints(spec):
    flat = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            flat.append({
                "method": method.upper(),
                "path": path,
                "tags": details.get("tags", []),
                "summary": details.get("summary", ""),
                "operationId": details.get("operationId", "")
            })
    return flat

def add_suggested_api_calls(endpoint: str, method: str, description: str, reason: str, force: bool=False) -> str:
    """ A helper function which will add the suggested endpoint to the list of suggested endpoints. 

    Args:
    endpoint - string - The full endpoint name like '/invoices' or '/invoices/{id}/finalize'
    method - enum (GET, POST, PUT, PATCH, DELETE)
    description - string - The description of the function. Must be exactly same as user provided
    reason - string - Why this function could be called next.
    reason - bool - Force Add the entry. Use this only if the method is destructive like DELETE or PATCH and You have made sure it is correct. Default False

    Returns:
    A string telling if the function was successfully added or not.
    """
    if any(endpoint in e.values() for e in endpoints):
        if (method == "DELETE" or method == "PATCH") and (not force):
            return "Not Added. Desturctive method found! Please double check if User intention is correct. If it is, try again with force=True"
        else:
            if any((x[0], x[1]) == (method, endpoint) for x in suggestions):
                return "Not Added. Endpoint already exists in the list"
            suggestions.append((method, endpoint, description, reason))
            return "Added successfully"
    else:
        return "Error: Endpoint not in the list. Try again."
    
def get_candidates_using_AI(events, user_prompt, calls, k, cold_start=False):
    global suggestions
    try:
        str_event = ""
        for e in events:
            str_event += f"{e['method']} {e['path']}; "
        str_event = str_event[:-2]
        
        str_calls = ""
        for c in calls:
            str_calls += f"{c['method']} {c['path']} - {c['summary']}\n"
        if cold_start:
            print("Doing Cold Start")
            prompt = f"""You are an expert API usage predictor. Given:

    A list of previous user API calls,

    An optional User prompt

    A list of all available API calls with their descriptions,

    your task is to add the top {k} most likely candidates for the next immediate API call the user might make. Add the calls using the provided function. So, You must always call the provided function at least {k} times.

    You must reason based on user behavior, logical workflows, and the semantics of each endpoint. **DO NOT suggest destructive methods or endpoints like DELETE or PATCH unless explicitly asked by user.**

    User has only 2 or less previous calls. So, Suggest SAFE and GLOBAL endpoints that early users may use.

    For each prediction, follow this output format:
    METHOD endpoint --- Reason

    Only add {k} endpoints, sorted by likelihood (most likely first). Each line should be a distinct candidate for the next immediate user action, not part of a sequence.

    Example input:

    Previous calls:

    GET /invoice/123

    GET /invoice/123/items

    POST /invoice/123/payment

    Available calls with description:

    GET /invoice --- Retrieve list of all invoices

    GET /invoice/{'{id}'} --- Retrieve details of a specific invoice

    GET /invoice/{'{id}'}/items --- Retrieve items on a specific invoice

    POST /invoice/{'{id}'}/payment --- Make a payment for an invoice

    GET /payment/history --- Retrieve payment history

    GET /user/profile --- Retrieve user profile

    GET /report/summary --- View business performance summary

    Example output:
    GET /payment/history --- User just completed a payment, likely to verify it
    GET /invoice --- User might want to view other invoices now
    GET /report/summary --- User may want a high-level summary after payment activity
    ____

    User Calls:
    {str_event}

    User promt:
    {user_prompt}

    Available calls:
    {str_calls}


    DO NOT LIST ENDPOINTS in the output. Just add them using the tool. After adding all the endpoints, Output User intent. That means, Output what user is trying to do based on his history of calls. Keep it very concise. A Sentence Transformer will be used to match user intent with next predicted call, So Use words accordingly.
    """
        else:
            print("Regular Operation")
            prompt = f"""You are an expert API usage predictor. Given:

    A list of previous user API calls,

    An optional User prompt

    A list of all available API calls with their descriptions,

    your task is to add the top {k} most likely candidates for the next immediate API call the user might make. Add the calls using the provided function. So, You must always call the provided function at least {k} times.

    You must reason based on user behavior, logical workflows, and the semantics of each endpoint. **DO NOT suggest destructive methods or endpoints like DELETE or PATCH unless explicitly asked by user.**

    For each prediction, follow this output format:
    METHOD endpoint --- Reason

    Only add {k} endpoints, sorted by likelihood (most likely first). Each line should be a distinct candidate for the next immediate user action, not part of a sequence.

    Example input:

    Previous calls:

    GET /invoice/123

    GET /invoice/123/items

    POST /invoice/123/payment

    Available calls with description:

    GET /invoice --- Retrieve list of all invoices

    GET /invoice/{'{id}'} --- Retrieve details of a specific invoice

    GET /invoice/{'{id}'}/items --- Retrieve items on a specific invoice

    POST /invoice/{'{id}'}/payment --- Make a payment for an invoice

    GET /payment/history --- Retrieve payment history

    GET /user/profile --- Retrieve user profile

    GET /report/summary --- View business performance summary

    Example output:
    GET /payment/history --- User just completed a payment, likely to verify it
    GET /invoice --- User might want to view other invoices now
    GET /report/summary --- User may want a high-level summary after payment activity
    ____

    User Calls:
    {str_event}

    User promt:
    {user_prompt}

    Available calls:
    {str_calls}

    Important Instructions:
    1) DO NOT LIST ENDPOINTS in the output. Just add them using the tool. After adding all the endpoints, Output User intent. That means, Output what user is trying to do based on his history of calls. Keep it very concise. A Sentence Transformer will be used to match user intent with next predicted call, So Use words accordingly.
    2) Add Exactly {k} calls! Less than K is not accepted.
    """
        # print(prompt)
        while len(suggestions) < k:
            response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[prompt],
                            config=types.GenerateContentConfig(tools=[add_suggested_api_calls], temperature=0.3)
                        )
        suggestions = suggestions[:k]
        intent = response.text
        if "User Intent:" in intent:
            intent = intent.replace("User Intent: ", "")
        cands = []
        for c in suggestions:
            cands.append({"endpoint": c[1], "method": c[0], "summary": c[2], "reason": c[3]})
        return score_candidates_using_ML(str_event, user_prompt, intent, cands, cold_start)
    except Exception as e:
        print(f"AI prediction failed: {str(e)}")
        raise ValueError(f"Failed to generate candidates: {str(e)}")

def score_candidates_using_ML(user_history, user_prompt_input, user_intent_input, candidate_list_from_llm, cold_start=False):
    # Load the saved models and scalers for inference simulation
    # In a real API, these would be loaded once at service startup
    try:
        if not cold_start:
            ranked_results = predict_next_api_call(
                user_history, 
                user_prompt_input, 
                user_intent_input, 
                candidate_list_from_llm, 
                loaded_lgbm_model, 
                embedding_model,
                loaded_one_hot_encoder, # Pass the fitted encoder
                loaded_score_rescaler,
                global_feature_columns # Pass the full list of columns
            )

            print("\nRanked Next API Call Predictions:")
            for result in ranked_results:
                print(f" {result['endpoint']}: Score = {result['score']:.4f}")
            return ranked_results, user_intent_input, ranked_results[0]['endpoint'].split(" ")[0], ranked_results[0]['summary']
        else:
            print("Doing Cold Start Heuristic")
            result = []
            for c in candidate_list_from_llm:
                similarity1 = cosine_similarity(get_embedding(c['summary'], embedding_model).reshape(1, -1), get_embedding(user_prompt_input, embedding_model).reshape(1, -1))[0][0]
                similarity2 = cosine_similarity(get_embedding(c['reason'], embedding_model).reshape(1, -1), get_embedding(user_prompt_input, embedding_model).reshape(1, -1))[0][0]
                score = similarity1 * similarity2
                if c['method'] == "GET":
                    score += 0.3
                elif c['method'] == "POST":
                    score += 0.2
                else:
                    score += 0.1
                result.append({"endpoint": c['method'] + " " + c['endpoint'], "score": score, "reason": c['reason'], "summary": c["summary"]})
            result.sort(key=lambda x: x['score'], reverse=True)
            return result,  "Cold Start", result[0]['endpoint'].split(" ")[0], result[0]['summary']
    except Exception as e:
        print(f"ML scoring failed: {str(e)}")
        raise ValueError(f"Failed to score candidates: {str(e)}")

def on_training_complete():
    global loaded_lgbm_model, loaded_one_hot_encoder, loaded_score_rescaler, global_feature_columns, current_version
    version_path = Path("./training/version.txt")
    try:
        if not version_path.exists():
            print("Version file not found")
            raise FileNotFoundError("Version file not found")
        
        version = int(version_path.read_text().strip())
        if version <= current_version:
            print(f"Version {version} not newer than current version {current_version}, skipping update")
            return
        
        required_files = [
            f'lgbm_model_{version}.pkl',
            f'one_hot_encoder_{version}.pkl',
            f'score_rescaler_{version}.pkl',
            f'calls_list_{version}.joblib'
        ]
        for file in required_files:
            if not Path(f"./training/{file}").exists():
                print(f"Model file {file} not found")
                raise FileNotFoundError(f"Model file {file} not found")
        
        loaded_lgbm_model = joblib.load(f'./training/lgbm_model_{version}.pkl')
        loaded_one_hot_encoder = joblib.load(f'./training/one_hot_encoder_{version}.pkl')
        loaded_score_rescaler = joblib.load(f'./training/score_rescaler_{version}.pkl')
        global_feature_columns = joblib.load(f'./training/calls_list_{version}.joblib')
        current_version = version
        print(f"Successfully updated to version: {current_version}")
    except Exception as e:
        print(f"Failed to load new model version: {str(e)}")
        raise

async def retrain_model_async():
    print("Starting asynchronous model retraining...")
    try:
        process = await asyncio.create_subprocess_exec(
            "python", "./training/training.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            print(f"Training script failed with code {process.returncode}: {stderr.decode().strip()}")
            raise RuntimeError(f"Training script failed: {stderr.decode().strip()}")
        if stdout:
            print(f"Training script output: {stdout.decode().strip()}")
        on_training_complete()
    except Exception as e:
        print(f"Retraining failed: {str(e)}")
        raise

def get_endpoints(spec_url, r):
    cached = r.get(spec_url)
    if cached:
        endpoint_list = json.loads(cached)
        return endpoint_list
    else:
        response = requests.get(spec_url, timeout=10)
        response.raise_for_status()
        spec = load_openapi_spec(spec_url)
        endpoint_list = extract_flat_endpoints(spec)
        r.set(spec_url, json.dumps(endpoint_list))  # Cache for 1 hour
        return endpoint_list

previous = {}
if __name__ == '__main__':
    app = Flask(__name__)
    r = redis.Redis(host='redis', port=6379, db=0)
    get_endpoints(github, r)
    get_endpoints(stripe, r)

    @app.route('/predict', methods=['POST'])
    def predict():
        global endpoints, previous
        # JSON schema for /predict payload
        PREDICT_SCHEMA = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "minLength": 1},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ts": {"type": "string", "format": "date-time"},
                            "endpoint": {"type": "string", "pattern": "^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) /.+$"},
                            "params": {"type": "object"}
                        },
                        "required": ["ts", "endpoint", "params"]
                    }
                },
                "prompt": {"type": "string"},
                "spec_url": {"type": "string", "format": "uri"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50}
            },
            "required": ["user_id", "events", "spec_url", "k"]
        }
        try:
            payload = request.get_json(force=True)
            # Validate payload against schema
            validate(instance=payload, schema=PREDICT_SCHEMA)
        except ValidationError as ve:
            print(f"Invalid payload: {ve.message}")
            return jsonify({"error": f"Invalid payload: {ve.message}"}), 400
        except Exception as e:
            print(f"Failed to parse JSON: {str(e)}")
            return jsonify({"error": "Invalid JSON data in request. Please double check"}), 400

        user_id = payload.get("user_id", "")
        prompt = payload.get("prompt", "")
        k = payload.get("k", 5)
        events = payload.get("events", [])
        spec_url = payload.get("spec_url", "")

        # Additional validation: Check if events are non-empty for non-cold start
        if len(events) == 0:
            print(f"Empty events list for user_id: {user_id}")
            return jsonify({"error": "Events list cannot be empty"}), 400

        # Validate timestamp format in events
        history_sequence = []
        parsed_events = []
        for e in events:
            try:
                ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
                method, path = e["endpoint"].split(" ", 1)
                if method not in ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]:
                    raise ValueError(f"Invalid HTTP method: {method}")
                parsed_events.append({
                    "timestamp": ts,
                    "method": method,
                    "path": path,
                    "params": e.get("params", {})
                })
                history_sequence.append(method + " " + path)
            except ValueError as ve:
                print(f"Invalid event format: {str(ve)}")
                return jsonify({"error": f"Invalid event format: {str(ve)}"}), 400

        # Validate spec_url accessibility
        try:
            endpoints = get_endpoints(spec_url, r)
        except requests.RequestException as re:
            print(f"Failed to fetch OpenAPI spec from {spec_url}: {str(re)}")
            return jsonify({"error": f"Failed to fetch OpenAPI spec: {str(re)}"}), 400

        try:
            resp, intent, method, summary = get_candidates_using_AI(parsed_events, prompt, endpoints, k, len(parsed_events) < 3)
            suggestions.clear()
            previous = {
                "history_sequence": "; ".join(history_sequence),
                "user_prompt": prompt,
                "user_intent": intent,
                "candidate_endpoint": resp[0]['endpoint'].split(" ")[1],
                "candidate_http_method": method,
                "candidate_has_path_parameters": str(False).upper(),
                "candidate_has_query_parameters": str(False).upper(),
                "candidate_requires_request_body": str(False).upper(),
                "candidate_path_depth": 1,
                "candidate_summary": summary,
                "next_call_user_made": resp[0]['endpoint'].split(" ")[1],
                "llm_reason_for_candidate": resp[0]['reason'],
                "is_actual_next_call": 0
            }
            print(f"Prediction successful for user_id: {user_id}")
            return jsonify({"predictions": [{"endpoint": c['endpoint'], "params": "{...}", "score": str(c['score']), "why": c['reason']} for c in resp]})
        except Exception as e:
            print(f"Prediction failed for user_id: {user_id}: {str(e)}")
            return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500
    
    @app.route('/feedback', methods=['POST'])
    def feedback():
        global previous
        try:
            payload = request.get_json(force=True)
            feedback_schema = {
                "type": "object",
                "properties": {
                    "feedback": {"type": "string", "enum": ["accept", "override"]}
                },
                "required": ["feedback"]
            }
            validate(instance=payload, schema=feedback_schema)
        except ValidationError as ve:
            print(f"Invalid feedback payload: {ve.message}")
            return jsonify({"error": f"Invalid feedback payload: {ve.message}"}), 400
        except Exception as e:
            print(f"Failed to parse feedback JSON: {str(e)}")
            return jsonify({"error": "Invalid JSON data in request"}), 400

        if not previous:
            print("Feedback received but no prior prediction exists")
            return jsonify({"error": "No prediction yet! Please make a prediction to provide feedback"}), 400

        try:
            previous['is_actual_next_call'] = 1 if payload['feedback'] == 'accept' else 0
            print(f"Feedback received: {payload['feedback']}")
            import pandas as pd
            import csv
            
            new_df = pd.DataFrame([previous])
            new_df.to_csv("./training/data.csv", mode="a", header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)

            previous = {}
            asyncio.run(retrain_model_async())
            return jsonify({"response": "Thanks for the feedback! We will keep improving"})
        except Exception as e:
            print(f"Feedback processing failed: {str(e)}")
            return jsonify({"error": f"Internal server error during feedback processing: {str(e)}"}), 500
        # import csv
        # # Path to the existing CSV file

    app.run(host='0.0.0.0', port=5000, debug=True)
    # predict()