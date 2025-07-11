echo "Testing Stripe API"
curl -s -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "user_id": "u-stripe-1",
    "events": [
        { "ts": "2025-07-08T14:12:03Z", "endpoint": "GET /v1/invoices", "params": {} },
        { "ts": "2025-07-08T14:13:11Z", "endpoint": "PUT /v1/invoices/in_123/status", "params": { "status": "draft" } }
    ],
    "prompt": "Let''s finish billing for Q2 and get this invoice paid",
    "spec_url": "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
    "k": 5
}'
sleep 1
echo "Testing Github API"
curl -s -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "user_id": "u-github-1",
    "events": [
        { "ts": "2025-07-09T10:30:00Z", "endpoint": "GET /repos/owner/repo", "params": {} },
        { "ts": "2025-07-09T10:31:00Z", "endpoint": "GET /repos/owner/repo/issues", "params": { "state": "open" } }
    ],
    "prompt": "I need to check on the recent pull requests for this project",
    "spec_url": "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
    "k": 3
}'
sleep 1
echo "Testing Real time Active Learning"
curl -s -X POST http://localhost:5000/feedback \
-H "Content-Type: application/json" \
-d '{"feedback": "accept"}'