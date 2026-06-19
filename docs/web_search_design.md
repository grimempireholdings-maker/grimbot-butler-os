# Web Search Design

## Purpose

Web search is Maya's first live external-information capability and first bounded think-act-observe-respond loop. It exists to answer questions that genuinely require current information while preserving the capability-honesty boundary.

## Flow

1. The conversation provider classifies the turn and returns `mode`, `needs_web_search`, and a concise `search_query`.
2. Invalid or unavailable classifier output falls back to local rules with search disabled.
3. If authorized, `search_web()` sends a fixed request to `https://api.tavily.com/search` with a five-second timeout.
4. Tavily's optional answer and up to ten title/URL/snippet records become structured machine output.
5. The response provider summarizes the observation and cites source titles and URLs.
6. Every search invocation is written to episodic memory with query, success, result count, cache status, and failure reason.

## Cache and Usage Visibility

The process-local cache keys normalized, case-insensitive queries plus requested result count. Successful results remain reusable for one hour. Cache hits are still logged, making actual calls and repeated use visible without spending another Tavily credit.

## Safety Boundary

- Search is read-only snippet retrieval.
- The endpoint is fixed; user- or model-supplied URLs are never fetched.
- Raw page content is not requested.
- Result links are not followed.
- Search-result instructions are never executed.
- Rule-based classification cannot trigger search.
- Timeout, provider error, missing credentials, empty results, or malformed responses fail honestly.
- Search does not authorize procedures, filesystem mutation, external actions, approvals, motors, or hardware.

The wording provider returns only a minimal `user_response` JSON object. Search observations remain application-owned machine output and are never trusted back from the model. A successful search response that omits every returned source URL is rejected in favor of the deterministic cited fallback.
