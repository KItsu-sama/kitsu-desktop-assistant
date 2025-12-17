# Action Token Specification (Kitsu)

## Purpose
Small internal-only markers emitted by the LLM to request narrow, safe tool usage
(e.g., search, get_time, continue). Tokens are machine-only and MUST be stripped
from any user-visible output.

## Token formats (exact; case-sensitive)
- `<action:search query="...">` — query string required (max 256 chars)
- `<action:get_time>` — no arguments
- `<action:continue reason="...">` — optional reason (max 128 chars)

Each token MUST appear on its own line. Tokens embedded inline within other text
are ignored.

## Parser behavior
- Synchronous, pure, deterministic parsing.
- Only one action token allowed per LLM response; multiple tokens are an error.
- Strict validation and sanitization: remove control characters and reject overlong fields.
- Returns a typed dataclass `Action(kind, payload, raw)` or an error structure.

## Executor stubs
- `execute_search(query, max_results)` — deterministic, offline-friendly fake results.
- `execute_get_time(timezone, now)` — returns ISO-formatted time; tests may inject `now`.
- `handle_continue(reason, continuation_count, max_allowed)` — returns whether another pass is allowed and updated count.

Executors return structured machine-only outputs (JSON-like dicts). The LLM is not
re-invoked automatically by executors; the meta-controller decides whether a
continuation re-invocation is appropriate.

## Controller flow (text diagram)
LLM -> (raw output with possible action token) -> parser -> meta-controller
-> {approve/deny/require_confirmation + executor stub}
-> if approved and executor present: call executor -> (tool result)
-> inject machine-only tool result into a single LLM continuation pass (if approved)

Note: The machine-only injection line looks like:

    [INTERNAL] Tool result (machine-only): <tool structured JSON>

and must be stripped before any user-visible presentation.

## Security rules
- Strip all action tokens and machine-only lines before sending content to the user.
- Only a single action token per LLM response is allowed.
- Sanitize attribute values and reject overlong fields.

## Comparison with Neuro-sama (public information only)
- Kitsu uses *machine-only* action markers (tokens are stripped from user output); Neuro-sama's public flow uses visible tags in some implementations.
- Kitsu employs a tiny, deterministic meta-controller with strict permission and emotion gating to keep behavior low-overhead, while Neuro-sama's design includes more extensive parsing and reward/looping features (publicly discussed).
- Kitsu's executor stubs are purposely small and offline-friendly (no network calls during tests); Neuro-sama integrates richer tooling by design.

***

This design favors minimal runtime cost, determinism, and safety for low-spec hardware.
If needed later, retrieval and time services can be integrated behind the executor hooks
(e.g., `core.memory.retrieval.search`).
