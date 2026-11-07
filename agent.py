"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "TotallyAwesomeDatathonParticipant"
AGENT_NAME = "JustAsAwesomeAgent"


# ===== YOUR CODE HERE =====

import time

PARTICIPANT = "TAMU-Datathon-CYH"
AGENT_NAME = "ShortDeckPLO-v1"

try:
    import equity as _E          # numpy-based equity (preflop table + Monte Carlo)
    import strategy as _S
    _E.preflop_strength((0, 4, 8, 12, 16))   # touch the table now, not in the first decision
    _E.equity((0, 4, 8, 12, 16), (1, 5, 9))  # warm caches / random banks
    _E.equity((0, 4, 8, 12, 16), (1, 5, 9), opp_top=0.5)   # lazy sort for range-restricted queries
    _E.equity((0, 4, 8, 12, 16), (), samples=200, opp_top=0.5)
    _E.clear_cache()
    _IMPORT_ERROR = None
except Exception as _exc:          # keep the server alive even if a module breaks
    _E = _S = None
    _IMPORT_ERROR = _exc

try:
    from opponent import OpponentModel as _OpponentModel
except Exception:
    _OpponentModel = None


_OVERHEAD = 0.0002   # seconds per request not seen by perf_counter inside choose_action


def _fallback(obs: Observation) -> Action:
    """Cheap, always-legal action: check when free, else call small / fold big."""
    try:
        legal = obs.legal_actions
        if obs.to_call == 0 or Action.FOLD not in legal:
            return Action.CHECK_CALL
        # call when the price is small, otherwise fold
        if obs.to_call * 4 <= obs.pot:
            return Action.CHECK_CALL
        return Action.FOLD
    except Exception:
        return Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["t_used"] = 0.0
    memory["errors"] = 0
    try:
        memory["stack"] = int(config.starting_stack)
        memory["t_bank"] = float(config.time_bank_seconds)
    except Exception:
        memory["stack"] = 80
        memory["t_bank"] = 8.0
    try:
        memory["opp"] = _OpponentModel(config) if _OpponentModel else None
    except Exception:
        memory["opp"] = None


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    action = None
    try:
        legal = obs.legal_actions
        if "t_used" not in memory:          # /act before /new-half data (defensive)
            new_half(AgentConfig(), memory)
        prof = None
        opp = memory.get("opp")
        if opp is not None:
            try:
                opp.update_from_obs(obs)
                prof = opp.profile()
            except Exception:
                prof = None
        # Time bank nearly empty -> cheap fallback only.
        if _S is not None and memory["t_used"] < 0.9 * memory.get("t_bank", 8.0):
            action = _S.decide(obs, prof or {}, memory)
        if action not in legal:
            action = _fallback(obs)
        if action not in legal:
            action = Action.CHECK_CALL
    except Exception:
        memory["errors"] = memory.get("errors", 0) + 1
        action = _fallback(obs)
    finally:
        try:
            # + estimated per-request server/JSON overhead the judge also charges
            memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0) + _OVERHEAD
        except Exception:
            pass
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    t0 = time.perf_counter()
    try:
        opp = memory.get("opp")
        if opp is not None:
            opp.on_hand_end(result)
    except Exception:
        pass
    try:
        memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0)
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
