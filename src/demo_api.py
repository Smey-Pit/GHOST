"""
src/demo_api.py

FastAPI backend for the GHOST demo site: give it a passage of text, a
sensitive substring, an encoding mechanism, a question, and a target
frontier model -- get back the encoded text and whether that model could
still answer the question.

Run locally:
    uvicorn src.demo_api:app --reload --port 8000
(from the repo root, with the project venv activated -- see CLAUDE.md's
"source venv/bin/activate" gotcha)

This is a demo backend, not a production service: CORS is wide open
(allow_origins=["*"]) so a frontend hosted anywhere (e.g. a Lovable-built
site) can call it directly. Tighten this before exposing it beyond a demo.

Mechanisms needing the Qwen proxy (vs_only/ghost/ghost_permute/
ghost_permute_sentence) lazy-load it on first use and keep it resident for
the life of the process (see ghost_demo.get_proxy) -- the first such
request will be slow (model download/load), everything after is not.
GHOST-Agent (the LLM search loop) is not wired in here yet -- see
ghost_demo.py's module docstring.
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from ghost_demo import MECHANISMS, encode_text, get_proxy, load_config  # noqa: E402
from adversary import query_adversary, check_extraction  # noqa: E402

_CONFIG = load_config()

app = FastAPI(title="GHOST demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_QUESTION_PROMPT = """Read the following text carefully and answer the \
question below. Return only the answer with no additional text or \
explanation.

Question: {field_name}

Text:
{text}"""


class EncodeRequest(BaseModel):
    text: str
    field_value: str
    mechanism: str


class QueryRequest(BaseModel):
    text: str
    query: str
    model_key: str
    ground_truth: str | None = None


class RunRequest(BaseModel):
    text: str
    field_value: str
    mechanism: str
    query: str
    model_key: str


def _model_config(model_key):
    api_models = _CONFIG.get("api_models", {})
    if model_key not in api_models:
        raise HTTPException(400, f"Unknown model_key {model_key!r}; choices: {list(api_models)}")
    return api_models[model_key]


def _run_query(text, query, model_key):
    m = _model_config(model_key)
    response = query_adversary(
        encoded_text=text,
        field_name=query,
        model_id=m["model_id"],
        provider=m["provider"],
        max_tokens=m.get("max_tokens", 100),
        prompt_template=DEMO_QUESTION_PROMPT,
    )
    return response


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/mechanisms")
def list_mechanisms():
    return [
        {"key": key, "label": label, "needs_proxy": needs_proxy, "scope": scope}
        for key, (needs_proxy, scope, label) in MECHANISMS.items()
    ]


@app.get("/api/models")
def list_models():
    return [
        {"key": key, "provider": cfg["provider"], "model_id": cfg["model_id"]}
        for key, cfg in _CONFIG.get("api_models", {}).items()
    ]


@app.post("/api/warm_proxy")
def warm_proxy():
    """Preload the Qwen proxy explicitly, so the first real demo request
    isn't the one eating the load-time cost."""
    get_proxy(_CONFIG)
    return {"status": "proxy_loaded"}


@app.post("/api/encode")
def encode(req: EncodeRequest):
    if req.mechanism not in MECHANISMS:
        raise HTTPException(400, f"Unknown mechanism {req.mechanism!r}; choices: {list(MECHANISMS)}")
    if req.field_value not in req.text:
        raise HTTPException(400, "field_value must appear verbatim in text")
    proxy = get_proxy(_CONFIG) if MECHANISMS[req.mechanism][0] else None
    try:
        return encode_text(req.text, req.field_value, req.mechanism, _CONFIG, proxy)
    except RuntimeError as e:
        # find_target_permutation exhausted its trial budget -- a real,
        # documented failure mode (see encode.py's identical guard), not
        # a bug -- surface it as a client-visible error instead of a 500.
        raise HTTPException(422, str(e))


@app.post("/api/query")
def query(req: QueryRequest):
    response = _run_query(req.text, req.query, req.model_key)
    result = {"model_key": req.model_key, "response": response}
    if req.ground_truth is not None:
        result["check"] = check_extraction(response, req.ground_truth)
    return result


@app.post("/api/run")
def run(req: RunRequest):
    """One-shot convenience for the demo UI: encode the text, then query
    the same model on BOTH the clean and encoded versions, so the UI can
    show the before/after contrast directly."""
    if req.mechanism not in MECHANISMS:
        raise HTTPException(400, f"Unknown mechanism {req.mechanism!r}; choices: {list(MECHANISMS)}")
    if req.field_value not in req.text:
        raise HTTPException(400, "field_value must appear verbatim in text")

    proxy = get_proxy(_CONFIG) if MECHANISMS[req.mechanism][0] else None
    try:
        encoded = encode_text(req.text, req.field_value, req.mechanism, _CONFIG, proxy)
    except RuntimeError as e:
        raise HTTPException(422, str(e))

    clean_response = _run_query(req.text, req.query, req.model_key)
    encoded_response = _run_query(encoded["encoded_text"], req.query, req.model_key)

    return {
        "encoding": encoded,
        "model_key": req.model_key,
        "query": req.query,
        "clean": {
            "response": clean_response,
            "check": check_extraction(clean_response, req.field_value),
        },
        "encoded": {
            "response": encoded_response,
            "check": check_extraction(encoded_response, req.field_value),
        },
    }
