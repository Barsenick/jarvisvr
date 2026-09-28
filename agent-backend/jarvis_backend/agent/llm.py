"""LLM provider abstraction + implementations.

This is a *self-contained* implementation of the envelope + payload schemas
defined in ``docs/PROTOCOL.md``. The ``shared-protocol/`` package will later
publish canonical Python bindings generated from the JSON Schema; when that
lands, reconcile this module against it (the file names/shapes here mirror the
protocol doc exactly, so the swap should be mechanical).

Design notes
------------
* ``Envelope`` keeps ``payload`` as a free-form dict. Typed payload models below
  are used to *build* and *validate* specific message types, but the transport
  layer never rejects a message just because its payload has extra keys
  (forward-compatibility, per the protocol's conformance checklist).
* All payload models ignore unknown keys (``extra="ignore"``).
* Helpers assign a uuid-v4 ``id`` and epoch-millisecond ``ts`` automatically.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .. import protocol

log = logging.getLogger("jarvis.llm")
log.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None  # assistant -> requested calls
    tool_call_id: Optional[str] = None  # role == "tool"
    name: Optional[str] = None  # tool name (role == "tool")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-schema function parameters


@dataclass
class ImageInput:
    """A multimodal image attached to the latest user turn (v1.1 vision)."""

    b64: str
    media_type: str = "image/jpeg"


@dataclass
class LLMResult:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMUnavailable(Exception):
    """Raised when a real provider can't be used (missing key/SDK)."""


class LLMProvider(ABC):
    name: str = "base"
    model: str = "base"
    # Whether this provider can natively consume image inputs.
    supports_vision: bool = False

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        ...


# ---------------------------------------------------------------------------
# Deterministic mock planner
# ---------------------------------------------------------------------------

# A few well-known cities for nicer demo data + better extraction.
_KNOWN_CITIES = {
    "tokyo": "Tokyo",
    "london": "London",
    "paris": "Paris",
    "new york": "New York",
    "seattle": "Seattle",
    "berlin": "Berlin",
    "sydney": "Sydney",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "singapore": "Singapore",
    "dubai": "Dubai",
}

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60,
}

_STOP_TRAIL = re.compile(
    r"\b(please|now|right now|today|tonight|for me|thanks|thank you)\b.*$", re.I
)


def extract_city(text: str) -> str:
    """Best-effort city extraction from a weather request. Title-cased."""
    low = text.lower()
    # 1) explicit "... in/for/at/of <city>"
    m = re.search(r"\b(?:in|for|at|of)\s+([a-z][a-z .'-]+)", low)
    candidate = ""
    if m:
        candidate = m.group(1)
    else:
        # 2) "weather <city>"
        m2 = re.search(r"\bweather\s+([a-z][a-z .'-]+)", low)
        if m2:
            candidate = m2.group(1)
    if candidate:
        # Cut at conjunctions / extra clauses.
        candidate = re.split(
            r"\b(and|then|with|plus|also|while)\b", candidate
        )[0]
        candidate = _STOP_TRAIL.sub("", candidate)
        candidate = candidate.strip(" .,!?'-")
        if candidate:
            if candidate in _KNOWN_CITIES:
                return _KNOWN_CITIES[candidate]
            return " ".join(w.capitalize() for w in candidate.split())
    # 3) any known city mentioned anywhere
    for key, nice in _KNOWN_CITIES.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return nice
    return "San Francisco"


def extract_duration_seconds(text: str) -> int:
    """Parse a timer duration from text; defaults to 60s."""
    low = text.lower().replace("-", " ")
    total = 0
    found = False
    for value, unit in re.findall(
        r"(\d+(?:\.\d+)?|[a-z]+)\s*"
        r"(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
        low,
    ):
        if value.isalpha():
            if value not in _WORD_NUMBERS:
                continue
            n = float(_WORD_NUMBERS[value])
        else:
            n = float(value)
        if unit.startswith("h"):  # h / hr / hrs / hour(s)
            total += n * 3600
        elif unit.startswith("m"):  # m / min / mins / minute(s)
            total += n * 60
        elif unit.startswith("s"):  # s / sec / secs / second(s)
            total += n
        found = True
    if not found:
        return 60
    return int(total) or 60


_LANGS = "spanish|french|german|japanese|hindi|english|italian|chinese|es|fr|de|ja|hi|it|zh"

_COMPANY_TICKERS = {
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA", "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "netflix": "NFLX",
}


def _clean(s: str) -> str:
    return s.strip(" .,!?'\"-")


def extract_target_lang(text: str) -> str:
    m = re.search(rf"\b(?:to|into|in)\s+({_LANGS})\b", text, re.I)
    return m.group(1).lower() if m else "spanish"


def extract_object_name(text: str) -> str:
    """Pull the object name from find/remember requests, e.g. 'my keys'."""
    low = text.lower()
    patterns = [
        r"where did i (?:leave|put|place)\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$| again| earlier)",
        r"where(?:'s| is| are)\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$)",
        r"have you seen\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$)",
        r"remember (?:that |where )?(?:my |the |your )?([a-z][a-z ]*?)\s+(?:is|are|'s)\b",
        r"(?:this|that) is (?:my |the )?([a-z][a-z ]*?)(?:\?|$|,| —| -)",
    ]
    for p in patterns:
        m = re.search(p, low)
        if m:
            return _clean(m.group(1)) or "it"
    return "it"


def extract_search_query(text: str) -> str:
    m = re.search(r"\b(?:search the web for|search for|search|look up|google)\s+(.+)", text, re.I)
    return _clean(m.group(1)) if m else ""


def extract_destination(text: str) -> str:
    m = re.search(
        r"\b(?:navigate to|directions to|take me to|how do i get to|guide me to|route to)\s+(.+)",
        text,
        re.I,
    )
    return _clean(m.group(1)) if m else ""


def extract_symbols(text: str) -> list[str]:
    syms = re.findall(r"\b([A-Z]{2,5})\b", text)
    syms = [s for s in syms if s not in {"USD", "AI", "VR", "CEO", "ID"}]
    if syms:
        return syms
    low = text.lower()
    found = [tk for name, tk in _COMPANY_TICKERS.items() if name in low]
    return found or ["AAPL", "TSLA", "NVDA"]


def _clean(s: str) -> str:
    return s.strip(" .,!?'\"-")


def extract_target_lang(text: str) -> str:
    m = re.search(rf"\b(?:to|into|in)\s+({_LANGS})\b", text, re.I)
    return m.group(1).lower() if m else "spanish"


def extract_object_name(text: str) -> str:
    """Pull the object name from find/remember requests, e.g. 'my keys'."""
    low = text.lower()
    patterns = [
        r"where did i (?:leave|put|place)\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$| again| earlier)",
        r"where(?:'s| is| are)\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$)",
        r"have you seen\s+(?:my |the |your )?([a-z][a-z ]*?)(?:\?|$)",
        r"remember (?:that |where )?(?:my |the |your )?([a-z][a-z ]*?)\s+(?:is|are|'s)\b",
        r"(?:this|that) is (?:my |the )?([a-z][a-z ]*?)(?:\?|$|,| —| -)",
    ]
    for p in patterns:
        m = re.search(p, low)
        if m:
            return _clean(m.group(1)) or "it"
    return "it"


def extract_search_query(text: str) -> str:
    m = re.search(r"\b(?:search the web for|search for|search|look up|google)\s+(.+)", text, re.I)
    return _clean(m.group(1)) if m else ""


def extract_destination(text: str) -> str:
    m = re.search(
        r"\b(?:navigate to|directions to|take me to|how do i get to|guide me to|route to)\s+(.+)",
        text,
        re.I,
    )
    return _clean(m.group(1)) if m else ""


def extract_symbols(text: str) -> list[str]:
    syms = re.findall(r"\b([A-Z]{2,5})\b", text)
    syms = [s for s in syms if s not in {"USD", "AI", "VR", "CEO", "ID"}]
    if syms:
        return syms
    low = text.lower()
    found = [tk for name, tk in _COMPANY_TICKERS.items() if name in low]
    return found or ["AAPL", "TSLA", "NVDA"]


def _new_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=protocol.new_id(), name=name, arguments=arguments)


def _plan_perception(text: str, low: str, want, calls: list[ToolCall]) -> None:
    """Append v1.1 perception + knowledge tool calls for matching intents."""
    # Vision: identify the focused object vs describe the whole view.
    if want("identify_object") and re.search(
        r"\bwhat(?:'s| is)? this\b|\bwhat am i looking at\b|\bwhat(?:'s| is)? that\b|"
        r"\bidentify\b|\blook at (?:this|that|it)\b",
        low,
    ):
        calls.append(_new_call("identify_object", {}))
    elif want("describe_view") and re.search(
        r"\bwhat do you see\b|\bwhat can you see\b|\bdescribe (?:the )?(?:room|view|scene|this|surroundings)\b|"
        r"\blook around\b|\btake a look\b|\bwhat'?s (?:in|around) (?:the|my) room\b|\bwhat'?s in front of me\b",
        low,
    ):
        calls.append(_new_call("describe_view", {}))
 
    # OCR read (avoid clashing with note listing).
    if (
        want("read_text")
        and re.search(r"\bread (?:this|the|that|it)\b|\bwhat does (?:this|it|that) say\b|\bocr\b", low)
        and not re.search(r"\bnotes?\b", low)
    ):
        calls.append(_new_call("read_text", {}))
 
    # Translate (view vs free text).
    if re.search(r"\btranslate\b", low) and (want("translate_view") or want("translate_text")):
        target = extract_target_lang(low)
        if re.search(
            r"\btranslate (?:this|it|that|the sign|the label|the menu|the text|the view)\b|\bread and translate\b",
            low,
        ) and want("translate_view"):
            calls.append(_new_call("translate_view", {"target_lang": target}))
        elif want("translate_text"):
            m = re.search(r"\btranslate\s+(.+)", text, re.I)
            phrase = m.group(1) if m else ""
            phrase = re.sub(rf"\s+(?:to|into)\s+({_LANGS})\b.*$", "", phrase, flags=re.I)
            phrase = _clean(phrase.strip('"\u201c\u201d'))
            if phrase:
                calls.append(_new_call("translate_text", {"text": phrase, "target_lang": target}))
            elif want("translate_view"):
                calls.append(_new_call("translate_view", {"target_lang": target}))
 
    # Spatial memory: remember vs find.
    if (
        want("remember_object")
        and re.search(r"\bremember\b", low)
        and re.search(r"\b(is|are|'s)\b.*\b(here|there|on|in|under|next to|beside|by)\b", low)
    ):
        calls.append(_new_call("remember_object", {"name": extract_object_name(text)}))
    elif want("find_object") and re.search(
        r"\bwhere did i (?:leave|put|place)\b|\bwhere(?:'s| is| are)\b.*\b(my|the)\b|"
        r"\bfind (?:my|the)\b|\bhave you seen (?:my|the)\b",
        low,
    ):
        calls.append(_new_call("find_object", {"name": extract_object_name(text)}))
 
    # Sound events.
    if want("identify_sound") and re.search(
        r"\bwhat was that (?:sound|noise)\b|\bdid you hear that\b|\bwhat(?:'s| is) that (?:sound|noise)\b",
        low,
    ):
        calls.append(_new_call("identify_sound", {}))
 
    # Web search / knowledge.
    if want("web_search") and re.search(r"\b(search the web|search for|search|look up|google)\b", low):
        q = extract_search_query(text)
        if q:
            calls.append(_new_call("web_search", {"query": q}))
 
    if want("get_news") and re.search(r"\b(news|headlines)\b", low):
        m = re.search(r"\bnews (?:about|on|for)\s+(.+)", low)
        calls.append(_new_call("get_news", {"topic": _clean(m.group(1)) if m else ""}))
 
    if want("get_stocks") and re.search(r"\bstock(?:s| price| quote)?\b|\bshare price\b|\bticker\b", low):
        calls.append(_new_call("get_stocks", {"symbols": extract_symbols(text)}))
 
    if want("get_calendar") and re.search(
        r"\b(calendar|schedule|agenda|my day|appointments?|what'?s on today)\b", low
    ):
        calls.append(_new_call("get_calendar", {}))
 
    if want("navigate_to") and re.search(
        r"\b(Navigate to|directions to|take me to|how do i get to|guide me to|route to)\b", low
    ):
        dest = extract_destination(text)
        if dest:
            calls.append(_new_call("navigate_to", {"destination": dest}))
 
    if want("measure") and re.search(
        r"\b(measure|how far|how wide|how tall|how big|how long is|distance (?:to|between|from))\b", low
    ):
        calls.append(_new_call("measure", {}))
 
 
def plan_tool_calls(text: str, available: set[str]) -> tuple[list[ToolCall], Optional[str]]:
    """Deterministically map user text -> tool calls (the mock 'planner').
 
    Returns ``(tool_calls, direct_reply)``. If ``tool_calls`` is empty and
    ``direct_reply`` is set, the agent should speak it directly with no tools.
    """
    low = text.lower().strip()
    calls: list[ToolCall] = []
 
    def want(name: str) -> bool:
        return name in available
 
    # Greeting / small talk -> speak directly, no tools.
    if re.fullmatch(r"(hi|hello|hey|yo|hiya)[ ,!.]*(jarvis)?[ ,!.]*", low):
        return [], "Hello. Jarvis here — how can I help?"
    if re.search(r"\b(thanks|thank you|cheers)\b", low):
        return [], "You're welcome."
 
    # Weather
    if want("get_weather") and re.search(r"\bweather|forecast|temperature\b", low):
        calls.append(_new_call("get_weather", {"city": extract_city(text)}))
 
    # Timer: start vs stop
    if re.search(r"\btimer\b|\bcountdown\b", low) or re.search(
        r"\bset (?:a|an)\b.*\b(minute|second|hour)", low
    ):
        if re.search(r"\b(stop|cancel|clear|kill|end)\b", low) and want("stop_timer"):
            calls.append(_new_call("stop_timer", {}))
        elif want("start_timer"):
            calls.append(
                _new_call(
                    "start_timer",
                    {"duration_seconds": extract_duration_seconds(text)},
                )
            )
 
    # Current time
    if want("get_time") and re.search(r"\bwhat(?:'s| is)? the time\b|\bcurrent time\b|\bwhat time\b", low):
        calls.append(_new_call("get_time", {}))
 
    # Notes
    if want("list_notes") and re.search(r"\b(list|show|read|what are).{0,12}notes?\b", low):
        calls.append(_new_call("list_notes", {}))
    elif want("take_note") and re.search(r"\b(take a note|note that|make a note|note:|remember that)\b", low):
        note = re.sub(
            r"^.*?\b(take a note|note that|make a note|note:|remember that)\b[:,]?\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        calls.append(_new_call("take_note", {"text": note or text}))
 
    # Reminders
    if want("set_reminder") and re.search(r"\bremind me\b|\breminder\b", low):
        what = re.sub(r"^.*?\bremind me (?:to|that)?\s*", "", text, flags=re.I).strip()
        calls.append(_new_call("set_reminder", {"text": what or text}))
 
    # v1.1 perception + knowledge intents (sight, sound, spatial recall, search…).
    _plan_perception(text, low, want, calls)
 
    # Explicit "open <widget>"
    if want("open_widget"):
        mo = re.search(r"\bopen (?:the |a |an )?([a-z_ ]+?)(?: widget| panel)?$", low)
        if mo and not calls:
            widget = mo.group(1).strip().replace(" ", "_")
            calls.append(_new_call("open_widget", {"widget_type": widget, "props": {}}))
 
    # Fallback: show a helpful panel echoing the request.
    if not calls:
        if want("show_text"):
            calls.append(
                _new_call(
                    "show_text",
                    {
                        "title": "Jarvis",
                        "text": (
                            "I can show the weather, set timers, take notes, set "
                            "reminders, tell the time, and open holographic widgets. "
                            f"You said: “{text.strip()}”."
                        ),
                    },
                )
            )
        else:
            return [], "I can help with weather, timers, notes, reminders, and more."
    return calls, None


class MockLLM(LLMProvider):
    """Deterministic provider: keyword intent planning + result-driven replies."""

    name = "mock"
    model = "mock"
    supports_vision = True  # via deterministic offline scene synthesis (tools)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        # The mock "sees" via perception context attached to messages + vision
        # tools; raw image bytes (``images``) are ignored offline.
        available = {t.name for t in tools}

        # Find the last user message and whether tools already ran for this turn.
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m.role == "user":
                last_user_idx = i
        tool_msgs = [
            m for m in messages[last_user_idx + 1 :] if m.role == "tool"
        ] if last_user_idx >= 0 else []

        # Finalize: synthesize a spoken reply from tool observations.
        if tool_msgs:
            return LLMResult(content=self._summarize_results(tool_msgs))

        # Plan: turn the latest user text into tool calls (or a direct reply).
        user_text = messages[last_user_idx].content if last_user_idx >= 0 else ""
        calls, direct = plan_tool_calls(user_text or "", available)
        if calls:
            return LLMResult(content=None, tool_calls=calls)
        return LLMResult(content=direct or "Done.")

    @staticmethod
    def _summarize_results(tool_msgs: list[LLMMessage]) -> str:
        speeches: list[str] = []
        for m in tool_msgs:
            if not m.content:
                continue
            try:
                data = json.loads(m.content)
            except (ValueError, TypeError):
                continue
            spoken = data.get("speech") if isinstance(data, dict) else None
            if spoken:
                speeches.append(str(spoken))
        return " ".join(speeches) if speeches else "Done."


# ---------------------------------------------------------------------------
# OpenAI provider (optional dep)
# ---------------------------------------------------------------------------


def _openai_tools_payload(toys: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in toys
    ]


def _parse_openai_tool_calls(raw_tool_calls) -> list[ToolCall]:
    """Parse OpenAI-style tool_calls (SDK objects or plain dicts) -> ToolCall[]."""
    calls: list[ToolCall] = []
    for tc in raw_tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {}) or {}
            tc_id = tc.get("id") or protocol.new_id()
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
        else:  # SDK object
            fn = getattr(tc, "function", None)
            tc_id = getattr(tc, "id", None) or protocol.new_id()
            name = getattr(fn, "name", "") if fn else ""
            raw_args = getattr(fn, "arguments", None) if fn else None
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except ValueError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if name:
            calls.append(ToolCall(id=tc_id, name=name, arguments=args))
    return calls


class OpenAILLM(LLMProvider):
    name = "openai"
    supports_vision = True

    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("OPENAI_API_KEY not set")
        try:
            from openai import AsyncOpenAI  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"openai SDK not installed ({exc})") from exc
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        client = self._get_client()
        oai_messages = [_to_openai_message(m) for m in messages]
        if images:
            _openai_attach_images(oai_messages, images)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": 0,
        }
        payload_toys = _openai_tools_payload(toys)
        if payload_toys:
            kwargs["tools"] = payload_toys
            kwargs["tool_choice"] = "auto"
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        return LLMResult(
            content=choice.content, tool_calls=_parse_openai_tool_calls(choice.tool_calls)
        )


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible provider (pure httpx — no extra SDK required)
# ---------------------------------------------------------------------------


class GenericOpenAILLM(LLMProvider):
    """Any OpenAI-compatible ``/chat/completions`` endpoint via plain ``httpx``.

    Powers Groq, OpenRouter, DeepSeek, xAI, Together, Mistral, Gemini (OpenAI
    endpoint), Ollama, LM Studio, vLLM, and arbitrary custom servers — without
    needing the ``openai`` SDK or LiteLLM. The API key is optional (local
    servers often need none); a ``base_url`` is required.
    """

    def __init__(
        self,
        provider_id: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        *,
        supports_tools: bool = True,
        supports_vision: bool = False,
        timeout: float = 60.0,
    ):
        if not base_url:
            raise LLMUnavailable(f"{provider_id}: base_url is required")
        self.name = provider_id
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._supports_tools = supports_tools
        self.supports_vision = supports_vision
        self._timeout = timeout

    def build_request(
        self,
        messages: list[LLMMessage],
        toys: list[ToolSpec],
        images: Optional[list[ImageInput]] = None,
    ) -> dict[str, Any]:
        oai_messages = [_to_openai_message(m) for m in messages]
        if images and self.supports_vision:
            _openai_attach_images(oai_messages, images)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": 0,
        }
        if toys and self._supports_tools:
            body["tools"] = _openai_tools_payload(toys)
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return {
            "url": f"{self._base_url}/chat/completions",
            "headers": headers,
            "json": body,
        }

    @staticmethod
    def parse_response(data: dict[str, Any]) -> LLMResult:
        choices = data.get("choices") or [{}]
        msg = (choices[0] or {}).get("message", {}) or {}
        return LLMResult(
            content=msg.get("content"),
            tool_calls=_parse_openai_tool_calls(msg.get("tool_calls")),
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        toys: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        import time
        import asyncio
        import json
        from urllib import request as urllib_request
        
        log.debug(f"GenericOpenAILLM.complete called with images: {images is not None}")
        if images is not None:
            log.debug(f"Number of images: {len(images)}")
            if len(images) > 0:
                log.debug(f"First image b64 length: {len(images[0].b64) if images[0].b64 else 0}")
        req = self.build_request(messages, toys, images)
        log.debug(f"GenericOpenAILLM request: {req}")
        # Log the full json body
        log.debug(f"GenericOpenAILLM request body: {json.dumps(req['json'], indent=2)}")

        def _fetch():
            # Convert the json dict to bytes
            data_bytes = json.dumps(req["json"]).encode('utf-8')
            # Create a urllib request
            urllib_req = urllib_request.Request(
                url=req["url"],
                data=data_bytes,
                headers=req["headers"],
                method='POST'
            )
            with urllib_request.urlopen(urllib_req, timeout=self._timeout) as response:
                full_content = ""
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                full_content += data["message"]["content"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                return full_content

        try:
            start_time = time.time()
            loop = asyncio.get_event_loop()
            full_content = await loop.run_in_executor(None, _fetch)
            end_time = time.time()
            log.debug(f"GenericOpenAILLM request took {end_time - start_time:.2f} seconds")
            
            # Create a mock response object for parse_response
            data = {
                "choices": [{
                    "message": {
                        "content": full_content
                    }
                }]
            }
            return self.parse_response(data)
        except Exception as exc:
            log.exception("Error calling GenericOpenAILLM via synchronous request")
            log.error(f"Error calling GenericOpenAILLM: {exc}")
            raise LLMUnavailable(f"Request failed: {exc}")


class OllamaLLM(GenericOpenAILLM):
    """Specialized LLM provider for Ollama vision support.
    
    Ollama's vision API uses a different endpoint (/api/chat) and format
    (images in message['images'] array) compared to standard OpenAI format.
    """

    def __init__(
        self,
        provider_id: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        *,
        supports_tools: bool = True,  # Ignored, Ollama does not support tools
        supports_vision: bool = False,
        timeout: float = 300.0,
    ):
        super().__init__(
            provider_id, model, api_key, base_url,
            supports_tools=False,   # Ollama does not support tools via the OpenAI API format
            supports_vision=supports_vision,
            timeout=timeout
        )
        # Override the base URL to remove the /v1 suffix for Ollama's vision endpoint
        if self._base_url.endswith('/v1'):
            self._base_url = self._base_url[:-3]  # Remove '/v1'

    def build_request(
        self,
        messages: list[LLMMessage],
        toys: list[ToolSpec],
        images: Optional[list[ImageInput]] = None,
    ) -> dict[str, Any]:
        log.debug(f"OllamaLLM.build_request: toys={toys}, self._supports_tools={self._supports_tools}")
        # For Ollama, we always use the /api/chat endpoint
        oai_messages = [_to_openai_message(m) for m in messages]
        
        # Handle vision-specific formatting if images are present
        if images and self.supports_vision:
            # Use Ollama's vision format: images in message['images'] array
            # Add images to the last user message (typically where vision requests go)
            for msg in reversed(oai_messages):
                if msg.get("role") == "user":
                    # For Ollama, keep content as string if it was string, 
                    # or convert to string if it's a list (though this shouldn't happen in our usage)
                    if isinstance(msg.get("content"), list):
                        # Extract text from content list
                        text_parts = []
                        for part in msg["content"]:
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        msg["content"] = " ".join(text_parts)
                    # Ensure content is a string
                    if not isinstance(msg.get("content"), str):
                        msg["content"] = str(msg.get("content", ""))
                    
                    # Add images in Ollama's format as a separate field (base64 strings)
                    msg["images"] = []
                    for img in images:
                        msg["images"].append(img.b64)
                    break
        
        body: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": 0,
            "stream": False,
        }
        if toys and self._supports_tools:
            log.debug(f"OllamaLLM.build_request: adding tools to request")
            # Note: Ollama may not support tools in the same way as OpenAI
            # For now, we'll pass them through but this may need adjustment
            body["tools"] = _openai_tools_payload(toys)
            body["tool_choice"] = "auto"
        else:
            log.debug(f"OllamaLLM.build_request: not adding toys")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return {
            "url": f"{self._base_url}/api/chat",  # Ollama's endpoint
            "headers": headers,
            "json": body,
        }


# ---------------------------------------------------------------------------
# LiteLLM universal adapter (optional dep: pip install '.[providers]')
# ---------------------------------------------------------------------------


class LiteLLMProvider(LLMProvider):
    """Universal provider via LiteLLM — one interface to 100+ providers.

    Routes by a litellm model string (e.g. ``gemini/gemini-1.5-flash``,
    ``bedrock/…``, ``azure/…``). Tools + vision use the OpenAI content format
    that LiteLLM normalizes for each backend.
    """

    def __init__(self, resolved, *, timeout: float = 60.0):
        try:
            import litellm  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(
                f"litellm not installed (pip install '.[providers]') ({exc})"
            ) from exc
        if resolved.requires_key and not resolved.api_key:
            raise LLMUnavailable(f"{resolved.env_var} not set")
        self._litellm = litellm
        self.name = resolved.provider_id
        self.model = resolved.litellm_model
        self.supports_vision = resolved.supports_vision
        self._supports_tools = resolved.supports_tools
        self._api_key = resolved.api_key
        self._base_url = resolved.base_url
        self._timeout = timeout

    async def complete(
        self,
        messages: list[LLMMessage],
        toys: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        oai_messages = [_to_openai_message(m) for m in messages]
        if images and self.supports_vision:
            _openai_attach_images(oai_messages, images)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": 0,
            "timeout": self._timeout,
        }
        if toys and self._supports_tools:
            kwargs["tools"] = _openai_tools_payload(toys)
            kwargs["tool_choice"] = "auto"
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        resp = await self._litellm.acompletion(**kwargs)
        # LiteLLM returns an OpenAI-style ModelResponse (object or dict-like).
        try:
            choice = resp.choices[0].message
            content = getattr(choice, "content", None)
            tool_calls = getattr(choice, "tool_calls", None)
        except AttributeError:
            msg = resp["choices"][0]["message"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
        return LLMResult(content=content, tool_calls=_parse_openai_tool_calls(tool_calls))


def _to_openai_message(m: LLMMessage) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "tc": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_use_id or "",
            "content": m.content or "",
        }
    return {"role": m.role, "content": m.content or ""}


def _openai_attach_images(messages: list[dict[str, Any]], images: list[ImageInput]) -> None:
    """Append image_url blocks to the last user message (OpenAI vision)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content") or ""
            blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for img in images:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.media_type};base64,{img.b64}"},
                    }
                )
            msg["content"] = blocks
            return


# ---------------------------------------------------------------------------
# Anthropic provider (optional dep)
# ---------------------------------------------------------------------------


class AnthropicLLM(LLMProvider):
    name = "anthropic"
    supports_vision = True

    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")
        try:
            from anthropic import AsyncAnthropic  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"anthropic SDK not installed ({exc})") from exc
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        *,
        images: Optional[list[ImageInput]] = None,
    ) -> LLMResult:
        client = self._get_client()
        system_text = "\n\n".join(
            m.content for m in messages if m.role == "system" and m.content
        )
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in toys
        ]
        anthropic_messages = _to_anthropic_messages(messages)
        if images:
            _anthropic_attach_images(anthropic_messages, images)
        resp = await client.messages.create(
            model=self.model,
            system=system_text or None,
            messages=anthropic_messages,
            tools=anthropic_tools or None,
            max_tokens=1024,
        )
        content_text: Optional[str] = None
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content_text = (content_text or "") + block.text
            elif btype == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )
        return LLMResult(content=content_text, tool_calls=calls)


def _to_anthropic_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.content or ""}]})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_use_id or "",
                            "content": m.content or "",
                        }
                    ],
                }
            )
    return out


def _anthropic_attach_images(messages: list[dict[str, Any]], images: list[ImageInput]) -> None:
    """Append image blocks to the last user message (Anthropic vision)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if not isinstance(content, list):
                content = [{"type": "text", "text": content or ""}]
            for img in images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img.b64,
                        },
                    }
                )
            msg["content"] = content
            return


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_native(resolved) -> LLMProvider:
    """Build a first-party / httpx provider for a resolved spec, or raise."""
    from .. import providers as P

    if resolved.kind == P.KIND_NATIVE_OPENAI:
        return OpenAILLM(resolved.model, resolved.api_key, resolved.base_url)
    if resolved.kind == P.KIND_NATIVE_ANTHROPIC:
        return AnthropicLLM(resolved.model, resolved.api_key, resolved.base_url)
    if resolved.kind == P.KIND_OPENAI_COMPATIBLE:
        # Special handling for Ollama which uses a different API endpoint and format
        if resolved.provider_id == "ollama":
            return OllamaLLM(
                resolved.provider_id,
                resolved.model,
                resolved.api_key,
                resolved.base_url,
                supports_tools=resolved.supports_tools,
                supports_vision=resolved.supports_vision,
            )
        if resolved.requires_key and not resolved.api_key:
            raise LLMUnavailable(f"{resolved.env_var} not set")
        return GenericOpenAILLM(
            resolved.provider_id,
            resolved.model,
            resolved.api_key,
            resolved.base_url,
            supports_tools=resolved.supports_tools,
            supports_vision=resolved.supports_vision,
        )
    raise LLMUnavailable(f"no native path for kind {resolved.kind!r}")


def create_llm(config) -> LLMProvider:
    """Select a provider from config via the registry; fall back to MockLLM.

    Never crashes on a missing key/SDK — logs a warning and returns MockLLM so
    the offline path always works.
    """
    from .. import providers as P

    resolved = P.resolve(config)
    if resolved.kind == P.KIND_MOCK:
        log.info("LLM provider: mock (deterministic, offline)")
        return MockLLM()

    prefer_litellm = bool(getattr(config, "use_litellm", False))

    def try_native() -> LLMProvider:
        return _build_native(resolved)

    def try_litellm() -> LLMProvider:
        return LiteLLMProvider(resolved)

    # litellm_only providers (Bedrock/Azure/Vertex/Cohere) have no native path.
    if resolved.kind == P.KIND_LITELLM_ONLY:
        order = [try_litellm]
    elif prefer_litellm:
        order = [try_litellm, try_native]
    else:
        order = [try_native, try_litellm]

    errors: list[str] = []
    for builder in order:
        try:
            llm = builder()
            log.info(
                "LLM provider: %s model=%s%s",
                resolved.provider_id,
                llm.model,
                f" base_url={resolved.base_url}" if resolved.base_url else "",
            )
            return llm
        except LLMUnavailable as exc:
            errors.append(str(exc))
            continue

    log.warning(
        "provider %r unavailable (%s); falling back to mock. "
        "Run `jarvis-backend setup` to configure a provider/key, or "
        "`pip install '.[providers]'` for LiteLLM providers.",
        resolved.provider_id,
        "; ".join(errors) or "unknown",
    )
    return MockLLM()


__all__ = [
    "ToolCall",
    "LLMMessage",
    "ToolSpec",
    "ImageInput",
    "LLMResult",
    "LLMProvider",
    "MockLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "GenericOpenAILLM",
    "LiteLLMProvider",
    "OllamaLLM",
    "LLMUnavailable",
    "create_llm",
    "extract_city",
    "extract_duration_seconds",
    "extract_object_name",
    "extract_search_query",
    "extract_destination",
    "extract_symbols",
    "extract_target_lang",
    "plan_tool_calls",
]