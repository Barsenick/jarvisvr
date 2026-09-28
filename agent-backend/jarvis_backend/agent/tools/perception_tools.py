from __future__ import annotations

import math
import json
from typing import Any, Optional

from ...perception.vision import (
    describe_scene,
    focus_object,
    mock_ocr,
    mock_translate,
    scene_objects,
)
from .base import SpawnDirective, ToolContext, ToolRegistry, ToolResult
from jarvis_backend.agent.spatial_memory import spatial_db
from jarvis_backend.agent.spatial_projection import project_vision_to_world

_HEAD = [0.0, 1.6, 0.0]

def _unit_vector(frm: list[float], to: list[float]) -> list[float]:
    v = [to[i] - frm[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [round(c / n, 4) for c in v]


def _annotation_transform(obj: dict[str, Any]) -> dict[str, Any]:
    pos = obj.get("position")
    anchor = obj.get("anchor", "world")
    if isinstance(pos, list) and len(pos) == 3:
        return {"anchor": anchor, "position": [pos[0], pos[1] + 0.15, pos[2]], "billboard": True}
    # Fallback: a comfortable spot ~0.8m in front of the user's head.
    return {"anchor": "head", "position": [0.0, 0.0, 0.8], "billboard": True}


def _remember_seen(ctx: ToolContext, objs: list[dict[str, Any]]) -> None:
    if not ctx.episodic:
        return
    for o in objs:
        if o.get("label"):
            ctx.episodic.remember_object(
                o["label"],
                position=o.get("position"),
                anchor=o.get("anchor", "world"),
                source="vision",
                confidence=o.get("confidence"),
            )


def _describe_view(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cd = ctx.perception.current_context()
    desc = describe_scene(cd)
    objs = desc["objects"][:3]
    directives = []
    annotations = []
    for o in objs:
        t = _annotation_transform(o)
        directives.append(
            SpawnDirective(
                widget_type="vision_annotation",
                props={"label": o.get("label", "object"), "confidence": float(o.get("confidence", 0.7))},
                transform=t,
                interactions=["tap", "grab"],
            )
        )
        annotations.append({"label": o.get("label", "object"), "position": t["position"], "anchor": t["anchor"]})
    _remember_seen(ctx, objs)
    if ctx.episodic:
        ctx.episodic.record_event("vision", desc["text"], anchor="world")
    return ToolResult(
        data={
            "speech": desc["text"],
            "objects": [o.get("label") for o in objs],
            "observation": {"text": desc["text"], "annotations": annotations},
        },
        directives=directives,
    )


def _identify_object(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cd = ctx.perception.current_context()
    obj = focus_object(cd)
    if not obj:
        msg = "I don't see anything I can identify right now."
        return ToolResult(data={"speech": msg, "observation": {"text": msg, "annotations": []}})
    label = obj.get("label", "object")
    conf = float(obj.get("confidence", 0.75))
    t = _annotation_transform(obj)
    text = f"That looks like a {label}."
    _remember_seen(ctx, [obj])
    if ctx.episodic:
        ctx.episodic.record_event("vision", f"Identified {label}.", anchor=t["anchor"])
    return ToolResult(
        data={
            "speech": text,
            "object": label,
            "confidence": conf,
            "observation": {"text": text, "annotations": [{"label": label, "position": t["position"], "anchor": t["anchor"]}]},
        },
        directives=[
            SpawnDirective(
                widget_type="vision_annotation",
                props={"label": label, "confidence": conf},
                transform=t,
                interactions=["tap", "grab"],
            )
        ],
    )


def _read_text(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cd = ctx.perception.current_context()
    text = mock_ocr(cd)
    speech = f"It reads: {text}"
    if ctx.episodic:
        ctx.episodic.record_event("ocr", f"Read: {text}")
    return ToolResult(
        data={"speech": speech, "text": text, "observation": {"text": speech, "annotations": []}},
        directives=[
            SpawnDirective(
                widget_type="panel",
                props={"title": "Read", "body": text},
                ref="ocr_panel",
                interactions=["tap", "grab", "resize"],
            )
        ],
    )


def _translate_text(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = (args.get("text") or "").strip()
    target = args.get("target_lang") or "spanish"
    if not text:
        return ToolResult(data={"speech": "What would you like me to translate?"})
    translated = mock_translate(text, target)
    speech = f"In {target}, that's: {translated}"
    return ToolResult(
        data={"speech": speech, "source_text": text, "translated": translated, "target_lang": target},
        directives=[
            SpawnDirective(
                widget_type="translator",
                props={
                    "source_lang": "auto",
                    "target_lang": target,
                    "source_text": text,
                    "translated_text": translated,
                    "mode": "text",
                },
                ref="translator",
                interactions=["tap", "grab"],
            )
        ],
    )


def _translate_view(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cd = ctx.perception.current_context()
    target = args.get("target_lang") or "spanish"
    text = mock_ocr(cd)
    translated = mock_translate(text, target)
    speech = f"The sign reads: {text}. In {target}: {translated}"
    if ctx.episodic:
        ctx.episodic.record_event("translate", f"{text} -> {translated} ({target})")
    return ToolResult(
        data={
            "speech": speech,
            "source_text": text,
            "translated": translated,
            "target_lang": target,
            "observation": {"text": speech, "annotations": []},
        },
        directives=[
            SpawnDirective(
                widget_type="translator",
                props={
                    "source_lang": "auto",
                    "target_lang": target,
                    "source_text": text,
                    "translated_text": translated,
                    "mode": "sign",
                },
                ref="translator",
                interactions=["tap", "grab"],
            )
        ],
    )


def _resolve_position(ctx: ToolContext, args: dict[str, Any]) -> tuple[Optional[list], str]:
    position = args.get("position")
    anchor = args.get("anchor", "world")
    if position is None:
        cd = ctx.perception.current_context()
        # Use spatial projection if we have a room mesh and gaze data
        head_pose = cd.get("head_pose")
        gaze_direction = cd.get("gaze", {}).get("direction")
        surfaces = cd.get("surfaces", [])
        
        if head_pose and gaze_direction and surfaces:
            projected_pos = project_vision_to_world(head_pose, gaze_direction, surfaces)
            if projected_pos:
                return projected_pos, "world"
        
        # Fallback to simple hit point if available
        gaze = cd.get("gaze") or {}
        if isinstance(gaze.get("hit_point"), list):
            return gaze["hit_point"], "world"
        
        fo = focus_object(cd)
        if fo and isinstance(fo.get("position"), list):
            return fo["position"], fo.get("anchor", "world")
    return position, anchor


def _remember_object(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = (args.get("name") or "it").strip() or "it"
    position, anchor = _resolve_position(ctx, args)
    
    # Save to long-term spatial memory
    if position:
        spatial_db.remember(
            name=name,
            position=position,
            anchor=anchor,
            source="user",
            timestamp=ctx.session.timestamp if hasattr(ctx.session, 'timestamp') else 0.0
        )
        
    if ctx.episodic:
        ctx.episodic.remember_object(name, position=position, anchor=anchor, source="user")
        ctx.episodic.record_event("memory", f"Remembered {name} location.", anchor=anchor)
    directives = []
    where = ""
    if isinstance(position, list) and len(position) == 3:
        directives.append(
            SpawnDirective(
                widget_type="vision_annotation",
                props={"label": name, "detail": "remembered"},
                transform={"anchor": anchor, "position": [position[0], position[1] + 0.1, position[2]], "billboard": True},
                interactions=["tap"],
            )
        )
        where = " — I marked the spot"
    speech = f"Got it. I'll remember where your {name} is{where}."
    return ToolResult(data={"speech": speech, "remembered": name, "position": position}, directives=directives)


def _find_object(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = (args.get("name") or "it").strip() or "it"
    
    # Check long-term spatial memory first
    rec = spatial_db.recall(name)
    
    if not rec:
        # Fallback to episodic memory
        if ctx.episodic:
            rec_dict = ctx.episodic.recall_object(name)
            if rec_dict:
                rec = SpatialRecord(
                    name=rec_dict.get("name", name),
                    position=rec_dict.get("position"),
                    anchor=rec_dict.get("anchor", "world"),
                    source="episodic",
                    confidence=rec_dict.get("confidence", 0.5),
                    timestamp=0.0
                )
    
    if not rec:
        msg = f"I haven't seen your {name} recently."
        return ToolResult(data={"found": False, "speech": msg, "observation": {"text": msg, "annotations": []}})
        
    position = rec.position
    anchor = rec.anchor
    directives = []
    annotations = []
    if isinstance(position, list) and len(position) == 3:
        t = {"anchor": anchor, "position": [position[0], position[1] + 0.1, position[2]], "billboard": True}
        directives.append(
            SpawnDirective(
                widget_type="vision_annotation",
                props={"label": rec.name, "detail": "last seen here"},
                transform=t,
                interactions=["tap"],
            )
        )
        distance = math.sqrt(sum((position[i] - _HEAD[i]) ** 2 for i in range(3)))
        directives.append(
            SpawnDirective(
                widget_type="navigation_arrow",
                props={
                    "target_label": rec.name,
                    "direction": _unit_vector(_HEAD, position),
                    "distance_m": round(distance, 2),
                },
                interactions=["tap"],
            )
        )
        annotations.append({"label": rec.name, "position": t["position"], "anchor": anchor})
        speech = f"Your {name} should be right here — I've marked the spot."
    else:
        speech = f"I remember your {name}, but I didn't note exactly where."
    return ToolResult(
        data={"found": True, "speech": speech, "location": position, "observation": {"text": speech, "annotations": annotations}},
        directives=directives,
    )


def _identify_sound(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    evt = ctx.perception.latest_audio_event()
    if not evt:
        msg = "I haven't heard any notable sounds recently."
        return ToolResult(data={"speech": msg, "observation": {"text": msg, "annotations": []}})
    label = evt.get("label", "a sound")
    speech = f"That sounded like {label}."
    if ctx.episodic:
        ctx.episodic.record_event("audio", f"Heard {label}.")
    return ToolResult(
        data={"speech": speech, "sound": label, "confidence": evt.get("confidence"), "observation": {"text": speech, "annotations": []}},
        directives=[
            SpawnDirective(
                widget_type="live_caption",
                props={"lines": [f"Heard: {label}"], "speaker": "other"},
                ref="live_caption",
                interactions=["grab", "tap"],
            )
        ],
    )


def _measure(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cd = ctx.perception.current_context()
    objs = [o for o in scene_objects(cd) if isinstance(o.get("position"), list) and len(o["position"]) == 3]
    if len(objs) >= 2:
        a, b = objs[0]["position"], objs[1]["position"]
        label = f"{objs[0].get('label', 'a')} \u2194 {objs[1].get('label', 'b')}"
    else:
        a, b, label = [0.0, 1.0, 0.5], [0.0, 1.0, 1.2], "span in front of you"
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    speech = f"That's about {dist:.2f} meters ({label})."
    return ToolResult(
        data={"speech": speech, "distance_m": round(dist, 2), "from": a, "to": b},
        directives=[
            SpawnDirective(
                widget_type="measuring_tape",
                props={
                    "points": [list(a), list(b)],
                    "distance_m": round(dist, 2),
                    "label": label,
                    "unit": "m",
                    "mode": "distance",
                },
                ref="measure",
                interactions=["grab", "tap"],
            )
        ],
    )


def register_perception_tools(reg: ToolRegistry) -> None:
    reg.add(
        "describe_view",
        "Describe what the user is currently looking at (passthrough camera); "
        "spawns vision_annotation labels on the real objects.",
        {"type": "object", "properties": {}},
        _describe_view,
    )
    reg.add(
        "identify_object",
        "Identify the specific object the user is looking at / pointing at "
        "('what is this?'); spawns a vision_annotation on it.",
        {"type": "object", "properties": {}},
        _identify_object,
    )
    reg.add(
        "read_text",
        "Read (OCR) text visible in the current view and show it on a panel.",
        {"type": "object", "properties": {}},
        _read_text,
    )
    reg.add(
        "translate_text",
        "Translate a piece of text into a target language.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string", "description": "e.g. 'spanish'."},
            },
            "required": ["text"],
        },
        _translate_text,
    )
    reg.add(
        "translate_view",
        "Read text in the current view and translate it (sign/menu/label).",
        {"type": "object", "properties": {"target_lang": {"type": "string"}}},
        _translate_view,
    )
    reg.add(
        "remember_object",
        "Remember where an object is for later recall (spatial memory).",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "position": {"type": "array", "items": {"type": "number"}},
                "anchor": {"type": "string"},
            },
            "required": ["name"],
        },
        _remember_object,
    )
    reg.add(
        "find_object",
        "Recall where an object was last seen ('where did I leave my keys?'); "
        "spawns a marker + navigation arrow.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        _find_object,
    )
    reg.add(
        "identify_sound",
        "Tell the user what the most recent ambient sound was.",
        {"type": "object", "properties": {}},
        _identify_sound,
    )
    reg.add(
        "measure",
        "Measure the distance between detected objects / points in the room.",
        {"type": "object", "properties": {}},
        _measure,
    )


__all__ = ["register_perception_tools"]

















