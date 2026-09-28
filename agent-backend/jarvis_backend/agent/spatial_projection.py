"""Spatial Projection Engine for JarvisVR.
Handles the intersection of vision rays with the room's semantic mesh.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Ray:
    origin: list[float]
    direction: list[float]

@dataclass
class Plane:
    center: list[float]
    normal: list[float]
    id: str
    label: str

def intersect_ray_plane(ray: Ray, plane: Plane) -> Optional[float]:
    """Returns the distance t from ray origin to plane intersection, or None."""
    # dot(normal, direction)
    denom = sum(ray.direction[i] * plane.normal[i] for i in range(3))
    
    # If denom is near 0, ray is parallel to plane
    if abs(denom) < 1e-6:
        return None
    
    # t = dot(plane_center - ray_origin, normal) / dot(ray_direction, normal)
    num = sum((plane.center[i] - ray.origin[i]) * plane.normal[i] for i in range(3))
    t = num / denom
    
    return t if t > 0 else None

def project_vision_to_world(
    head_pose: dict, 
    gaze_direction: list[float], 
    surfaces: list[dict]
) -> Optional[list[float]]:
    """
    Projects a gaze ray into the room's semantic surfaces to find the 3D world position.
    """
    origin = head_pose.get("position", [0.0, 0.0, 0.0])
    
    ray = Ray(origin=origin, direction=gaze_direction)
    
    best_t = float('inf')
    best_pos = None
    
    for s in surfaces:
        plane = Plane(
            center=s.get("center", [0.0, 0.0, 0.0]),
            normal=s.get("normal", [0.0, 1.0, 0.0]),
            id=s.get("id", "unknown"),
            label=s.get("type", "surface")
        )
        
        t = intersect_ray_plane(ray, plane)
        if t and t < best_t:
            best_t = t
            # pos = origin + t * direction
            best_pos = [origin[i] + t * ray.direction[i] for i in range(3)]
            
    return best_pos
