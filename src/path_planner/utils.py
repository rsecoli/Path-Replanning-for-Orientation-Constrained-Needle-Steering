#!/usr/bin/env python3
"""
Path Replanner - Utilities Module

Copyright 2026 Riccardo Secoli
LinkedIn: www.linkedin.com/in/riccardosecoli

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

def quaternion_to_direction(quaternion):
    """
    Convert a quaternion to a 3D direction vector.
    
    Quaternion represents a rotation, and we extract the direction
    by rotating the default [0, 0, 1] vector by the quaternion.
    
    Args:
        quaternion: np.array([w, x, y, z]) or np.array([x, y, z, w])
                    Assumes [w, x, y, z] format (scalar first)
    
    Returns:
        direction: np.array([dx, dy, dz]) - normalized direction vector
    """
    # Normalize quaternion
    q = np.array(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    
    w, x, y, z = q
    
    # Rotate the [0, 0, 1] vector by the quaternion
    # Using formula: v' = q * v * q_conjugate
    # Simplified for rotating [0, 0, 1]
    direction = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y)
    ])
    
    # Normalize
    direction = direction / np.linalg.norm(direction)
    
    return direction

def direction_to_quaternion(direction):
    """
    Convert a 3D direction vector to a quaternion.
    
    Finds the quaternion that rotates [0, 0, 1] to the given direction.
    
    Args:
        direction: np.array([dx, dy, dz]) - direction vector
    
    Returns:
        quaternion: np.array([w, x, y, z]) - unit quaternion
    """
    # Normalize direction
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)
    
    # Default direction is [0, 0, 1]
    default = np.array([0.0, 0.0, 1.0])
    
    # Calculate rotation axis (cross product)
    axis = np.cross(default, d)
    axis_length = np.linalg.norm(axis)
    
    # Calculate rotation angle
    dot = np.dot(default, d)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Handle special cases
    if axis_length < 1e-6:
        if dot > 0:
            # Same direction - identity quaternion
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Opposite direction - 180° rotation around any perpendicular axis
            # Choose x-axis if z is not aligned with it
            if abs(d[0]) < 0.9:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 1.0, 0.0])
            return np.array([0.0, axis[0], axis[1], axis[2]])
    
    # Normalize axis
    axis = axis / axis_length
    
    # Create quaternion from axis-angle
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])
