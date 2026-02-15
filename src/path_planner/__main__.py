#!/usr/bin/env python3
"""
Path Replanner - Entry Point

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

Entry point for the path planner package.
Allows running scenarios with configurable parameters.
"""
from .simulation import main
from .utils import quaternion_to_direction
import sys
import numpy as np

def run_main():
    """Console script entry point - calls main with command line args."""
    # Parse simple command-line arguments
    scenario = "random"
    
    # Check for scenario argument
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        if scenario not in ["simple", "complex", "random"]:
            print(f"Unknown scenario: {scenario}")
            print("Available scenarios: simple, complex, random")
            sys.exit(1)
    
    # Define orientations as quaternions [w, x, y, z]
    start_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity -> +z direction
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around x -> -z direction
    
    # Convert to direction vectors
    start_dir = quaternion_to_direction(start_quat)
    target_dir = quaternion_to_direction(target_quat)
    
    # Run with configured parameters
    # To customize, modify the values below:
    main(
        scenario=scenario,
        r_b=5.0,              # Bubble radius (mm)
        delta=0.96,           # Minimum bubble overlap (mm)
        r_min=10.0,           # Minimum needle curvature radius (mm)
        tar_tol=0.5,          # Target position tolerance (mm)
        # Pose configuration with quaternions
        start_pose=(np.array([0, 0, 0]), start_dir),
        target_pose=(np.array([15, 5, 80]), target_dir),
        # Obstacle configuration
        num_obstacles=4,      # Number of random obstacles
        seed=42,              # Random seed (None for random)
        obstacle_radius=3.0,  # Radius of obstacles (mm)
        # Path configuration
        num_path_points=150,  # Number of points in final interpolated path
        # Retry configuration
        max_attempts=10       # Maximum retry attempts
    )

if __name__ == "__main__":
    run_main()
