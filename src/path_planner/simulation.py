#!/usr/bin/env python3
"""
Path Replanner - Simulation Module

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
import matplotlib.pyplot as plt
from .planner import PathReplanner

class NeedleSteeringCase:
    """
    Simulation Use Case for the path planner.
    Sets up obstacles, start/target poses, and deformation fields.
    """
    def __init__(self, r_b=3.0, delta=0.96, r_min=70.0, tar_tol=2.5, num_path_points=100):
        """
        Initialize simulation with planner parameters.
        
        Args:
            r_b (float): Radius of the bubbles (mm).
            delta (float): Minimum overlapping value between bubbles (mm).
            r_min (float): Minimum radius of curvature for the needle (mm).
            tar_tol (float): Target position tolerance (mm).
            num_path_points (int): Number of points in interpolated final path.
        """
        self.planner = PathReplanner(r_b=r_b, delta=delta, r_min=r_min, tar_tol=tar_tol)
        self.num_path_points = num_path_points
        self.obstacles = [] # List of (center, radius)
        
    def add_spherical_obstacle(self, center, radius):
        self.obstacles.append((np.array(center), radius))
        
    def get_nearest_obstacle(self, point):
        """
        Returns nearest obstacle point and distance.
        Used as the obstacle map query.
        """
        min_dist = float('inf')
        nearest_pt = point
        
        for center, radius in self.obstacles:
            dist_center = np.linalg.norm(point - center)
            dist_surface = dist_center - radius
            
            if dist_surface < min_dist:
                min_dist = dist_surface
                # Point on the surface of the obstacle closest to query
                direction = (point - center) / (dist_center + 1e-6)
                nearest_pt = center + direction * radius
                
        return nearest_pt, min_dist

    def simple_deformation_field(self, point):
        """
        Simulates brain shift[cite: 257]. 
        Magnitude decreases with depth (z-axis).
        """
        # Assume entry at z=0, target at z=100
        # Deformation max at surface, decreases 0.59mm per mm depth [cite: 259]
        depth = point[2]
        max_shift = 6.0 # mm
        decay = 0.05 
        
        magnitude = max(0, max_shift - decay * depth)
        
        # Shift in Y direction (gravity simulation)
        return np.array([0, magnitude, 0])

    def _setup_random_scenario(self, start_pose=None, target_pose=None, num_obstacles=5, seed=None):
        """Random obstacles scenario for varied testing."""
        if seed is not None:
            np.random.seed(seed)
        
        if start_pose is None:
            start_pos = np.array([0.0, 0.0, 0.0])
            start_dir = np.array([0.0, 0.0, 1.0])
        else:
            start_pos, start_dir = start_pose
        
        if target_pose is None:
            target_pos = np.array([12.0, 6.0, 85.0])
            target_dir = np.array([0.0, 0.0, 1.0])
        else:
            target_pos, target_dir = target_pose
        
        # Clear and generate random obstacles
        self.obstacles = []
        path_length = np.linalg.norm(target_pos - start_pos)
        for _ in range(num_obstacles):
            # Random position along rough path direction, with some spread
            t = np.random.uniform(0.2, 0.8)  # Avoid start/end
            center = start_pos + t * (target_pos - start_pos)
            # Add random offset
            center += np.random.uniform(-path_length*0.15, path_length*0.15, 3)
            radius = np.random.uniform(5.0, 12.0)
            self.add_spherical_obstacle(center, radius)
        
        # Initial path (straight line)
        num_init_points = 20 + num_obstacles * 2
        initial_path = np.linspace(start_pos, target_pos, num_init_points)
        
        return {
            'initial_path': initial_path,
            'tip_pose': (start_pos, start_dir),
            'target_pose': (target_pos, target_dir)
        }

    def run_simulation(self, scenario="simple", start_pose=None, target_pose=None, num_obstacles=5, seed=None, max_attempts=3, obstacle_radius=6.0):
        """
        Run the simulation with a specified scenario.
        
        Args:
            scenario: "simple", "complex", or "random"
            start_pose: Optional (position, direction) tuple for starting pose
            target_pose: Optional (position, direction) tuple for target pose
            num_obstacles: Number of obstacles for random scenario
            seed: Random seed for reproducible random scenarios
            max_attempts: Maximum number of retry attempts for collision-free path
            obstacle_radius: Radius of obstacles in mm
        """
        if scenario == "simple":
            config = self._setup_simple_scenario(start_pose, target_pose, obstacle_radius)
        elif scenario == "complex":
            config = self._setup_complex_scenario(start_pose, target_pose, obstacle_radius)
        elif scenario == "random":
            config = self._setup_random_scenario(start_pose, target_pose, num_obstacles, seed, obstacle_radius)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Run planner with scenario configuration
        print(f"Running Path Replanner with '{scenario}' scenario...")
        print(f"Planner parameters: r_b={self.planner.r_b:.1f}mm, r_min={self.planner.r_min:.1f}mm, delta={self.planner.delta:.2f}mm, tar_tol={self.planner.tar_tol:.1f}mm")
        print(f"Obstacles: {len(self.obstacles)} obstacles, radius={obstacle_radius:.1f}mm")
        print(f"Max attempts: {max_attempts}")
        attempts_info = None
        try:
            final_path, bubbles, attempts_info = self.planner.plan(
                config['initial_path'],
                config['tip_pose'],
                config['target_pose'],
                self.get_nearest_obstacle,
                self.simple_deformation_field,
                max_attempts=max_attempts
            )
            print("Planning Complete.")
        except Exception as e:
            if "NO FEASIBLE PATH" in str(e) or type(e).__name__ == "PathPlanningError":
                print(str(e))
                # Get attempts info from exception if available
                if hasattr(e, 'attempts_info'):
                    attempts_info = e.attempts_info
                    print(f"\n📊 Visualizing all {len(attempts_info)} failed attempts...")
                    self.plot_failed_attempts(config['initial_path'], attempts_info, config['target_pose'][0])
                else:
                    print("\n💡 Suggestions:")
                    print("  - Try a different scenario")
                    print("  - Adjust planner parameters (r_b, r_min, tar_tol)")
                    print("  - Modify obstacle positions or sizes")
                return  # Exit without standard visualization
            else:
                raise  # Re-raise unexpected errors
        
        # Visualization for successful path
        self.plot_results(config['initial_path'], bubbles, final_path, config['tip_pose'], config['target_pose'], attempts_info)
    
    def _setup_simple_scenario(self, start_pose=None, target_pose=None, obstacle_radius=6.0):
        """Single obstacle scenario - original test case."""
        # Clear previous obstacles
        self.obstacles = []
        if start_pose is None:
            start_pos = np.array([0.0, 0.0, 0.0])
            start_dir = np.array([0.0, 0.0, 1.0])
        else:
            start_pos, start_dir = start_pose
        
        if target_pose is None:
            target_pos = np.array([10.0, 5.0, 80.0])
            target_dir = np.array([0.0, 0.0, 1.0])
        else:
            target_pos, target_dir = target_pose
        
        # Clear and add obstacle
        self.obstacles = []
        self.add_spherical_obstacle([5.0, 2.0, 40.0], obstacle_radius)
        
        # Initial path (straight line)
        num_init_points = 20
        initial_path = np.linspace(start_pos, target_pos, num_init_points)
        
        return {
            'initial_path': initial_path,
            'tip_pose': (start_pos, start_dir),
            'target_pose': (target_pos, target_dir)
        }
    
    def _setup_complex_scenario(self, start_pose=None, target_pose=None, obstacle_radius=6.0):
        """Multiple obstacles scenario - more challenging path planning."""
        if start_pose is None:
            start_pos = np.array([0.0, 0.0, 0.0])
            start_dir = np.array([0.0, 0.0, 1.0])
        else:
            start_pos, start_dir = start_pose
        
        if target_pose is None:
            target_pos = np.array([15.0, -8.0, 90.0])
            target_dir = np.array([0.0, 0.0, 1.0])
        else:
            target_pos, target_dir = target_pose
        
        # Clear and add multiple obstacles
        self.obstacles = []
        self.add_spherical_obstacle([5.0, 2.0, 30.0], obstacle_radius * 0.8)   # First obstacle (smaller)
        self.add_spherical_obstacle([10.0, -3.0, 55.0], obstacle_radius * 1.2)  # Second obstacle (larger)
        self.add_spherical_obstacle([8.0, -6.0, 75.0], obstacle_radius * 0.7)   # Third obstacle (smaller)
        
        # Initial path (straight line)
        num_init_points = 25
        initial_path = np.linspace(start_pos, target_pos, num_init_points)
        
        return {
            'initial_path': initial_path,
            'tip_pose': (start_pos, start_dir),
            'target_pose': (target_pos, target_dir)
        }

    def _setup_random_scenario(self, start_pose=None, target_pose=None, num_obstacles=5, seed=None, obstacle_radius=6.0):
        """Random scenario with configurable obstacles."""
        # Clear previous obstacles
        self.obstacles = []
        
        if start_pose is None:
            start_pos = np.array([0.0, 0.0, 0.0])
            start_dir = np.array([0.0, 0.0, 1.0])
        else:
            start_pos, start_dir = start_pose
        
        if target_pose is None:
            target_pos = np.array([15.0, 5.0, 85.0])
            target_dir = np.array([0.0, 0.0, 1.0])
        else:
            target_pos, target_dir = target_pose
        
        # Generate random obstacles
        rng = np.random.default_rng(seed)
        
        for i in range(num_obstacles):
            # Place obstacles along the path with some randomness
            z_pos = rng.uniform(20, 70)
            x_pos = rng.uniform(-5, 15)
            y_pos = rng.uniform(-5, 10)
            obs_center = [x_pos, y_pos, z_pos]
            
            # Vary radius around the base value (±30%)
            obs_radius = obstacle_radius * rng.uniform(0.7, 1.3)
            self.add_spherical_obstacle(obs_center, obs_radius)
        
        # Initial path (straight line)
        num_init_points = 25
        initial_path = np.linspace(start_pos, target_pos, num_init_points)
        
        return {
            'initial_path': initial_path,
            'tip_pose': (start_pos, start_dir),
            'target_pose': (target_pos, target_dir)
        }

    def plot_results(self, initial, bubbles, final, start_pose, target_pose, attempts_info=None):
        """
        Plot the results of path planning.
        
        Args:
            start_pose: (position, direction) tuple for starting pose
            target_pose: (position, direction) tuple for target pose  
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Obstacles
        for center, radius in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius*np.cos(u)*np.sin(v)
            y = center[1] + radius*np.sin(u)*np.sin(v)
            z = center[2] + radius*np.cos(v)
            ax.plot_wireframe(x, y, z, color='r', alpha=0.3, label='Obstacle')

        # Plot Paths
        ax.plot(initial[:,0], initial[:,1], initial[:,2], 'k--', alpha=0.5, label='Preoperative Path')
        
        # Plot Starting Pose with full orientation axes
        start_pos, start_dir = start_pose
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='g', marker='s', s=150, 
                   edgecolors='black', linewidths=2, label='Starting Pose', zorder=10)
        
        # Draw coordinate frame for start pose (full XYZ)
        self._draw_coordinate_frame(ax, start_pos, start_dir, scale=2.0, label_prefix='Start')
        
        # Plot Bubbles (Centers)
        ax.scatter(bubbles[:,0], bubbles[:,1], bubbles[:,2], c='cyan', marker='o', alpha=0.3, s=50, label='Bubble Centers')
        
        # Plot Final Smooth Path
        ax.plot(final[:,0], final[:,1], final[:,2], 'b-', linewidth=3, label='✓ Replanned Path', zorder=5)
        
        # Plot Target Pose with X-axis only (magenta)
        target_pos, target_dir = target_pose
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='m', marker='*', s=200, label='Target', zorder=10)
        
        # Draw only X-axis for target pose (magenta)
        self._draw_single_x_axis(ax, target_pos, target_dir, scale=2.0, color='magenta', label='Target X-axis')
        
        # Draw X-axis for final path pose (blue)
        # Compute direction from last two points of path
        if len(final) >= 2:
            final_pos = final[-1]
            # Direction vector from second-to-last to last point
            final_dir = final[-1] - final[-2]
            final_dir = final_dir / np.linalg.norm(final_dir)
            self._draw_single_x_axis(ax, final_pos, final_dir, scale=2.0, color='blue', label='Path End X-axis')

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.legend()
        plt.title('Path Replanning - Successful')
        plt.show()
    
    def _draw_coordinate_frame(self, ax, position, z_direction, scale=5.0, label_prefix=''):
        """
        Draw a 3D coordinate frame (XYZ axes) at a given position with Z-axis aligned to direction.
        
        Args:
            ax: matplotlib 3D axis
            position: np.array([x, y, z]) - origin of the frame
            z_direction: np.array([dx, dy, dz]) - direction for Z-axis (will be normalized)
            scale: length of the axis arrows
            label_prefix: prefix for axis labels (e.g., 'Start' or 'Target')
        """
        # Normalize Z direction
        z_axis = np.array(z_direction) / np.linalg.norm(z_direction)
        
        # Find perpendicular X axis
        # Choose a vector not parallel to z_axis
        if abs(z_axis[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y axis completes the right-handed frame
        y_axis = np.cross(z_axis, x_axis)
        
        # Draw axes as arrows
        # X-axis in red
        ax.quiver(position[0], position[1], position[2],
                 x_axis[0], x_axis[1], x_axis[2],
                 length=scale, color='red', arrow_length_ratio=0.3, linewidth=2)
        
        # Y-axis in green
        ax.quiver(position[0], position[1], position[2],
                 y_axis[0], y_axis[1], y_axis[2],
                 length=scale, color='green', arrow_length_ratio=0.3, linewidth=2)
        
        # Z-axis in blue
        ax.quiver(position[0], position[1], position[2],
                 z_axis[0], z_axis[1], z_axis[2],
                 length=scale, color='blue', arrow_length_ratio=0.3, linewidth=2)
    
    def _draw_single_x_axis(self, ax, position, z_direction, scale=5.0, color='red', label='X-axis'):
        """
        Draw only the X-axis at a given position with Z-axis aligned to direction.
        
        Args:
            ax: matplotlib 3D axis
            position: np.array([x, y, z]) - origin of the frame
            z_direction: np.array([dx, dy, dz]) - direction for Z-axis (will be normalized)
            scale: length of the axis arrow
            color: color of the X-axis arrow
            label: label for the axis
        """
        # Normalize Z direction
        z_axis = np.array(z_direction) / np.linalg.norm(z_direction)
        
        # Find perpendicular X axis
        # Choose a vector not parallel to z_axis
        if abs(z_axis[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Draw only X-axis as arrow
        ax.quiver(position[0], position[1], position[2],
                 x_axis[0], x_axis[1], x_axis[2],
                 length=scale, color=color, arrow_length_ratio=0.3, linewidth=2.5)


    def plot_failed_attempts(self, initial, attempts_info, target):
        """Visualize all failed path attempts with different colors."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Obstacles
        for center, radius in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius*np.cos(u)*np.sin(v)
            y = center[1] + radius*np.sin(u)*np.sin(v)
            z = center[2] + radius*np.cos(v)
            ax.plot_wireframe(x, y, z, color='r', alpha=0.4)

        # Color scheme for attempts
        colors = ['orange', 'yellow', 'cyan', 'magenta', 'lime', 'pink', 'brown', 'gray']
        
        # Plot initial path
        ax.plot(initial[:,0], initial[:,1], initial[:,2], 'k--', alpha=0.4, linewidth=1, label='Initial Path')
        
        # Plot Starting Pose
        ax.scatter(initial[0,0], initial[0,1], initial[0,2], c='g', marker='s', s=150, 
                   edgecolors='black', linewidths=2, label='Start', zorder=10)
        
        # Plot each attempt
        for info in attempts_info:
            attempt_num = info['attempt']
            path = info['path']
            collisions = info['collision_indices']
            color = colors[attempt_num % len(colors)]
            
            # Plot path
            label = f"Attempt {attempt_num + 1} ({info['num_collisions']} collisions, k_ext={info['k_ext']:.1f})"
            ax.plot(path[:,0], path[:,1], path[:,2], color=color, linewidth=2, 
                   alpha=0.7, label=label)
            
            # Highlight collision points
            if len(collisions) > 0:
                collision_pts = path[collisions]
                ax.scatter(collision_pts[:,0], collision_pts[:,1], collision_pts[:,2], 
                          c='red', marker='x', s=80, linewidths=3, zorder=8)
        
        # Plot Target
        ax.scatter(target[0], target[1], target[2], c='m', marker='*', s=300, 
                  label='Target', zorder=10, edgecolors='black', linewidths=1.5)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        
        # Legend with collision markers explanation
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        # Add custom entry for collision markers
        handles.append(Line2D([0], [0], marker='x', color='red', linestyle='None', 
                            markersize=8, markeredgewidth=2, label='Collision Points'))
        handles.append(Line2D([0], [0], color='r', linestyle='-', alpha=0.4, label='Obstacles'))
        
        ax.legend(handles=handles, loc='upper left', fontsize=9)
        plt.title(f'Path Replanning - All Failed Attempts ({len(attempts_info)} total)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main(scenario="simple", r_b=3.0, delta=0.96, r_min=70.0, tar_tol=2.5, 
         start_pose=None, target_pose=None, num_obstacles=5, seed=None, max_attempts=3, 
         obstacle_radius=6.0, num_path_points=100):
    """
    Run the path replanning simulation.
    
    Args:
        scenario: Scenario name ("simple", "complex", or "random")
        r_b: Radius of the bubbles (mm)
        delta: Minimum overlapping value between bubbles (mm)
        r_min: Minimum radius of curvature for the needle (mm)
        tar_tol: Target position tolerance (mm)
        start_pose: Optional (position_array, direction_array) for starting pose
        target_pose: Optional (position_array, direction_array) for target pose
        num_obstacles: Number of obstacles for random scenario
        seed: Random seed for reproducible random scenarios
        max_attempts: Maximum number of retry attempts if path has collisions
        obstacle_radius: Radius of obstacles in mm
        num_path_points: Number of points in interpolated final path
    """
    sim = NeedleSteeringCase(r_b=r_b, delta=delta, r_min=r_min, tar_tol=tar_tol, num_path_points=num_path_points)
    sim.run_simulation(scenario=scenario, start_pose=start_pose, target_pose=target_pose,
                       num_obstacles=num_obstacles, seed=seed, max_attempts=max_attempts,
                       obstacle_radius=obstacle_radius)

if __name__ == "__main__":
    main()