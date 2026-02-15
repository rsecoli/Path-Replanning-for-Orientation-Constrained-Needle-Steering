#!/usr/bin/env python3
"""
Path Replanner - Planner Module

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
import cvxpy as cp
from scipy.interpolate import interp1d


class PathPlanningError(Exception):
    """Raised when path planner cannot find a collision-free path."""
    pass


class PathReplanner:
    """
    Implements the Extended Bubble Bending and Constrained Smoothing algorithm
    for needle steering as described in Pinzi et al. (2021).
    """
    def __init__(self, r_b=3.0, delta=0.96, r_min=70.0, tar_tol=2.5):
        """
        Initialize planner parameters[cite: 264].
        
        Args:
            r_b (float): Radius of the bubbles (mm).
            delta (float): Minimum overlapping value between bubbles (mm).
            r_min (float): Minimum radius of curvature for the needle (mm).
            tar_tol (float): Target position tolerance (mm).
        """
        self.r_b = r_b
        self.delta = delta
        self.r_min = r_min
        self.tar_tol = tar_tol
        # Calculated channel radius Rc [cite: 80]
        self.r_c = np.sqrt(r_b**2 - (r_b - delta/2)**2)

    def bubble_reorganization(self, path):
        """
        Ensures bubbles overlap by Delta. Inserts/removes bubbles as needed[cite: 116].
        """
        if len(path) < 2:
            return path
            
        new_path = [path[0]]
        for i in range(1, len(path)):
            prev_pt = new_path[-1]
            curr_pt = path[i]
            dist = np.linalg.norm(curr_pt - prev_pt)
            
            # If gap is too large (overlap too small), insert points
            max_dist = 2 * self.r_b - self.delta
            if dist > max_dist:
                num_inserts = int(np.ceil(dist / max_dist))
                for k in range(1, num_inserts + 1):
                    interp_pt = prev_pt + (curr_pt - prev_pt) * (k / num_inserts)
                    new_path.append(interp_pt)
            else:
                new_path.append(curr_pt)
                
        return np.array(new_path)

    def apply_deformation(self, path, deformation_field_func):
        """
        Applies the deformation field D(x) to the bubble centers[cite: 120].
        
        Args:
            path: (N, 3) array of points.
            deformation_field_func: Callable accepting (x,y,z) returning (dx,dy,dz).
        """
        deformed_path = []
        for pt in path:
            displacement = deformation_field_func(pt)
            deformed_path.append(pt + displacement)
        return np.array(deformed_path)

    def bubble_bending(self, path, obstacle_map_func, max_iter=20, k_ext=1.8):
        """
        Iteratively adjusts path based on internal (elastic) and external (repulsive) forces[cite: 123].
        
        Args:
            k_ext: External repulsion constant (increase for stronger obstacle avoidance)
        """
        current_path = path.copy()
        
        # Equilibrium parameters
        k_int = 0.5  # Elastic constant
        
        for _ in range(max_iter):
            forces = np.zeros_like(current_path)
            
            # Calculate forces for internal bubbles (exclude start/end)
            for j in range(2, len(current_path) - 2):
                p = current_path[j]
                p_prev = current_path[j-1]
                p_next = current_path[j+1]
                
                # Internal Force: Keep equidistant/smooth [cite: 105]
                f_int = k_int * ((p_prev + p_next)/2 - p)
                
                # External Force: Repulsion from obstacles [cite: 102]
                # We check obstacles within r_b
                obs_pt, distance = obstacle_map_func(p)
                f_ext = np.zeros(3)
                if distance < self.r_b:
                    # Direction away from obstacle
                    if distance > 0:
                        dir_vec = (p - obs_pt) / distance
                    else:
                        dir_vec = np.random.rand(3) # Random push if perfectly inside
                        dir_vec /= np.linalg.norm(dir_vec)
                    
                    # Magnitude increases as distance decreases
                    f_ext = k_ext * dir_vec * (self.r_b - distance)
                
                forces[j] = f_int + f_ext
            
            # Apply forces
            current_path += forces
            
            # Reorganize to maintain overlap constraint [cite: 126]
            current_path = self.bubble_reorganization(current_path)
            
        return current_path

    def constrained_smoothing(self, bubbles, tip_pose, target_pose, obstacle_map_func):
        """
        Optimizes path using Convex Elastic Smoothing (CES) with 3D constraints[cite: 129, 200].
        
        Args:
            bubbles: (N, 3) array of bubble centers.
            tip_pose: (pos, direction_vector)
            target_pose: (pos, direction_vector)
            obstacle_map_func: Callable for querying nearest obstacle distance
        """
        N = len(bubbles)
        if N < 5: return bubbles # Too short to smooth

        # Variables: p_i are the optimized waypoints [cite: 218]
        P = cp.Variable((N, 3))
        
        # Objective Function: Minimize O1 (curvature) + O2 (distance from bubbles)
        # O1 = sum ||2p_k - p_{k-1} - p_{k+1}||^2 [cite: 199]
        # O2 = sum ||p_i - q_i|| [cite: 199]
        
        O1 = cp.sum_squares(2*P[1:N-1] - P[0:N-2] - P[2:N])
        O2 = cp.sum_squares(P[2:N-2] - bubbles[2:N-2]) 

        
        constraints = []
        
        # 1. Start Pose Constraints [cite: 201, 202]
        # p1 = Tip_pos
        constraints.append(P[0] == tip_pose[0])
        # p2 ensures initial tangent matches tip vector
        # Approximate step size d based on bubbles
        d_avg = np.linalg.norm(bubbles[-1] - bubbles[0]) / N
        constraints.append(P[1] == tip_pose[0] + tip_pose[1] * d_avg)
        
        # 2. Target Pose Constraints [cite: 203]
        # Final orientation constraint (approach vector)
        constraints.append(P[N-2] == target_pose[0] - target_pose[1] * d_avg)
        
        # Target position tolerance [cite: 215]
        constraints.append(cp.sum_squares(P[N-1] - target_pose[0]) <= self.tar_tol**2)
        
        # 3. Curvature Constraint [cite: 214]
        # ||2p_k - p_{k-1} - p_{k+1}|| <= d^2 / R_min
        # This is a Second Order Cone constraint, supported by CVXPY
        limit = (d_avg**2) / self.r_min
        for k in range(1, N-1):
            constraints.append(cp.norm(2*P[k] - P[k-1] - P[k+1]) <= limit)
        
        # 4. Obstacle Avoidance (Soft Constraint via Penalty)
        # Instead of hard constraints that can make problem infeasible,
        # add penalty for waypoints that deviate toward obstacles
        penalty_terms = []
        penalty_weight = 0.5
        for i in range(N):
            bubble_center = bubbles[i]
            obs_pt, distance = obstacle_map_func(bubble_center)
            
            # Only penalize if bubble is close to obstacle
            if distance < self.r_b * 1.5:
                if distance > 1e-6:
                    grad = (bubble_center - obs_pt) / distance
                    # Penalize movement toward obstacle (negative gradient direction)
                    penalty_terms.append(penalty_weight * cp.neg(grad @ (P[i] - bubble_center)))
        
        # Update objective with obstacle penalty
        if len(penalty_terms) > 0:
            obstacle_penalty = cp.sum(penalty_terms)
            objective = cp.Minimize(O1 + O2 + obstacle_penalty)
        else:
            objective = cp.Minimize(O1 + O2)



        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except cp.SolverError:
            print("Smoothing solver failed.")
            return bubbles

        if P.value is None:
            print("No feasible path found during smoothing.")
            return bubbles
            
        return P.value

    def interpolate(self, waypoints, num_points=100):
        """Generates the final smooth path from waypoints."""
        # Simple spline interpolation for visual smoothness
        if len(waypoints) < 3: return waypoints
        
        t = np.linspace(0, 1, len(waypoints))
        f = interp1d(t, waypoints, axis=0, kind='cubic')
        t_new = np.linspace(0, 1, num_points)
        return f(t_new)


    def validate_path(self, path, obstacle_map_func):
        """
        Check if any point in the path collides with obstacles.
        
        Returns:
            (bool, list): (is_valid, list of collision indices)
        """
        collisions = []
        for i, point in enumerate(path):
            _, distance = obstacle_map_func(point)
            # More lenient threshold - inside the obstacle surface
            if distance < 0:  # Only flag if actually penetrating obstacle
                collisions.append(i)
        return len(collisions) == 0, collisions

    def plan(self, initial_path, tip_pose, target_pose, obstacle_map_func, deformation_func, max_attempts=3, num_path_points=100):
        """
        Main execution pipeline[cite: 114].
        Iteratively recomputes if path collides with obstacles.
        
        Args:
            max_attempts: Maximum number of attempts to find collision-free path
            num_path_points: Number of points in interpolated final path
            
        Returns:
            If successful: (final_path, bubbles, attempts_info)
            If failed: raises PathPlanningError with attempts_info attached
        """
        attempts_info = []  # Store info about each attempt
        
        for attempt in range(max_attempts):
            # Adjust repulsion strength for retries
            k_ext = 1.8 * (1.5 ** attempt)  # Exponentially increase repulsion
            
            if attempt > 0:
                print(f"Path collision detected. Retry {attempt}/{max_attempts-1} with k_ext={k_ext:.2f}...")
            
            # 1. Bubble Reorganisation
            q_b = self.bubble_reorganization(initial_path)
            
            # 2. Applied Deformation
            q_d = self.apply_deformation(q_b, deformation_func)
            
            # 3. Bubble Bending (with more iterations and stronger repulsion on retry)
            max_iter = 20 + (attempt * 10)
            q_c = self.bubble_bending(q_d, obstacle_map_func, max_iter=max_iter, k_ext=k_ext)
            
            # 4. Constrained Smoothing
            p_smooth = self.constrained_smoothing(q_c, tip_pose, target_pose, obstacle_map_func)
            
            # 5. Interpolation
            final_path = self.interpolate(p_smooth, num_points=num_path_points)
            
            # 6. Validation
            is_valid, collisions = self.validate_path(final_path, obstacle_map_func)
            
            # Store attempt info
            attempt_info = {
                'attempt': attempt,
                'k_ext': k_ext,
                'path': final_path.copy(),
                'bubbles': q_c.copy(),
                'is_valid': is_valid,
                'collision_indices': collisions,
                'num_collisions': len(collisions)
            }
            attempts_info.append(attempt_info)
            
            if is_valid:
                if attempt > 0:
                    print(f"✓ Valid obstacle-free path found on attempt {attempt + 1}")
                return final_path, q_c, attempts_info
            else:
                print(f"  {len(collisions)} collision points detected")
        
        # No feasible path found after all attempts
        error_msg = (
            f"NO FEASIBLE PATH: Could not find collision-free path after {max_attempts} attempts. "
            f"Last attempt had {len(collisions)} collision points. "
            "Try adjusting planner parameters (increase r_b, decrease r_min) or modifying obstacles."
        )
        print(f"❌ {error_msg}")
        
        # Attach attempts_info to exception for visualization
        error = PathPlanningError(error_msg)
        error.attempts_info = attempts_info
        raise error