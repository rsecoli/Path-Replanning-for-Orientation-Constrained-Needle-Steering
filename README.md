# Needle Steering Planner

Path Replanning for Orientation-Constrained Needle Steering implementation.

## Algorithm Overview

This implementation uses the **Extended Bubble Bending and Constrained Smoothing** algorithm for needle path replanning in 3D space with orientation constraints.

### Key Concepts

**Bubble-Based Representation**: The path is represented as a series of overlapping spheres ("bubbles") with radius `r_b` and minimum overlap `delta`. This ensures smooth curvature constraints are maintained.

**Curvature Constraints**: The needle has a minimum bending radius `r_min` that limits how sharply it can turn, representing physical steering limitations.

### Algorithm Pipeline

The planning process consists of four main stages:

1. **Bubble Reorganization**
   - Ensures bubbles maintain proper overlap (`delta`)
   - Maintains smooth spacing along the path

2. **Deformation Field Application**
   - Applies external forces (e.g., tissue deformation)
   - Currently uses a simple deformation model

3. **Bubble Bending**
   - Iteratively adjusts path using internal and external forces
   - **Internal forces** keep bubbles equidistant (elastic behavior)
   - **External forces** push bubbles away from obstacles (repulsion)
   - Repulsion strength (`k_ext`) increases exponentially on retry attempts
   - More iterations on retries for better convergence

4. **Constrained Smoothing (CES)**
   - Convex optimization to minimize curvature and deviation
   - Enforces start/end pose constraints
   - Enforces curvature limits (`r_min`)
   - Soft obstacle avoidance penalties (feasibility-preserving)

### Collision Handling

The planner includes automatic retry logic:
- **Validation**: Checks final path for obstacle penetration
- **Retry**: On collision, increases repulsion force and iterations
- **Max attempts**: Configurable (default: 10)
- **Error handling**: Raises clear error if no feasible path found

### Visualization

- **Success**: Shows final path with bubbles and Start pose
- **Failure**: Shows all retry attempts in different colors with collision points marked

## Installation

```bash
uv sync
```

## Usage

Run the simulation:

```bash
# Run with default scenario (random)
uv run path-planner-sim

# Or run specific scenarios
uv run python -m path_planner simple
uv run python -m path_planner complex
uv run python -m path_planner random
```

## Configuration

Edit parameters in `src/path_planner/__main__.py` to customize:
- **Planner parameters**: `r_b` (bubble radius), `delta` (overlap), `r_min` (min curvature), `tar_tol` (target tolerance)
- **Start/target poses**: Custom starting and ending positions/orientations
- **Obstacles**: Number, size (`obstacle_radius`), and random seed
- **Retry attempts**: `max_attempts` for collision-free path finding

## References

This implementation is based on the following paper:

> R. Secoli and F. Rodriguez y Baena, "Adaptive Path-Planning for Flexible Needle Steering Using Parallel Optimization and Replanning," 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Prague, Czech Republic, 2021, pp. 4965-4972, doi: 10.1109/IROS51168.2021.9359507.

**IEEE Xplore**: [https://ieeexplore.ieee.org/document/9359507](https://ieeexplore.ieee.org/document/9359507)

## License & Author

**License**: Apache License 2.0 - see [LICENSE](LICENSE) file for details

**Author**: Riccardo Secoli  
**LinkedIn**: [www.linkedin.com/in/riccardosecoli](https://www.linkedin.com/in/riccardosecoli)

### Disclaimer

This software is provided for research and educational purposes. It should not be used in production medical systems or clinical environments without thorough validation, testing, and regulatory approval. See [LICENSE](LICENSE) for full disclaimer.