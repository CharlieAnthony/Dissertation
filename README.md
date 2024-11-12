# Simultaneous Localization and Mapping (SLAM) Simulation

This project explores the implementation of a basic Simultaneous Localization and Mapping (SLAM) algorithm within a simulated environment. SLAM enables an agent to create a map of its surroundings while estimating its own position, a capability critical for applications in autonomous vehicles, drones, and robotic navigation systems.

## Project Overview

The project is divided into two main components:

1. **Feature Detection**: Uses the Seeded Region Growing algorithm to detect and extract features in the environment.
2. **Localization**: Employs the Extended Kalman Filter (EKF) to estimate the agent's position based on measurements from the feature detection module.

## Simulation Environment

The project is developed in Python, using Pygame for visualization and various scientific libraries for processing. The agent navigates a virtual space, incrementally updating its position estimate and mapping its environment.

## Key Findings

- Successfully achieved localization of the agent within the simulated environment.
- Encountered limitations in fully integrating the feature detection and localization components.

This project serves as a foundational implementation of SLAM, with potential for further enhancements in feature detection and data association techniques.
