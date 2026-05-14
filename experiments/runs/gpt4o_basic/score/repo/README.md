# Training Language Models to Self-Correct via RL (SCoRe)

## Overview
This implementation replicates key contributions from the paper titled 'Training Language Models to Self-Correct via RL (SCoRe)'. The work introduces SCoRe, a two-stage reinforcement learning framework that improves intrinsic self-correction abilities of language models for tasks like MATH and HumanEval.

## Directory Structure
- **src/**: Contains core implementation files for RL training and metrics.
- **data/**: Placeholder location for datasets and synthetic traces used for training.
- **experiments/**: Reserved for configurations and experiment tracking.

## Replication Goals
1. Implement the SCoRe two-stage RL pipeline:
   - Stage I: Initialize the self-correction policy to decouple first and second attempts.
   - Stage II: Refine self-correction strategy with reward shaping.
2. Define and calculate paper-specific metrics.
3. Recreate experiments involving reasoning and arithmetic examples leveraging model traces.

## Notes
- Some synthetic data examples (e.g., modular arithmetic, reasoning traces) will be generated as stubs for testing the framework.


