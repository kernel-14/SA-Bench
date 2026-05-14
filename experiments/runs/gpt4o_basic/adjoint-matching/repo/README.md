# Adjoint Matching: Fine-tuning Flow and Diffusion Models

## Project Overview

This repository implements and reproduces the core contributions of the paper 'Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC'.

Implemented Components:
- Generative models for Flow Matching and Denoising Diffusion techniques.
- Memoryless noise schedule for fine-tuning generative processes.
- Adjoint Matching algorithm for stable and efficient fine-tuning.

Next Steps:
- Expand on the experimental setup using pretrained architectures.
- Validate extensively with proper metrics and sampled distributions.

## Repository Structure

- src/models.py: Base classes for generative models
- src/noise_schedule.py: Implementation of memoryless noise schedules
- src/adjoint_matching.py: Adjoint Matching fine-tuning
- tests/basic_test.py: Placeholder tests

