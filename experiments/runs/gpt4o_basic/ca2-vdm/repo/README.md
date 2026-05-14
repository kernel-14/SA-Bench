# Ca2-VDM Replication

This repository contains the implementation of the core concepts introduced in the Ca2-VDM paper, focusing on efficient autoregressive video generation via causal computation and cache sharing.

## Key Features
- **Causal Temporal Attention:** Restricts attention to preceding frames for autoregression.
- **Prefix Spatial Attention:** Enhances spatial guidance using conditional prefix frames.
- **KV-Cache Mechanisms:** Optimized computation and memory-sharing strategies for temporal and spatial caches.

## Files and Directories
- **model/**: Contains implementations of causal attention, prefix spatial attention, and KV-cache mechanisms.
  - `causal_attention.py`: Causal attention function.
  - `prefix_spatial_attention.py`: Spatial attention enriched with prefix guidance.
  - `kv_cache.py`: Temporal and spatial cache handling.
- **scripts/**: Supports model training and evaluation.
  - `train_ca2vdm.py`: Training loops with cache integration.
- **dataset/**: Placeholder for data handling.

## Usage
1. Clone the repository.
2. Integrate datasets into the **dataset/** directory.
3. Run the training script:
    ```bash
    python scripts/train_ca2vdm.py
    ```

## Outstanding Tasks
- Finalize model architecture.
- Complete data preprocessing and loading.
- Extend evaluation metrics and implement benchmark comparisons.

## References
Refer to the original Ca2-VDM paper published as part of this implementation.

## Contact
For questions, contact the repository maintainer.

