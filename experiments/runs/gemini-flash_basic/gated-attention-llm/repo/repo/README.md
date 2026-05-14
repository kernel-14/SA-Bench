# Gated Attention for Large Language Models: Reproduction

This repository contains a reproduction attempt of the core contributions from the paper "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free".

## Implemented Core Contribution: Gated Self-Attention (G1 Position)

The primary focus of this reproduction is the implementation of the **Gated Self-Attention mechanism** at the `G1` position, as described in Section 2.2 and highlighted as the most effective variant in Table 1 of the paper. This involves applying a head-specific, element-wise sigmoid gate after the Scaled Dot-Product Attention (SDPA) output.

### `gated_attention.py`

This file contains the `GatedSelfAttention` class, which is a PyTorch `nn.Module` that implements the gated attention mechanism. Key aspects of the implementation follow the paper's descriptions:

-   **Gating Position (`G1`):** The gate is applied directly to the output of the Scaled Dot-Product Attention (SDPA) before the final linear projection (`W_O`).
-   **Gating Mechanism:** As formalized in Equation 5 of the paper, the gating is multiplicative, i.e., `Y' = Y * sigma(X W_theta)`. Here, `Y` is the SDPA output, `X` (used to compute gating scores) is the original `query` tensor, `W_theta` is a learnable linear projection (`self.gating_W_theta`), and `sigma` is the sigmoid activation function.
-   **Granularity (Element-wise):** The gating scores have the same dimensionality as the multi-head SDPA output, allowing for fine-grained, per-dimension modulation.
-   **Head-Specific:** Each attention head receives its specific gating scores, enabling independent modulation for each head. This is achieved by ensuring `self.gating_W_theta` projects to `embed_dim`, which is then reshaped to match the multi-head SDPA output dimensions.
-   **Activation Function (Sigmoid):** As specified, a sigmoid activation function is used for the gate, providing scores in the range `[0, 1]` for multiplicative gating.
-   **Query-Dependency:** The gating scores are derived from the original query tensor, aligning with the paper's finding that query-dependent sparsity is crucial for effectiveness (Section 4.2, point iii).

### Relation to Paper Sections:

-   **Section 2.1: Preliminary: Multi-Head Softmax Attention:** The base multi-head attention mechanism (QKV projections, SDPA, concatenation, final output layer) is implemented according to this section.
-   **Section 2.2: Augmenting Attention Layer with Gating Mechanisms:** This section provides the core formalization of the gating mechanism and the different variants explored. Our implementation specifically targets the `G1` position with head-specific, element-wise, multiplicative sigmoid gating based on the query.
-   **Table 1: Gating variant performance and results:** Our chosen implementation (SDPA Elementwise G1 with sigmoid) corresponds to row (5) in Table 1, which shows significant performance improvements.
-   **Section 4.1: Non-linearity Improves the Expressiveness of Low-Rank Mapping in Attention:** The introduction of sigmoid non-linearity via gating at `G1` directly addresses the low-rank mapping issue discussed here.
-   **Section 4.2: Gating Introduces Input-Dependent Sparsity:** The design ensures input-dependent and head-specific gating, which the paper identifies as crucial for inducing sparsity and mitigating attention sinks.

## Assumptions and Missing Details

-   **Integration into a Full Transformer:** This submission only provides the `GatedSelfAttention` module. To fully reproduce the paper's results, this module would need to be integrated into a complete Transformer architecture (e.g., within a Transformer Encoder/Decoder layer).
-   **Hyperparameters:** Specific hyperparameters (e.g., `embed_dim`, `num_heads`, `dropout`) would be determined by the larger model configuration, which is not part of this isolated module.
-   **Attention Masking:** The implementation includes basic support for `attn_mask` and `key_padding_mask` based on standard Transformer practices. The specific types of masks (e.g., causal masking for LLMs) would depend on the use case.
-   **Weight Initialization:** Standard PyTorch `nn.Linear` weight initialization is assumed. The paper does not specify custom initialization for `W_theta`.
-   **GQA and other advanced features:** While the paper mentions Group Query Attention (GQA) for MoE models, this general `GatedSelfAttention` implementation does not explicitly incorporate GQA as it focuses on the fundamental G1 gating mechanism for standard multi-head attention.

This reproduction provides the fundamental building block for the gated attention mechanism, adhering closely to the paper's most impactful findings regarding gating at the SDPA output.
