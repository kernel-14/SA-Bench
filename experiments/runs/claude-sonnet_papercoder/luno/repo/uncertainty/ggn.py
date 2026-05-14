## Code: uncertainty/ggn.py

```python
## uncertainty/ggn.py
"""Generalized Gauss-Newton (GGN) low-rank approximation for LUNO-LA.

This module computes the top-rank eigenpairs of the GGN matrix restricted to
the last Fourier block parameters of a trained FNO. The GGN is never formed
explicitly; all operations are matrix-free via JAX JVP/VJP primitives.

Mathematical foundation:
  For MSE loss with Gaussian likelihood, H_f = I, so:
    G = sum_{i=1}^{n} J_i^T J_i
  where J_i = D_{w_{L-1}} f(x^(i), w)|_{w*} is the last-layer Jacobian.

  GGN-vector product (forward-over-reverse):
    G * v = sum_i J_i^T (J_i * v)
    Step 1: Jv = J_i * v  via jax.jvp  (shape: [n_out])
    Step 2: JtJv = J_i^T * Jv  via jax.vjp  (shape: [p_last])

  Low-rank approximation via Lanczos:
    G ≈ V * diag(lambda) * V^T  (top-rank eigenpairs)

  Used by LaplaceApprox (uncertainty/luno.py) for Woodbury-based posterior
  covariance: (n*G + prior_prec*I)^{-1} * v.

Paper references:
  - Appendix C.1: Last-layer LUNO for FNOs (w_{L-1} = (R^{(L-1)}, W^{(L-1)}))
  - Appendix D.3.4: "low rank of 500", "all input-output pairs" (low-data),
    "minibatch of 1000 input-output pairs" (OOD)
  - config.yaml uncertainty.ggn: rank=500, last_layer_only=True,
    n_pairs_low_data=25, n_pairs_ood=1000
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from data.dataset import PDEDataset
from models.fno import FNO
from utils.jax_utils import flatten_params

logger = logging.getLogger(__name__)


class GGNComputer:
    """Computes a low-rank approximation of the GGN for the FNO's last layer.

    Restricts the GGN computation to the last Fourier block parameters
    w_{L-1} = (Re(R^{(L-1)}), Im(R^{(L-1)}), W^{(L-1)}) as described in
    Appendix C.1 of the LUNO paper. Uses matrix-free JVP/VJP operations and
    the Lanczos algorithm to extract the top-rank eigenpairs.

    Attributes:
        model: The trained FNO instance.
        params: Full parameter pytree of the trained FNO (MAP weights w*).
        rank: Number of eigenpairs to compute. Config: 500.
        last_layer_only: Whether to restrict GGN to the last Fourier block.
            Config: True. Always True in the current implementation.

    Example::

        ggn = GGNComputer(model=fno, params=trained_params, rank=500,
                          last_layer_only=True)
        key = jax.random.PRNGKey(0)
        eigvecs, eigvals = ggn.compute_low_rank(train_dataset, n_pairs=25, key=key)
        # eigvecs.shape == (p_last, 500)
        # eigvals.shape == (500,)
    """

    def __init__(
        self,
        model: FNO,
        params: dict,
        rank: int = 500,
        last_layer_only: bool = True,
    ) -> None:
        """Initialise the GGN computer.

        Args:
            model: The trained FNO instance. Used to access the graph
                definition for functional forward passes.
            params: Full parameter pytree of the trained FNO at the MAP
                weights w*. Typically the ``state`` returned by
                ``Trainer.train()``.
            rank: Number of top eigenpairs to compute via Lanczos.
                From ``config.uncertainty.ggn.rank`` (default 500).
            last_layer_only: Whether to restrict the GGN to the last
                Fourier block parameters only. From
                ``config.uncertainty.ggn.last_layer_only`` (default True).
                The current implementation always uses last-layer-only mode.

        Raises:
            ValueError: If ``rank <= 0``.
        """
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.model: FNO = model
        self.params: dict = params
        self.rank: int = rank
        self.last_layer_only: bool = last_layer_only

        # Cache the graphdef for functional forward passes
        self._graphdef: Optional[object] = None
        self._init_graphdef()

        logger.info(
            "GGNComputer initialised: rank=%d, last_layer_only=%s",
            rank,
            last_layer_only,
        )

    def _init_graphdef(self) -> None:
        """Extract and cache the NNX graph definition from the model.

        The graph definition is needed for ``nnx.merge`` in functional
        forward passes. It is extracted once and cached to avoid repeated
        splitting.
        """
        try:
            graphdef, _ = nnx.split(self.model)
            self._graphdef = graphdef
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Could not extract graphdef from model: %s. "
                "Will attempt extraction on first forward pass.",
                e,
            )
            self._graphdef = None

    # -----------------------------------------------------------------------
    # Last-Layer Parameter Extraction
    # -----------------------------------------------------------------------

    def _get_last_layer_params(
        self,
        params: dict,
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], dict]]:
        """Extract and flatten the last Fourier block parameters.

        Navigates the Flax NNX parameter pytree to extract the sub-pytree
        corresponding to ``last_fourier_block``, then flattens it to a 1-D
        array using ``jax.flatten_util.ravel_pytree``.

        The last Fourier block contains:
          - ``spectral_conv.weights_real``: Re(R^{(L-1)}), shape [modes, C, C]
          - ``spectral_conv.weights_imag``: Im(R^{(L-1)}), shape [modes, C, C]
          - ``pointwise.kernel``: W^{(L-1)}, shape [C, C]

        where C = config.model.channels = 18 and modes = config.model.modes = 12.

        Args:
            params: Full parameter pytree of the FNO. This is the NNX
                ``state`` object returned by ``nnx.split(model)``.

        Returns:
            Tuple ``(flat_last_layer, unflatten_fn)`` where:
            - ``flat_last_layer``: 1-D array of shape ``[p_last]`` containing
              all last-layer parameters concatenated.
            - ``unflatten_fn``: Callable that maps a flat array of shape
              ``[p_last]`` back to the last-layer sub-pytree structure.

        Raises:
            KeyError: If the expected parameter keys are not found in the
                pytree (indicates a mismatch between FNO architecture and
                expected key names).
        """
        # ------------------------------------------------------------------
        # Extract the last_fourier_block sub-pytree from the NNX state.
        # NNX state is a nested structure; we need to find the sub-tree
        # corresponding to last_fourier_block.
        # ------------------------------------------------------------------
        last_layer_subtree = self._extract_last_layer_subtree(params)

        # ------------------------------------------------------------------
        # Flatten the sub-pytree to a 1-D array
        # ------------------------------------------------------------------
        flat_last_layer, unflatten_fn = flatten_params(last_layer_subtree)

        logger.debug(
            "_get_last_layer_params: p_last=%d", int(flat_last_layer.shape[0])
        )

        return flat_last_layer, unflatten_fn

    def _extract_last_layer_subtree(self, params: object) -> object:
        """Navigate the NNX state to find the last_fourier_block sub-tree.

        Handles both NNX State objects and plain dict pytrees. The NNX
        state structure mirrors the module attribute hierarchy:
          model.last_fourier_block → params['last_fourier_block']

        For NNX State objects, we use ``nnx.State`` traversal. For plain
        dicts (e.g., after ``jax.tree_util`` operations), we use dict access.

        Args:
            params: The full parameter pytree (NNX State or nested dict).

        Returns:
            The sub-pytree corresponding to ``last_fourier_block``.

        Raises:
            KeyError: If ``last_fourier_block`` is not found in the pytree.
            TypeError: If the pytree structure is not navigable.
        """
        # ------------------------------------------------------------------
        # Strategy: convert to a flat dict of (path, value) pairs and
        # filter for paths containing 'last_fourier_block'.
        # This is robust to different NNX state representations.
        # ------------------------------------------------------------------

        # Try direct attribute access first (NNX State object)
        if hasattr(params, "last_fourier_block"):
            return params.last_fourier_block

        # Try dict-style access
        if isinstance(params, dict) and "last_fourier_block" in params:
            return params["last_fourier_block"]

        # ------------------------------------------------------------------
        # Fallback: use jax.tree_util to find the subtree.
        # NNX State objects are registered as JAX pytrees; we can use
        # jax.tree_util.tree_map_with_path to find the relevant leaves.
        # ------------------------------------------------------------------
        # Convert to a nested dict representation via leaves and paths
        try:
            return self._find_subtree_by_key(params, "last_fourier_block")
        except (KeyError, AttributeError) as e:
            raise KeyError(
                f"Could not find 'last_fourier_block' in parameter pytree. "
                f"Ensure the FNO was built with the expected architecture "
                f"(models/fno.py). Original error: {e}"
            ) from e

    def _find_subtree_by_key(self, tree: object, target_key: str) -> object:
        """Recursively search a pytree for a sub-tree with the given key.

        Handles NNX State objects, plain dicts, and other pytree containers.

        Args:
            tree: The pytree to search.
            target_key: The key to search for.

        Returns:
            The sub-tree associated with ``target_key``.

        Raises:
            KeyError: If ``target_key`` is not found.
        """
        # Check if this node has the target key as an attribute
        if hasattr(tree, target_key):
            return getattr(tree, target_key)

        # Check dict-style access
        if isinstance(tree, dict):
            if target_key in tree:
                return tree[target_key]
            # Recurse into dict values
            for v in tree.values():
                try:
                    return self._find_subtree_by_key(v, target_key)
                except KeyError:
                    continue

        # For NNX State objects, try to access via __dict__ or vars()
        if hasattr(tree, "__dict__"):
            d = vars(tree)
            if target_key in d:
                return d[target_key]
            for v in d.values():
                if v is tree:
                    continue  # avoid infinite recursion
                try:
                    return self._find_subtree_by_key(v, target_key)
                except (KeyError, RecursionError):
                    continue

        raise KeyError(f"Key '{target_key}' not found in pytree.")

    def _merge_last_layer_into_params(
        self,
        flat_last_layer: jnp.ndarray,
        unflatten_fn: Callable[[jnp.ndarray], object],
    ) -> object:
        """Reconstruct the full parameter pytree with updated last-layer params.

        Takes a flat last-layer parameter vector, unflattens it to the
        last-layer sub-pytree, and merges it back into the full parameter
        pytree (replacing the original last-layer values).

        This is used in ``_jvp_fn`` and ``_vjp_fn`` to define the
        differentiable function ``last_layer_forward(flat_w_last)``.

        Args:
            flat_last_layer: Flat last-layer parameter vector, shape [p_last].
            unflatten_fn: Callable that maps flat_last_layer back to the
                last-layer sub-pytree.

        Returns:
            Full parameter pytree with the last-layer sub-tree replaced by
            the unflattened ``flat_last_layer``.
        """
        # Unflatten to last-layer sub-pytree
        new_last_layer = unflatten_fn(flat_last_layer)

        # ------------------------------------------------------------------
        # Merge into the full params pytree.
        # We use jax.tree_util.tree_map to create a new pytree where the
        # last_fourier_block sub-tree is replaced.
        #
        # Strategy: use the NNX merge approach — split the current params,
        # replace the last_fourier_block portion, and return the modified tree.
        # ------------------------------------------------------------------
        # The cleanest approach: use nnx.merge with a modified state.
        # We need to create a new state where last_fourier_block is replaced.

        # Use a path-based replacement via jax.tree_util
        new_params = self._replace_subtree(self.params, "last_fourier_block", new_last_layer)
        return new_params

    def _replace_subtree(
        self,
        tree: object,
        target_key: str,
        new_subtree: object,
    ) -> object:
        """Create a new pytree with the sub-tree at ``target_key`` replaced.

        This is a shallow replacement: only the top-level occurrence of
        ``target_key`` is replaced. The rest of the tree is shared (not copied).

        Args:
            tree: The original pytree.
            target_key: The key whose value should be replaced.
            new_subtree: The new sub-tree to insert at ``target_key``.

        Returns:
            A new pytree with the replacement applied.

        Raises:
            KeyError: If ``target_key`` is not found at the top level.
        """
        # For NNX State objects: use nnx.State's update mechanism
        # The safest approach is to work with the flat leaves and paths.

        # Convert to a mutable representation
        if isinstance(tree, dict):
            if target_key in tree:
                new_tree = dict(tree)
                new_tree[target_key] = new_subtree
                return new_tree
            raise KeyError(f"Key '{target_key}' not found in dict pytree.")

        # For NNX State objects, we need to use the NNX API
        # nnx.State supports attribute-style access and can be updated
        if hasattr(tree, target_key):
            # Use nnx.State's copy-with-replacement if available
            # Otherwise, create a new state by modifying the flat representation
            try:
                # Try NNX State update (creates a new state)
                import copy
                new_tree = copy.copy(tree)
                object.__setattr__(new_tree, target_key, new_subtree)
                return new_tree
            except Exception:  # pylint: disable=broad-except
                pass

        # Fallback: use jax.tree_util path-based replacement
        # This works for any registered pytree
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        # We need to find which leaves correspond to last_fourier_block
        # and replace them. This requires path information.
        try:
            paths_and_leaves = jax.tree_util.tree_leaves_with_path(tree)
            new_leaves_subtree, _ = jax.tree_util.tree_flatten(new_subtree)

            # Find the indices of leaves that belong to last_fourier_block
            target_indices = []
            for idx, (path, _) in enumerate(paths_and_leaves):
                path_str = str(path)
                if target_key in path_str:
                    target_indices.append(idx)

            if len(target_indices) == len(new_leaves_subtree):
                new_leaves = list(leaves)
                for i, idx in enumerate(target_indices):
                    new_leaves[idx] = new_leaves_subtree[i]
                return jax.tree_util.tree_unflatten(treedef, new_leaves)
        except Exception:  # pylint: disable=broad-except
            pass

        raise KeyError(
            f"Could not replace subtree at key '{target_key}'. "
            f"Tree type: {type(tree)}"
        )

    # -----------------------------------------------------------------------
    # Functional Forward Pass
    # -----------------------------------------------------------------------

    def _make_last_layer_forward(
        self,
        unflatten_fn: Callable[[jnp.ndarray], object],
        x: jnp.ndarray,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Create a function mapping flat last-layer weights to model output.

        The returned function is differentiable w.r.t. its argument
        (flat last-layer weights) via JAX's AD system.

        The function:
          1. Unflattens flat_w_last to the last-layer sub-pytree.
          2. Merges into the full params (replacing last_fourier_block).
          3. Runs the FNO forward pass on input x.
          4. Returns the flattened output (shape [n_out]).

        Args:
            unflatten_fn: Callable that maps flat [p_last] → last-layer subtree.
            x: Single input function, shape [1, spatial, in_channels] (1D)
               or [1, H, W, in_channels] (2D). Batch dimension = 1.

        Returns:
            A callable ``f(flat_w_last) -> output_flat`` where:
            - Input: flat last-layer weights, shape [p_last].
            - Output: flattened model output, shape [n_out] where
              n_out = spatial * out_channels (1D) or H * W * out_channels (2D).
        """
        # Capture graphdef for nnx.merge
        graphdef = self._graphdef
        full_params = self.params

        def last_layer_forward(flat_w_last: jnp.ndarray) -> jnp.ndarray:
            """Forward pass with differentiable last-layer weights.

            Args:
                flat_w_last: Flat last-layer parameter vector, shape [p_last].

            Returns:
                Flattened model output, shape [n_out].
            """
            # Reconstruct last-layer sub-pytree
            new_last_layer = unflatten_fn(flat_w_last)

            # Merge into full params
            new_params = _replace_last_layer_in_state(
                full_params, new_last_layer
            )

            # Run forward pass via nnx.merge
            if graphdef is not None:
                model_copy = nnx.merge(graphdef, new_params)
                output = model_copy(x)
            else:
                # Fallback: use model directly (less safe for AD)
                raise RuntimeError(
                    "graphdef not available for functional forward pass. "
                    "Ensure GGNComputer._init_graphdef() succeeded."
                )

            # Flatten output: [1, spatial, out_channels] → [n_out]
            output_flat = output.reshape(-1)
            return output_flat

        return last_layer_forward

    # -----------------------------------------------------------------------
    # JVP and VJP
    # -----------------------------------------------------------------------

    def _jvp_fn(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        flat_last_layer: jnp.ndarray,
        unflatten_fn: Callable[[jnp.ndarray], object],
    ) -> jnp.ndarray:
        """Compute the Jacobian-vector product J(x) * v.

        Computes the directional derivative of the model output w.r.t. the
        last-layer weights in direction v, using forward-mode AD (jax.jvp).

        Args:
            x: Single input, shape [1, spatial, in_channels] (1D) or
               [1, H, W, in_channels] (2D).
            v: Tangent vector in last-layer weight space, shape [p_last].
            flat_last_layer: Current flat last-layer weights, shape [p_last].
                Serves as the primal point for the JVP.
            unflatten_fn: Callable mapping flat [p_last] → last-layer subtree.

        Returns:
            JVP result J(x) * v, shape [n_out] where
            n_out = spatial * out_channels (1D) or H * W * out_channels (2D).
        """
        forward_fn = self._make_last_layer_forward(unflatten_fn, x)

        # jax.jvp: (primals_out, tangents_out)
        _primals_out, tangents_out = jax.jvp(
            forward_fn,
            (flat_last_layer,),
            (v,),
        )

        return tangents_out  # shape [n_out]

    def _vjp_fn(
        self,
        x: jnp.ndarray,
        g: jnp.ndarray,
        flat_last_layer: jnp.ndarray,
        unflatten_fn: Callable[[jnp.ndarray], object],
    ) -> jnp.ndarray:
        """Compute the vector-Jacobian product J(x)^T * g.

        Computes the pullback of the cotangent g through the model output
        w.r.t. the last-layer weights, using reverse-mode AD (jax.vjp).

        Args:
            x: Single input, shape [1, spatial, in_channels] (1D) or
               [1, H, W, in_channels] (2D).
            g: Cotangent vector in output space, shape [n_out].
            flat_last_layer: Current flat last-layer weights, shape [p_last].
                Serves as the primal point for the VJP.
            unflatten_fn: Callable mapping flat [p_last] → last-layer subtree.

        Returns:
            VJP result J(x)^T * g, shape [p_last].
        """
        forward_fn = self._make_last_layer_forward(unflatten_fn, x)

        # jax.vjp: (primals_out, vjp_fn)
        _primals_out, vjp_pullback = jax.vjp(forward_fn, flat_last_layer)

        # Apply pullback: vjp_pullback(g) returns a tuple (grad_w,)
        (grad_w,) = vjp_pullback(g)

        return grad_w  # shape [p_last]

    # -----------------------------------------------------------------------
    # GGN-Vector Product
    # -----------------------------------------------------------------------

    def _ggn_vector_product(
        self,
        v: jnp.ndarray,
        dataset: PDEDataset,
        n_pairs: int,
    ) -> jnp.ndarray:
        """Compute G * v = sum_{i=1}^{n_pairs} J_i^T J_i v matrix-free.

        Iterates over ``n_pairs`` data points from ``dataset``, computing
        the GGN-vector product via forward-over-reverse AD:
          1. Jv_i = J_i * v  (JVP, forward mode)
          2. JtJv_i = J_i^T * Jv_i  (VJP, reverse mode)
          3. Accumulate: result += JtJv_i

        For MSE loss, H_f = I, so the GGN is exactly sum_i J_i^T J_i.

        Args:
            v: Vector in last-layer weight space, shape [p_last].
            dataset: Training dataset. The first ``n_pairs`` pairs are used.
                For low-data: n_pairs=25 (all training pairs).
                For OOD: n_pairs=1000 (minibatch).
            n_pairs: Number of data points to use. From config:
                ``config.uncertainty.ggn.n_pairs_low_data = 25`` or
                ``config.uncertainty.ggn.n_pairs_ood = 1000``.

        Returns:
            G * v, shape [p_last].

        Notes:
            - Uses a Python loop over data points (not jax.lax.fori_loop)
              because the Lanczos algorithm calls this function in a Python
              loop anyway, and JIT on the individual JVP/VJP calls is the
              right granularity.
            - Each JVP+VJP pair is JIT-compiled on the first call and
              cached by JAX's tracing mechanism.
        """
        # Extract last-layer params once (reused for all data points)
        flat_last_layer, unflatten_fn = self._get_last_layer_params(self.params)

        # Initialise accumulator
        result: jnp.ndarray = jnp.zeros_like(v)

        # Use at most n_pairs data points
        n_use: int = min(n_pairs, dataset.n_pairs)

        # JIT-compile the inner step for efficiency
        jit_jvp = jax.jit(self._jvp_fn, static_argnames=())
        jit_vjp = jax.jit(self._vjp_fn, static_argnames=())

        for i in range(n_use):
            # Get single input (add batch dimension)
            x_i: jnp.ndarray = dataset.inputs[i : i + 1]  # [1, spatial, in_channels]

            # Step 1: JVP — compute J_i * v
            jv_i: jnp.ndarray = jit_jvp(x_i, v, flat_last_layer, unflatten_fn)
            # jv_i.shape: [n_out]

            # Step 2: VJP — compute J_i^T * (J_i * v)
            jtjv_i: jnp.ndarray = jit_vjp(x_i, jv_i, flat_last_layer, unflatten_fn)
            # jtjv_i.shape: [p_last]

            # Accumulate
            result = result + jtjv_i

        return result

    # -----------------------------------------------------------------------
    # Lanczos Algorithm
    # -----------------------------------------------------------------------

    def _lanczos(
        self,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        dim: int,
        rank: int,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute top-rank eigenpairs of a symmetric PSD matrix via Lanczos.

        Implements the Lanczos algorithm with full re-orthogonalization
        (modified Gram-Schmidt) for numerical stability. The matrix is
        accessed only through matrix-vector products via ``matvec``.

        Algorithm:
          1. Initialize random unit vector q_1.
          2. For j = 1, ..., rank:
             a. z = matvec(q_j)
             b. alpha_j = q_j^T z
             c. z = z - alpha_j * q_j - beta_{j-1} * q_{j-1}  (3-term recurrence)
             d. Full re-orthogonalization: z -= sum_{k<j} (q_k^T z) * q_k
             e. beta_j = ||z||
             f. If beta_j < eps: early termination
             g. q_{j+1} = z / beta_j
          3. Build tridiagonal matrix T from alphas and betas.
          4. Eigendecompose T via jnp.linalg.eigh.
          5. Map eigenvectors back to original space: V = Q_mat @ S.
          6. Return top-rank eigenpairs sorted by descending eigenvalue.

        Args:
            matvec: Callable mapping [dim] → [dim]. Represents the symmetric
                PSD matrix (the GGN restricted to last-layer parameters).
            dim: Dimension of the parameter space (p_last).
            rank: Number of eigenpairs to compute. Config: 500.
            key: JAX PRNG key for initializing the random starting vector.

        Returns:
            Tuple ``(eigvecs, eigvals)`` where:
            - ``eigvecs``: Top-rank eigenvectors, shape [dim, rank].
              Columns are orthonormal.
            - ``eigvals``: Corresponding eigenvalues, shape [rank].
              Sorted in descending order. Non-negative (GGN is PSD).

        Notes:
            - Full re-orthogonalization costs O(rank * dim) per step,
              totaling O(rank^2 * dim). For rank=500 and dim~20k, this is
              ~5e9 FLOPs — feasible on GPU, slow on CPU.
            - Eigenvalues are clamped to [0, inf) after computation to
              guard against small negative values from floating-point errors.
            - If early termination occurs (beta < eps), the remaining
              eigenpairs are filled with zeros.
        """
        eps: float = 1e-10  # Breakdown threshold for beta

        # ------------------------------------------------------------------
        # Step 1: Initialize random unit vector
        # ------------------------------------------------------------------
        q_init: jnp.ndarray = jax.random.normal(key, shape=(dim,), dtype=jnp.float32)
        q_init = q_init / (jnp.linalg.norm(q_init) + eps)

        # Storage for Lanczos basis vectors and tridiagonal elements
        # Q_list[j] = q_{j+1} (0-indexed), shape [dim] each
        Q_list = [q_init]
        alphas = []  # diagonal of T
        betas = []   # off-diagonal of T (length rank-1)

        q_prev: jnp.ndarray = jnp.zeros(dim, dtype=jnp.float32)
        beta_prev: float = 0.0
        n_converged: int = 0  # actual number of Lanczos steps completed

        logger.info(
            "Lanczos: dim=%d, rank=%d, starting iterations ...", dim, rank
        )

        for j in range(rank):
            q_j: jnp.ndarray = Q_list[j]

            # ------------------------------------------------------------------
            # Step 2a: Matrix-vector product
            # ------------------------------------------------------------------
            z: jnp.ndarray = matvec(q_j)  # [dim]

            # ------------------------------------------------------------------
            # Step 2b: Diagonal element alpha_j = q_j^T z
            # ------------------------------------------------------------------
            alpha_j: float = float(jnp.dot(q_j, z))
            alphas.append(alpha_j)

            # ------------------------------------------------------------------
            # Step 2c: Three-term recurrence
            # z = z - alpha_j * q_j