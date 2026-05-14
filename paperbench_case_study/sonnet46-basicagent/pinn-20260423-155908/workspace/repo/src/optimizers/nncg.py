"""
NysNewton-CG (NNCG) optimizer.

Implements Algorithm 4 from the paper:
  - Randomized Nyström Approximation (Algorithm 5)
  - NyströmPCG (Algorithm 6)
  - Armijo line search (Algorithm 7)

Reference: Rathore et al., "Challenges in Training PINNs: A Loss Landscape Perspective", ICML 2024.
"""

import torch
import numpy as np


def randomized_nystrom_approximation(hvp_fn, p, s, device):
    """
    Algorithm 5: RandomizedNyströmApproximation.

    Args:
        hvp_fn: function that computes Hessian-vector products H @ v
        p: number of parameters
        s: sketch size
        device: torch device

    Returns:
        U: (p, s) approximate eigenvectors
        Lambda_hat: (s,) approximate eigenvalues
    """
    # Generate test matrix S ~ N(0, I)
    S = torch.randn(p, s, device=device)
    # Thin QR decomposition
    Q, _ = torch.linalg.qr(S, mode='reduced')  # (p, s)

    # Compute sketch Y = H @ Q column by column
    Y = torch.zeros(p, s, device=device)
    for i in range(s):
        Y[:, i] = hvp_fn(Q[:, i])

    # Compute shift for numerical stability
    norm_Y = torch.linalg.norm(Y, ord=2)
    nu = torch.sqrt(torch.tensor(float(p), device=device)) * torch.finfo(torch.float32).eps * norm_Y

    Y_nu = Y + nu * Q  # (p, s)

    # Cholesky decomposition of Q^T Y_nu
    QtY = Q.t() @ Y_nu  # (s, s)
    QtY = (QtY + QtY.t()) / 2  # symmetrize

    lam = 0.0
    try:
        C = torch.linalg.cholesky(QtY)  # C^T C = Q^T Y_nu
        B = torch.linalg.solve_triangular(C.t(), Y.t(), upper=True).t()  # Y @ C^{-1}
    except Exception:
        # Fallback: eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(QtY)
        lam_min = eigvals.min().item()
        lam = lam_min
        shift = abs(lam_min)
        R = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + shift)) @ eigvecs.t()
        B = Y @ R  # (p, s)

    # Thin SVD of B
    U_hat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)  # U_hat: (p, s), Sigma: (s,)

    # Compute eigenvalues, remove shift
    shift_total = nu + abs(lam)
    Lambda_hat = torch.clamp(Sigma ** 2 - shift_total, min=0.0)

    return U_hat, Lambda_hat


def nystrom_pcg(hvp_fn, b, x0, U, Lambda_hat, mu, eps, max_iter, device):
    """
    Algorithm 6: NyströmPCG.

    Solves (H + mu*I) x = b using preconditioned CG with Nyström preconditioner.

    Preconditioner:
      P^{-1} = (lambda_s + mu) * U @ diag(1/(Lambda_hat + mu)) @ U^T + (I - U @ U^T)
    where lambda_s = Lambda_hat[-1] (smallest approximate eigenvalue).

    Args:
        hvp_fn: Hessian-vector product function
        b: right-hand side vector (p,)
        x0: initial guess (p,)
        U: (p, s) approximate eigenvectors
        Lambda_hat: (s,) approximate eigenvalues
        mu: damping parameter
        eps: CG tolerance
        max_iter: max CG iterations
        device: torch device

    Returns:
        x: solution (p,)
    """
    s = Lambda_hat.shape[0]
    lambda_s = Lambda_hat[-1].item() if s > 0 else 0.0

    def apply_precond_inv(r):
        """Apply P^{-1} to vector r."""
        # P^{-1} r = (lambda_s + mu) * U @ diag(1/(Lambda_hat + mu)) @ U^T @ r + (I - U @ U^T) @ r
        Utr = U.t() @ r  # (s,)
        scaled = (lambda_s + mu) / (Lambda_hat + mu) * Utr  # (s,)
        UUtr = U @ Utr  # (p,)
        return (lambda_s + mu) * (U @ (scaled / (lambda_s + mu))) + (r - UUtr)

    def apply_A(v):
        """Apply (H + mu*I) to v."""
        return hvp_fn(v) + mu * v

    x = x0.clone()
    r = b - apply_A(x)
    z = apply_precond_inv(r)
    p_vec = z.clone()
    rz = torch.dot(r, z)

    for k in range(max_iter):
        if torch.linalg.norm(r) < eps:
            break
        Ap = apply_A(p_vec)
        pAp = torch.dot(p_vec, Ap)
        if pAp <= 0:
            break
        alpha = rz / pAp
        x = x + alpha * p_vec
        r = r - alpha * Ap
        z = apply_precond_inv(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p_vec = z + beta * p_vec
        rz = rz_new

    return x


def armijo_line_search(loss_fn, params, grad, direction, eta, alpha=0.1, beta=0.5, max_iter=50):
    """
    Algorithm 7: Armijo line search.

    Finds step size t such that:
      f(x + t*d) <= f(x) + alpha * t * (grad^T d)

    Args:
        loss_fn: callable returning scalar loss
        params: list of parameter tensors
        grad: flat gradient vector
        direction: search direction (negative Newton step, i.e., d = -newton_step)
        eta: initial step size
        alpha: sufficient decrease parameter
        beta: backtracking factor
        max_iter: maximum backtracking iterations

    Returns:
        t: step size
    """
    t = eta
    f0 = loss_fn().item()
    slope = torch.dot(grad, direction).item()

    # Save original params
    orig_params = [p.data.clone() for p in params]

    for _ in range(max_iter):
        # Update params
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.add_(t * direction[offset:offset + numel].view_as(p.data))
            offset += numel

        f_new = loss_fn().item()

        if f_new <= f0 + alpha * t * slope:
            return t

        # Restore params and shrink step
        for p, orig in zip(params, orig_params):
            p.data.copy_(orig)
        t *= beta

    # Restore params if no improvement found
    for p, orig in zip(params, orig_params):
        p.data.copy_(orig)
    return 0.0


class NNCG:
    """
    NysNewton-CG optimizer (Algorithm 4).

    Args:
        params: model parameters
        loss_fn: callable returning scalar loss (no arguments)
        n_iterations: number of NNCG iterations (K)
        sketch_size: Nyström sketch size (s)
        precond_update_freq: frequency of preconditioner updates (F)
        mu: damping parameter
        cg_tol: CG tolerance (epsilon)
        cg_max_iter: max CG iterations (M)
        eta: initial step size for Armijo
        armijo_alpha: Armijo sufficient decrease parameter
        armijo_beta: Armijo backtracking factor
    """

    def __init__(self, params, loss_fn, n_iterations=2000, sketch_size=60,
                 precond_update_freq=20, mu=1e-2, cg_tol=1e-16, cg_max_iter=1000,
                 eta=1.0, armijo_alpha=0.1, armijo_beta=0.5):
        self.params = list(params)
        self.loss_fn = loss_fn
        self.n_iterations = n_iterations
        self.sketch_size = sketch_size
        self.precond_update_freq = precond_update_freq
        self.mu = mu
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.eta = eta
        self.armijo_alpha = armijo_alpha
        self.armijo_beta = armijo_beta

        self.device = self.params[0].device if self.params else torch.device('cpu')
        self.p = sum(param.numel() for param in self.params)

        self.U = None
        self.Lambda_hat = None
        self.d_prev = torch.zeros(self.p, device=self.device)

    def _get_flat_grad(self):
        """Get flat gradient vector."""
        grads = []
        for param in self.params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros(param.numel(), device=self.device))
        return torch.cat(grads)

    def _hvp(self, v):
        """Compute Hessian-vector product H @ v using autograd."""
        loss = self.loss_fn()
        grad = torch.autograd.grad(loss, self.params, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad])
        hvp = torch.autograd.grad(flat_grad, self.params,
                                   grad_outputs=v.view_as(flat_grad),
                                   retain_graph=False)
        return torch.cat([h.view(-1) for h in hvp]).detach()

    def step(self):
        """Run NNCG for n_iterations steps."""
        losses = []
        grad_norms = []

        for k in range(self.n_iterations):
            # Compute loss and gradient
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

            loss = self.loss_fn()
            loss.backward()
            grad = self._get_flat_grad().detach()
            grad_norm = torch.linalg.norm(grad).item()

            losses.append(loss.item())
            grad_norms.append(grad_norm)

            # Update Nyström preconditioner every F iterations
            if k % self.precond_update_freq == 0:
                self.U, self.Lambda_hat = randomized_nystrom_approximation(
                    self._hvp, self.p, self.sketch_size, self.device
                )

            # Compute Newton step using NyströmPCG
            d_k = nystrom_pcg(
                self._hvp, grad, self.d_prev,
                self.U, self.Lambda_hat,
                self.mu, self.cg_tol, self.cg_max_iter, self.device
            )
            self.d_prev = d_k.detach()

            # Armijo line search: search direction is -d_k
            direction = -d_k

            def loss_fn_no_grad():
                with torch.no_grad():
                    return self.loss_fn()

            eta_k = armijo_line_search(
                loss_fn_no_grad, self.params, grad, direction,
                self.eta, self.armijo_alpha, self.armijo_beta
            )

            if eta_k == 0.0:
                # No progress, skip update
                continue

            # Update parameters: w_{k+1} = w_k - eta_k * d_k
            offset = 0
            for p in self.params:
                numel = p.numel()
                p.data.add_(eta_k * direction[offset:offset + numel].view_as(p.data))
                offset += numel

        return losses, grad_norms
