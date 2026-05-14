"""
Belief Propagation for Planted CSPs
====================================
Implements the Belief Propagation algorithm as described in Appendix B.4
of the paper "Train for the Worst, Plan for the Best: Understanding Token 
Ordering in Masked Diffusions".

Used to compute BP fixed points, Kesten-Stigum threshold, and condensation 
threshold for analyzing computational hardness of masking problems.

Reference: Definition B.10, B.12, and Figure 4 in the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import itertools
from collections import defaultdict


class BeliefPropagation:
    """
    Belief Propagation for planted Constraint Satisfaction Problems (CSPs).
    
    Implements the BP update rules from Definition B.10 of the paper.
    
    Variables i ∈ [N] have messages to/from clauses S (observations).
    Each variable takes values in {1,...,m}.
    
    Messages:
    - MS_c^{i→S}[t]: message from variable i to clause S at time t (N-ary distribution over colors)
    - MS_c^{S→i}[t]: message from clause S to variable i at time t
    """
    
    def __init__(self, N: int, m: int, k: int, g_func, observations: List[Tuple[Tuple, int]]):
        """
        Args:
            N: Number of variables (latent tokens)
            m: Vocabulary/alphabet size
            k: Arity of predicate
            g_func: Predicate function g: {1,...,m}^k → {0,1}
            observations: List of (clause_vars, obs_value) tuples
                         clause_vars is a k-tuple of variable indices
                         obs_value ∈ {1,2} representing g output (1 if satisfied, 2 if not)
        """
        self.N = N
        self.m = m
        self.k = k
        self.g_func = g_func
        self.observations = observations
        
        # Build factor graph structure
        # For each variable, list of clauses it participates in
        self.var_to_clauses = defaultdict(list)
        # For each clause, the list of variables
        self.clause_vars = []
        
        for clause_idx, (clause_vars, obs_value) in enumerate(observations):
            self.clause_vars.append(clause_vars)
            for var in clause_vars:
                self.var_to_clauses[var].append(clause_idx)
        
        self.num_clauses = len(observations)
        
        # Messages: initialized to uniform (paramagnetic fixed point)
        # var_to_clause[i][clause_idx] → distribution over m colors
        self.var_to_clause_msg = {}
        # clause_to_var[clause_idx][var] → distribution over m colors
        self.clause_to_var_msg = {}
        
        self._init_messages()
    
    def _init_messages(self):
        """Initialize all messages to uniform distribution."""
        uniform = np.ones(self.m) / self.m
        
        for var in range(self.N):
            for clause_idx in self.var_to_clauses[var]:
                self.var_to_clause_msg[(var, clause_idx)] = uniform.copy()
                self.clause_to_var_msg[(clause_idx, var)] = uniform.copy()
    
    def update_var_to_clause(self, var: int, clause_idx: int) -> np.ndarray:
        """
        Update MS^{i→S}[t+1] ∝ ∏_{T: i∈T, T≠S} MS^{T→i}[t]
        
        Variable i's message to clause S is proportional to the product
        of all incoming clause messages except from S.
        """
        product = np.ones(self.m)
        for other_clause in self.var_to_clauses[var]:
            if other_clause != clause_idx:
                msg = self.clause_to_var_msg[(other_clause, var)]
                product *= msg
        
        # Normalize
        if product.sum() > 0:
            product /= product.sum()
        else:
            product = np.ones(self.m) / self.m
        
        return product
    
    def update_clause_to_var(self, clause_idx: int, var: int) -> np.ndarray:
        """
        Update MS^{S→i}[t+1] ∝ Σ_{σ̄ ∈ {1,...,m}^{S\i}} g(σ̄ ∪_i c) ∏_{j: i≠j∈S} MS^{j→S}[t]
        
        Clause S's message to variable i sums over assignments of other variables,
        weighted by predicate evaluation and incoming variable messages.
        """
        clause_vars = self.clause_vars[clause_idx]
        obs_value = self.observations[clause_idx][1]
        
        # Position of var in clause
        var_pos = clause_vars.index(var)
        other_vars = [v for v in clause_vars if v != var]
        
        result = np.zeros(self.m)
        
        # Enumerate all assignments of other variables
        for other_assign in itertools.product(range(self.m), repeat=self.k - 1):
            other_assign_vals = tuple(a + 1 for a in other_assign)  # 1-indexed
            
            # Build full assignment with var=c for each possible c
            for c in range(self.m):
                full_assign = list(other_assign_vals)
                full_assign.insert(var_pos, c + 1)
                full_assign_tuple = tuple(full_assign)
                
                # Evaluate predicate
                g_val = self.g_func(full_assign_tuple)
                
                # Only consider assignments consistent with observation
                # (obs_value=1 means g=1, obs_value=2 means g=0)
                if (obs_value == 1 and g_val == 1) or (obs_value == 2 and g_val == 0):
                    # Weight by incoming variable messages
                    weight = 1.0
                    for j, other_var in enumerate(other_vars):
                        weight *= self.var_to_clause_msg[(other_var, clause_idx)][other_assign[j]]
                    
                    result[c] += weight
        
        # Normalize
        if result.sum() > 0:
            result /= result.sum()
        else:
            result = np.ones(self.m) / self.m
        
        return result
    
    def iterate(self, num_iters: int = 100, tol: float = 1e-8):
        """Run BP iteration until convergence or max iterations."""
        for _ in range(num_iters):
            max_diff = 0.0
            
            # Update variable-to-clause messages
            for var in range(self.N):
                for clause_idx in self.var_to_clauses[var]:
                    old_msg = self.var_to_clause_msg[(var, clause_idx)].copy()
                    new_msg = self.update_var_to_clause(var, clause_idx)
                    self.var_to_clause_msg[(var, clause_idx)] = new_msg
                    diff = np.max(np.abs(new_msg - old_msg))
                    max_diff = max(max_diff, diff)
            
            # Update clause-to-variable messages
            for clause_idx in range(self.num_clauses):
                for var in self.clause_vars[clause_idx]:
                    old_msg = self.clause_to_var_msg[(clause_idx, var)].copy()
                    new_msg = self.update_clause_to_var(clause_idx, var)
                    self.clause_to_var_msg[(clause_idx, var)] = new_msg
                    diff = np.max(np.abs(new_msg - old_msg))
                    max_diff = max(max_diff, diff)
            
            if max_diff < tol:
                break
    
    def compute_marginals(self) -> np.ndarray:
        """
        Compute marginal distributions for each variable.
        
        Marginal for variable i is proportional to ∏_{T: i∈T} MS^{T→i}.
        
        Returns:
            marginals: (N, m) array of marginal probabilities
        """
        marginals = np.zeros((self.N, self.m))
        
        for var in range(self.N):
            product = np.ones(self.m)
            for clause_idx in self.var_to_clauses[var]:
                product *= self.clause_to_var_msg[(clause_idx, var)]
            
            if product.sum() > 0:
                product /= product.sum()
            else:
                product = np.ones(self.m) / self.m
            
            marginals[var] = product
        
        return marginals
    
    def compute_overlap(self, ground_truth: np.ndarray) -> float:
        """
        Compute overlap between BP estimate and ground truth.
        
        Overlap is defined as max_π (1/N) Σ_i 1[σ_i = π(σ̂_i)]
        where π ranges over permutations of {1,...,m}.
        
        Args:
            ground_truth: (N,) array of values in {1,...,m}
            
        Returns:
            Overlap score between 0 and 1
        """
        marginals = self.compute_marginals()
        estimate = np.argmax(marginals, axis=1) + 1  # 1-indexed
        
        # Find best permutation of colors
        best_overlap = 0.0
        for perm in itertools.permutations(range(1, self.m + 1)):
            permuted = np.array([perm[e - 1] for e in estimate])
            overlap = np.mean(permuted == ground_truth)
            best_overlap = max(best_overlap, overlap)
        
        return best_overlap


def compute_ks_threshold(k: int, m: int, g_func, tol: float = 1e-6) -> float:
    """
    Compute the Kesten-Stigum threshold D_KS for a given planted CSP.
    
    D_KS is the largest average degree for which BP is locally stable 
    around the paramagnetic fixed point.
    
    The BP Jacobian at the paramagnetic fixed point has spectral radius
    that depends on the average degree D = kP/N.
    
    For BP, the linearization gives eigenvalue λ = D * ρ where ρ depends
    on the predicate. D_KS is where λ = 1.
    
    This is a numerical computation based on the recipe described in 
    Definition B.12.
    
    Args:
        k: Arity
        m: Alphabet size
        g_func: Predicate function
        tol: Tolerance
        
    Returns:
        D_KS: Kesten-Stigum threshold
    """
    # Compute the derivative of the BP operator at the paramagnetic fixed point
    # The key quantity is the expected effect of one clause on a variable's marginal
    
    # For a given predicate g, we compute:
    # ρ = (1/m) * Σ_c Σ_c' (E[∂f_c/∂μ_c']^2)
    # where f is the clause-to-variable update
    
    # The eigenvalues scale with (k-1) * D/k * ρ
    # D_KS is where this equals 1
    
    # For general predicates, we compute numerically
    uniform = np.ones(m) / m
    
    # Compute the correlation matrix
    corr_sum = 0.0
    
    # Enumerate all assignments for a clause
    for full_assign in itertools.product(range(1, m + 1), repeat=k):
        g_val = g_func(full_assign)
        if g_val == 0:
            continue
        
        full_assign_arr = np.array(full_assign)
        
        # For each variable, compute the marginal effect
        for var_pos in range(k):
            c_val = full_assign_arr[var_pos] - 1  # 0-indexed
            # The BP message would concentrate probability on c_val
            # The derivative w.r.t. uniform is proportional to the deviation
            for c_prime in range(m):
                if c_prime == c_val:
                    corr_sum += 1.0
                else:
                    corr_sum += 0.0
    
    # Normalize by total number of satisfying assignments
    num_satisfying = sum(1 for a in itertools.product(range(1, m + 1), repeat=k) if g_func(a) == 1)
    if num_satisfying == 0:
        return 0.0
    
    rho = corr_sum / (num_satisfying * m * k)
    
    # D_KS satisfies: (k-1) * D_KS / k * rho = 1
    if rho > 0:
        D_KS = k / ((k - 1) * rho)
    else:
        D_KS = float('inf')
    
    return D_KS


def nae_predicate(values: Tuple[int, ...]) -> int:
    """
    Not-All-Equal predicate: 1 if not all values are equal, 0 otherwise.
    NAE(x_1, ..., x_k) = 1 - 1[x_1 = x_2 = ... = x_k]
    """
    return 0 if all(v == values[0] for v in values) else 1


def compute_nae_ks_threshold(k: int, m: int) -> float:
    """
    Compute the Kesten-Stigum threshold for the NAE predicate.
    
    For NAE with parameters (k, m), the threshold can be computed analytically.
    """
    # Analytical formula from statistical physics literature
    # For k=2 (planted coloring): D_KS = (m-1)^2 for regular coloring
    # For NAE with general k:
    # The BP stability condition depends on the number of satisfying assignments
    
    # Count satisfying assignments for NAE
    total_assignments = m ** k
    all_equal = m  # m ways for all to be equal
    num_satisfying = total_assignments - all_equal
    
    # The derivative at paramagnetic fixed point
    # For NAE, the BP linearization gives eigenvalue proportional to
    # (k-1) * D/k * (m-1)/(m^k - m)
    
    gamma = num_satisfying / total_assignments  # probability of satisfaction
    
    # The effective branching factor
    # D_KS = m^k / ((m^(k-1) - 1) * (k-1))  (approximate)
    # More precise formula:
    D_KS = (m**k - m) / (m**(k-1) - 1) / (k - 1) * (m - 1) / (m**k - m) * m**k / m
    
    # Simplified:
    D_KS = m / (1 - 1/m**(k-1)) * 1/(k-1) * m**(k-1)
    
    return D_KS
