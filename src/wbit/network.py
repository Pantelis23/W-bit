import random
import math

class WDitNetwork:
    def __init__(self, num_cells, R):
        """
        Initialize the W-bit Network (Pure Python).
        
        Args:
            num_cells (int): Number of cells in the network.
            R (int): Radix (number of possible states per cell).
        """
        self.num_cells = num_cells
        self.R = R
        
        # State: list of integers [0, R-1]
        self.state = [random.randint(0, R-1) for _ in range(num_cells)]
        
        # Local preference (bias) theta: list of lists (N, R)
        self.theta = [[0.0] * R for _ in range(num_cells)]
        
        # Pairwise compatibility Theta: Dictionary (i, j) -> [[R x R]]
        # Sparse storage is easier. If (i, j) not in dict, weight is 0.
        self.Theta = {}

    def set_local_weights(self, cell_idx, weights):
        """
        Set local preference weights for a specific cell.
        """
        if len(weights) != self.R:
            raise ValueError(f"Weights must be of length {self.R}")
        self.theta[cell_idx] = list(weights)

    def reset_local_weights(self):
        self.theta = [[0.0] * self.R for _ in range(self.num_cells)]

    def set_interaction_weights(self, i, j, weights_matrix):
        """
        Set pairwise interaction weights between cell i and cell j.
        weights_matrix: List of lists [[...], [...]] size RxR
        """
        if len(weights_matrix) != self.R or any(len(row) != self.R for row in weights_matrix):
            raise ValueError(f"Interaction matrix must be {self.R}x{self.R}")
        self.Theta[(i, j)] = [list(row) for row in weights_matrix]

    def get_energy(self):
        """
        Calculate the global 'Compatibility'.
        """
        total = 0.0
        # Local term
        for i in range(self.num_cells):
            total += self.theta[i][self.state[i]]
            
        # Interaction term
        # Iterate over defined interactions
        for (i, j), matrix in self.Theta.items():
            total += matrix[self.state[i]][self.state[j]]
            
        return total

    def compute_scores(self):
        """
        Compute scores for every cell.
        Returns: list of lists [[score_r0, score_r1...], ...]
        """
        scores = []
        for i in range(self.num_cells):
            cell_scores = []
            for r in range(self.R):
                # Local bias
                s = self.theta[i][r]
                
                # Interaction input from neighbors
                # We need to sum Theta[i, j][r][state[j]] for all j
                # To do this efficiently with sparse Theta, we iterate neighbors?
                # Or just iterate all j? Since we stored sparse keys (i, j),
                # we can find relevant j for this i.
                pass 
                
                cell_scores.append(s)
            scores.append(cell_scores)
            
        # Add interactions
        # Optimization: Iterate over Theta keys instead of N*N
        for (i, j), matrix in self.Theta.items():
            # i is target, j is source
            # matrix[r_i][r_j]
            s_j = self.state[j]
            for r_i in range(self.R):
                scores[i][r_i] += matrix[r_i][s_j]
                
        return scores

    def step(self, mode='deterministic', temperature=1.0):
        scores = self.compute_scores()
        new_state = [0] * self.num_cells
        
        for i in range(self.num_cells):
            cell_scores = scores[i]
            
            if mode == 'deterministic':
                # Argmax
                best_r = 0
                best_val = cell_scores[0]
                for r in range(1, self.R):
                    if cell_scores[r] > best_val:
                        best_val = cell_scores[r]
                        best_r = r
                new_state[i] = best_r
                
            elif mode == 'stochastic':
                # Softmax
                max_val = max(cell_scores)
                shifted = [s - max_val for s in cell_scores]
                exps = [math.exp(s / temperature) for s in shifted]
                sum_exps = sum(exps)
                probs = [e / sum_exps for e in exps]
                
                # Sample
                r_val = random.random()
                cum_prob = 0.0
                selected = self.R - 1
                for r, p in enumerate(probs):
                    cum_prob += p
                    if r_val <= cum_prob:
                        selected = r
                        break
                new_state[i] = selected
                
        self.state = new_state
        return self.state

    def run_until_stable(self, max_steps=100, mode='deterministic', temperature=1.0):
        for _ in range(max_steps):
            old_state = list(self.state)
            self.step(mode=mode, temperature=temperature)
            if self.state == old_state and mode == 'deterministic':
                break