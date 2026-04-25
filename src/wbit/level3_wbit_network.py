import math
import random
from .analog_network import AnalogWbitNetwork

class Level3WbitNetwork(AnalogWbitNetwork):
    """
    Level 3 W-bit Architecture: Adaptive Logic Substrate
    
    This preserves the Level 2 ML-friendly foundation (R x R compatibility matrices)
    but introduces dynamic runtime adjustment of representational capacity (R_effective).
    
    The physical cell has a maximum capacity (R_max). The OS dynamically 
    schedules R_effective based on noise, complexity, and latency targets.
    This simulates the hardware adjusting its DAC/ADC comparator thresholds on the fly.
    """
    def __init__(self, num_cells, R_max=9, mode="adaptive"):
        super().__init__(num_cells, R=R_max, mode=mode)
        self.R_max = self.R
        self.R_eff = self.R_max
        self.noise_estimate = 0.0
        
    def schedule_capacity(self, task_complexity: float, environmental_noise: float):
        """
        The Level 3 Control Law.
        Dynamically adjusts R_effective based on runtime conditions.
        
        A physical switch: wider spacing = robust to noise, tighter spacing = denser logic.
        """
        self.noise_estimate = environmental_noise
        
        # High noise or low complexity -> robust phi-mode (R=3 or 5 depending on symmetry)
        if environmental_noise > 0.4 or task_complexity < 0.3:
            self.R_eff = 3  # Most robust (Binary + Neutral)
        elif environmental_noise > 0.15:
            self.R_eff = 5  # Phi-ish mode
        elif environmental_noise > 0.05:
            self.R_eff = 7
        else:
            self.R_eff = self.R_max  # Full precision (Dense Matrix mode)
            
        print(f"[OS Scheduler] Noise: {environmental_noise:.2f} | Complexity: {task_complexity:.2f} -> Shifted R_eff to {self.R_eff}")
        
    def _project_state(self, state_dist):
        """
        Simulates the hardware adjusting its DAC/ADC comparator thresholds.
        Projects the continuous R_max probability distribution down to R_eff legal states.
        It groups the probability mass into R_eff bins and centers it.
        """
        if self.R_eff == self.R_max:
            return state_dist
            
        projected = [0.0] * self.R_max
        bin_size = self.R_max / self.R_eff
        
        for r in range(self.R_max):
            # Find which coarse bin this r belongs to
            bin_idx = min(int(r / bin_size), self.R_eff - 1)
            # Map the bin back to a representative central index in R_max space
            center_r = int((bin_idx + 0.5) * bin_size)
            center_r = min(center_r, self.R_max - 1)
            
            projected[center_r] += state_dist[r]
            
        return projected

    def step(self, temperature=1.0, noise_level=None, dt=0.5):
        """
        Level 3 Execution: Enforces R_eff via active energy modulation.
        Integrates KV-Compaction Beta Bias, H-Neuron Suppression, and Calibration Margins.
        """
        if noise_level is None:
            noise_level = self.noise_estimate
            
        new_states = []
        
        for i in range(self.num_cells):
            # 1. Base Energy Landscape (Local preference)
            input_drive = list(self.theta[i])
            
            # 2. Pairwise Compatibility (Matrix math)
            for (target, source), matrix in self.Theta.items():
                if target == i:
                    source_state = self.state[source]
                    for r_i in range(self.R_max):
                        dot_prod = 0.0
                        for r_j in range(self.R_max):
                            dot_prod += matrix[r_i][r_j] * source_state[r_j]
                        input_drive[r_i] += dot_prod
            
            # 3. Inject Environmental Noise
            if noise_level > 0:
                for r in range(self.R_max):
                    input_drive[r] += random.gauss(0, noise_level)
            
            # 4. LEVEL 3 ACTIVE MODULATION (When OS restricts capacity)
            if self.R_eff < self.R_max:
                bin_size = self.R_max / self.R_eff
                allowed_centers = [min(int((b + 0.5) * bin_size), self.R_max - 1) for b in range(self.R_eff)]
                
                for r in range(self.R_max):
                    if r in allowed_centers:
                        # KV COMPACTION: Beta Bias
                        # Restore the lost "mass" from the dropped adjacent states by boosting the center
                        beta_bias = math.log(bin_size) * 2.0 # Proportional mass correction
                        input_drive[r] += beta_bias
                    else:
                        # H-NEURON SUPPRESSION: Active Inhibition
                        # Boundary states are prone to hallucinating under noise. Suppress them aggressively.
                        input_drive[r] -= 10.0 # Massive negative energy penalty
                
                # CALIBRATION LOSS: Confidence Margin
                # Find the top 2 states. If the gap between them is less than the margin, 
                # the noise is too high. Flatten the distribution to abstain rather than hallucinate.
                sorted_drives = sorted(input_drive, reverse=True)
                margin = 1.0 # The Delta gap required to confidently settle
                if (sorted_drives[0] - sorted_drives[1]) < margin:
                    temperature *= 5.0 # Spike the temperature to blur the output (Uncertainty)
            
            # 5. Physical settling (Softmax)
            target_dist = self.softmax(input_drive, temperature)
            
            # 6. Apply physical inertia (Low Pass Filter)
            current_dist = self.state[i]
            updated_dist = []
            for r in range(self.R_max):
                val = current_dist[r] + dt * (target_dist[r] - current_dist[r])
                updated_dist.append(val)
                
            new_states.append(updated_dist)
            
        self.state = new_states
