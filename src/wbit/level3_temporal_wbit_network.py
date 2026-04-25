import math
import random
from .analog_network import AnalogWbitNetwork

class Level3TemporalWbitNetwork(AnalogWbitNetwork):
    """
    Level 3 W-bit Architecture: Temporal & Adaptive Logic Substrate
    
    This preserves the Level 2 ML-friendly foundation (R x R compatibility matrices)
    and the Level 3 Adaptive R_effective scheduler.
    
    Crucially, it translates three ML concepts into true analog physical mechanics:
    1. KV Beta-Bias -> Local Threshold Sensitization
    2. H-Neurons -> Voltage Drain (Active Leak)
    3. Calibration Margin -> Temporal Integration (Tau-bit)
    """
    def __init__(self, num_cells, R_max=9, mode="adaptive"):
        super().__init__(num_cells, R=R_max, mode=mode)
        self.R_max = self.R
        self.R_eff = self.R_max
        self.noise_estimate = 0.0
        
        # Temporal State (Tau-bit Integration)
        self.V_m = [[0.0 for _ in range(self.R_max)] for _ in range(num_cells)]
        
    def schedule_capacity(self, task_complexity: float, environmental_noise: float):
        self.noise_estimate = environmental_noise
        if environmental_noise > 0.4 or task_complexity < 0.3:
            self.R_eff = 3 
        elif environmental_noise > 0.15:
            self.R_eff = 5 
        elif environmental_noise > 0.05:
            self.R_eff = 7
        else:
            self.R_eff = self.R_max 
            
    def _project_state(self, state_dist):
        if self.R_eff == self.R_max:
            return state_dist
        projected = [0.0] * self.R_max
        bin_size = self.R_max / self.R_eff
        for r in range(self.R_max):
            bin_idx = min(int(r / bin_size), self.R_eff - 1)
            center_r = int((bin_idx + 0.5) * bin_size)
            center_r = min(center_r, self.R_max - 1)
            projected[center_r] += state_dist[r]
        return projected

    def step_temporal(self, temperature=1.0, noise_level=None, dt=1.0, 
                      use_beta=True, use_h_neuron=True, use_margin=True):
        """
        Translated Physical Mechanisms:
        - Beta-Bias: Instead of adding energy to active states, we slightly lower their 
          integration threshold (make them easier to trigger).
        - H-Neurons: Instead of a massive -10 penalty that shatters the landscape, we apply 
          a faster voltage leak to boundary states, draining them over time.
        - Calibration Margin: Instead of spiking temperature instantly, we require the 
          voltage to integrate over a temporal window (tau). If the noise is high, the 
          voltage fluctuates and fails to reach the threshold within the window.
        """
        if noise_level is None:
            noise_level = self.noise_estimate
            
        new_states = []
        
        # Physical Hyperparameters for Translation
        tau_leak = 5.0       # Base voltage leak rate
        tau_h_leak = 1.0     # Aggressive leak for H-Neurons (Boundary states)
        V_threshold = 10.0   # The voltage required to trigger a state change (Tau-bit margin)
        beta_sensitization = 0.8 # Multiplier on threshold for active states
        
        for i in range(self.num_cells):
            input_current = list(self.theta[i])
            
            # 1. Accumulate Current from Crossbar
            for (target, source), matrix in self.Theta.items():
                if target == i:
                    source_state = self.state[source]
                    for r_i in range(self.R_max):
                        dot_prod = sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(self.R_max))
                        input_current[r_i] += dot_prod
                        
            # 2. Inject Environmental Noise
            if noise_level > 0:
                for r in range(self.R_max):
                    input_current[r] += random.gauss(0, noise_level)
            
            # 3. Integrate Voltage (Tau-bit Mechanics)
            allowed_centers = list(range(self.R_max))
            if self.R_eff < self.R_max:
                bin_size = self.R_max / self.R_eff
                allowed_centers = [min(int((b + 0.5) * bin_size), self.R_max - 1) for b in range(self.R_eff)]
            
            for r in range(self.R_max):
                # Physical Leak
                current_tau = tau_h_leak if (use_h_neuron and r not in allowed_centers) else tau_leak
                dv = -(self.V_m[i][r] / current_tau) * dt
                
                # Apply current
                self.V_m[i][r] += dv + (input_current[r] * dt)
                
                # Prevent negative voltage in this specific model
                self.V_m[i][r] = max(0.0, self.V_m[i][r])
            
            # 4. Check Thresholds (Calibration Margin translated to Voltage Margin)
            target_dist = list(self.state[i]) # Default to staying in current state (Hysteresis)
            
            if use_margin:
                # The state only changes if a specific voltage bucket crosses the physical threshold.
                for r in range(self.R_max):
                    # Beta-bias translated: Active centers have slightly lower thresholds
                    local_threshold = V_threshold
                    if use_beta and r in allowed_centers:
                        local_threshold *= beta_sensitization
                        
                    if self.V_m[i][r] >= local_threshold:
                        # SNAP to this state
                        target_dist = [1.0 if x == r else 0.0 for x in range(self.R_max)]
                        # Reset all voltages (Discharge)
                        self.V_m[i] = [0.0 for _ in range(self.R_max)]
                        break # First to cross wins
            else:
                # If we don't use the temporal margin, we just fallback to instant softmax
                target_dist = self.softmax(self.V_m[i], temperature)
            
            # 5. Apply Output Projection
            projected_dist = self._project_state(target_dist)
            new_states.append(projected_dist)
            
        self.state = new_states