import math

def run_energy_profiler():
    print("=== W-BIT LEVEL 3 ENERGY PROFILER ===")
    print("Estimating femtojoules per operation (fJ/op) for state transitions.\n")

    # --- Physical Constants (Estimates for Oxide/Memristor crossbars) ---
    # Voltage and time
    V_read = 0.5        # Read voltage [V]
    V_spike = 1.0       # Threshold voltage for tau-bit trigger [V]
    T_step = 1e-9       # Physical time per integration step dt (1 ns)
    
    # Resistance / Capacitance
    R_on = 1e6          # Crossbar memristor ON resistance (1 MOhm)
    R_off = 1e8         # Crossbar memristor OFF resistance (100 MOhm)
    C_mem = 10e-15      # Membrane capacitance per cell (10 fF)
    
    # Static Power Components
    P_comparator = 5e-6 # Power per active ADC comparator (5 uW)
    
    # Scenario definitions
    cycles = 50         # Number of integration dt steps required to settle
    
    # --- Energy Calculations ---
    
    print(f"Base Parameters: V_read={V_read}V, dt={T_step*1e9}ns, R_on={R_on/1e6}MOhm, C_mem={C_mem*1e15}fF\n")

    # 1. High-Precision Mode (R=9)
    # 8 comparators active. No beta-bias. Instantaneous read (1 step) vs temporal (50 steps).
    static_power_R9 = 8 * P_comparator
    energy_static_R9_instant = static_power_R9 * T_step
    energy_static_R9_temporal = static_power_R9 * (cycles * T_step)
    
    # Dynamic power: Charging the membrane capacitor and flowing through crossbar
    I_read_avg = V_read / R_on # simplified worst-case
    P_dynamic = V_read * I_read_avg
    energy_dynamic_instant = P_dynamic * T_step
    energy_dynamic_temporal = P_dynamic * (cycles * T_step)
    
    total_R9_instant = (energy_static_R9_instant + energy_dynamic_instant) * 1e15 # convert to fJ
    total_R9_temporal = (energy_static_R9_temporal + energy_dynamic_temporal) * 1e15
    
    print("--- High-Precision Mode (R=9) ---")
    print(f"Instantaneous Read (Level 2):    {total_R9_instant:6.2f} fJ / op")
    print(f"Temporal Integration (Level 3):  {total_R9_temporal:6.2f} fJ / op")
    
    # 2. Phi-Mode Survival (R=3)
    # 2 comparators active.
    static_power_R3 = 2 * P_comparator
    
    # Beta-bias power cost
    # Beta injects a constant bias current to 3 active centers
    I_beta = 50e-9 # 50 nA targeted bias current
    P_beta = V_read * I_beta * 3 # Applied to 3 states
    
    energy_static_R3_temporal = (static_power_R3 + P_beta) * (cycles * T_step)
    
    total_R3_temporal = (energy_static_R3_temporal + energy_dynamic_temporal) * 1e15
    
    print("\n--- Phi-Mode Survival (R=3) ---")
    print(f"Temporal + Beta Bias (Level 3):  {total_R3_temporal:6.2f} fJ / op")
    
    # Analysis
    print("\n--- Structural Analysis ---")
    savings = total_R9_temporal - total_R3_temporal
    print(f"Delta entering Survival Mode:   -{savings:.2f} fJ / op")
    
    if savings > 0:
        print("CONCLUSION: Level 3 Survival Mode uses LESS energy than maintaining High-Precision.")
        print("Why? Dropping 6 ADC comparators saves more static power than the Beta-bias consumes.")
    else:
        print("CONCLUSION: Level 3 Survival Mode uses MORE energy.")

if __name__ == "__main__":
    run_energy_profiler()
