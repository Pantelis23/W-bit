import subprocess
import os
import sys

def run_smoke_tests():
    print("=== Phase 1.8 Smoke Tests ===")
    python_exe = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    
    # Helper to run cmd
    def run(cmd):
        print(f"\n>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=project_root)

    # Orchestrated smoke
    run([python_exe, 'run_phase1.py', '--smoke', '--run_expB_grid'])

    # Optional lightweight grid sanity (keeps coverage even if orchestrator skips grid flag)
    run([python_exe, 'experiments/exp_b_weight_noise_grid.py', '--trials', '3'])

if __name__ == "__main__":
    run_smoke_tests()
