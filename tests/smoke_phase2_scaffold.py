import os
import subprocess
import sys

def run_smoke_phase2():
    print("=== Phase 2 Scaffold Smoke ===")
    python_exe = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))

    def run(cmd):
        print(f"\n>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=project_root)

    run([python_exe, 'run_phase2.py', '--smoke', '--run_expB_grid'])

    wbit_dir = os.path.join(project_root, 'results', 'phase2', 'wbit')
    binary_dir = os.path.join(project_root, 'results', 'phase2', 'binary')
    report_path = os.path.join(project_root, 'results', 'phase2', 'phase2_report.csv')

    assert os.path.isdir(wbit_dir), "Missing wbit results dir"
    assert os.path.isdir(binary_dir), "Missing binary results dir"
    assert os.path.exists(report_path), "Missing phase2 report"

    with open(report_path, 'r') as f:
        content = f.read()
        assert 'wbit' in content and 'binary' in content, "Report missing mode entries"

    # Delta summary and plots
    run([python_exe, 'analysis/phase2_delta.py'])
    delta_path = os.path.join(project_root, 'results', 'phase2', 'phase2_delta_summary.csv')
    assert os.path.exists(delta_path), "Missing delta summary CSV"
    run([python_exe, 'analysis/plot_phase2_delta.py'])

    # Frontier summary and plots
    run([python_exe, 'analysis/phase2_frontier.py'])
    frontier_path = os.path.join(project_root, 'results', 'phase2', 'phase2_frontier_summary.csv')
    assert os.path.exists(frontier_path), "Missing frontier summary CSV"
    with open(frontier_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1, "Frontier summary has no data rows"
    run([python_exe, 'analysis/plot_phase2_frontier.py'])

    # Pareto summary and plots
    run([python_exe, 'analysis/phase2_pareto.py'])
    pareto_w = os.path.join(project_root, 'results', 'phase2', 'phase2_pareto_wbit.csv')
    pareto_b = os.path.join(project_root, 'results', 'phase2', 'phase2_pareto_binary.csv')
    assert os.path.exists(pareto_w), "Missing Pareto wbit CSV"
    assert os.path.exists(pareto_b), "Missing Pareto binary CSV"
    run([python_exe, 'analysis/plot_phase2_pareto.py'])

    # Steps audit
    run([python_exe, 'analysis/phase2_steps_audit.py'])
    steps_audit = os.path.join(project_root, 'results', 'phase2', 'phase2_steps_audit.csv')
    assert os.path.exists(steps_audit), "Missing steps audit CSV"
    with open(steps_audit, newline='') as f:
        import csv
        rows = list(csv.DictReader(f))
        mean_steps_wbit = None
        has_adaptive = False
        for r in rows:
            if r.get('experiment') == 'expA' and r.get('mode') == 'wbit':
                try:
                    mean_steps_wbit = float(r.get('mean_steps', 0.0))
                except (TypeError, ValueError):
                    mean_steps_wbit = 0.0
            if r.get('mode') == 'binary' and r.get('experiment') == 'expA':
                zero_all = float(r.get('zero_step_fraction_all', 1.0) or 1.0)
                zero_success = float(r.get('zero_step_fraction_success', 1.0) or 1.0)
                assert zero_all < 0.1, "Binary expA zero-step fraction (all) too high"
                assert zero_success == 0.0, "Binary expA zero-step fraction (success) too high"
                if mean_steps_wbit is not None and mean_steps_wbit > 0:
                    mean_steps_binary = float(r.get('mean_steps', 0.0) or 0.0)
                    assert mean_steps_binary >= 0.3 * mean_steps_wbit, "Binary expA mean steps too low vs wbit"
            if r.get('mode') == 'adaptive' and r.get('experiment') == 'expA':
                has_adaptive = True
                zero_success_a = float(r.get('zero_step_fraction_success', 1.0) or 1.0)
                mean_steps_a = float(r.get('mean_steps', 0.0) or 0.0)
                assert zero_success_a == 0.0, "Adaptive expA zero-step fraction (success) too high"
                assert mean_steps_a > 0, "Adaptive expA mean steps must be > 0"
    run([python_exe, 'analysis/plot_phase2_steps_audit.py'])
    assert has_adaptive, "Adaptive mode missing from steps audit"

    # No pandas check
    run([python_exe, 'tests/test_no_pandas_imports.py'])

if __name__ == "__main__":
    run_smoke_phase2()
