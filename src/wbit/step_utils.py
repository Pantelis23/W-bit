def summarize_steps(success, steps_taken, max_steps, path_exists=True, converged=True):
    """
    Shared step/termination summary.
    Returns (steps, reason)
    """
    reason = "unknown"
    steps = int(steps_taken) if steps_taken is not None else 0

    if not path_exists:
        reason = "no_feasible_path"
    elif success:
        reason = "goal_reached"
    elif not converged and steps >= max_steps:
        reason = "convergence_fail"
    elif steps >= max_steps:
        reason = "max_steps"
    return steps, reason
