import math
import random
from typing import Iterable, List, Sequence, Tuple


def _to_list_vector(x: Sequence[float]) -> List[float]:
    """Convert any sequence / numpy array into a plain Python list of floats."""
    try:
        if hasattr(x, "tolist"):
            return list(x.tolist())
    except Exception:
        pass
    return [float(v) for v in x]


def _to_list_matrix(W: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert a matrix-like container to list of lists."""
    try:
        if hasattr(W, "tolist"):
            W = W.tolist()
    except Exception:
        pass
    return [ _to_list_vector(row) for row in W ]


def _argmax(values: Sequence[float]) -> int:
    """Return the index of the maximum value."""
    best_idx = 0
    best_val = values[0]
    for idx in range(1, len(values)):
        if values[idx] > best_val:
            best_val = values[idx]
            best_idx = idx
    return best_idx


def _matmul(W: List[List[float]], x: List[float]) -> List[float]:
    """Simple matrix-vector multiply."""
    outputs = []
    for row in W:
        acc = 0.0
        for w_ij, x_j in zip(row, x):
            acc += w_ij * x_j
        outputs.append(acc)
    return outputs


def _mean_squared_error(a: List[List[float]], b: List[List[float]]) -> float:
    """Mean squared error between two batches of vectors."""
    total = 0.0
    count = 0
    for va, vb in zip(a, b):
        for xa, xb in zip(va, vb):
            diff = xa - xb
            total += diff * diff
            count += 1
    if count == 0:
        return 0.0
    return total / count


def quantize_weights(W: Sequence[Sequence[float]], R: int) -> Tuple[List[List[float]], List[float]]:
    """
    Quantize weights into R evenly spaced levels between -max_abs and +max_abs.

    Args:
        W: 2D weight matrix.
        R: number of discrete states.

    Returns:
        (quantized_weights, levels)
    """
    if R < 2:
        raise ValueError("R must be >= 2 for quantization")

    W_list = _to_list_matrix(W)
    max_abs = 0.0
    for row in W_list:
        for w in row:
            max_abs = max(max_abs, abs(w))
    if max_abs == 0.0:
        max_abs = 1e-6

    step = (2 * max_abs) / (R - 1)
    levels = [(-max_abs + i * step) for i in range(R)]

    def snap(value: float) -> float:
        best = levels[0]
        best_dist = abs(value - best)
        for lvl in levels[1:]:
            dist = abs(value - lvl)
            if dist < best_dist:
                best = lvl
                best_dist = dist
        return best

    quantized = []
    for row in W_list:
        quantized.append([snap(w) for w in row])

    return quantized, levels


def _add_weight_noise(W: List[List[float]], sigma: float) -> List[List[float]]:
    """Add Gaussian noise to each weight."""
    if sigma <= 0:
        return [list(row) for row in W]
    noisy = []
    for row in W:
        noisy.append([w + random.gauss(0.0, sigma) for w in row])
    return noisy


def _forward_batch(W: List[List[float]], x_batch: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
    """Forward a batch through a linear layer."""
    outputs = []
    preds = []
    for x in x_batch:
        out_vec = _matmul(W, x)
        outputs.append(out_vec)
        preds.append(_argmax(out_vec))
    return outputs, preds


def resolve_effective_params(mode: str, base_R: int, sigma: float, binary_force_R2: bool = False, adaptive_max_n: int = None) -> Tuple[int, int, str, str]:
    """
    Mirror the repo's mode/effective parameter logic for layer evaluations.
    Returns (R_effective, n_effective, mode_variant, net_mode).
    """
    if mode == 'binary':
        r_eff = 2 if binary_force_R2 else base_R
        n_eff = max(1, (r_eff - 1) // 2)
        variant = 'binary_strict_R2' if binary_force_R2 else 'binary_quantized'
        net_mode = 'binary' if r_eff == 2 else 'wbit'
        return r_eff, n_eff, variant, net_mode

    if mode == 'adaptive':
        score = 0
        if sigma is not None and sigma >= 0.5:
            score += 1
        n = 1 if score <= 0 else 2 if score == 1 else 3 if score == 2 else 4
        max_n = adaptive_max_n if adaptive_max_n is not None else max(1, (base_R - 1) // 2)
        n = max(1, min(n, max_n))
        r_eff = max(3, 2 * n + 1)
        return r_eff, n, f"adaptive_heuristic_n{n}", 'adaptive'

    r_eff = base_R
    n_eff = max(1, (r_eff - 1) // 2)
    return r_eff, n_eff, 'wbit', 'wbit'


def _compute_rcp(effective_R: int, steps_taken: int, n_active_cells: int) -> float:
    """
    Compute RCP using the repo's convention (steps taken, not steps converged).
    """
    if steps_taken < 1:
        steps_taken = 1
    c_int = effective_R * effective_R
    return (n_active_cells * steps_taken * c_int) / 1.0


def wbit_eval_layer(W: Sequence[Sequence[float]], x_batch: Iterable[Sequence[float]], mode: str, R: int, sigma: float, trials: int = 10, binary_force_R2: bool = False, adaptive_max_n: int = None) -> dict:
    """
    Quantize a real layer into W-bit states, inject noise, and measure degradation.

    Args:
        W: weight matrix (out_dim x in_dim)
        x_batch: iterable of input vectors
        mode: 'wbit', 'binary', or 'adaptive'
        R: configured radix
        sigma: Gaussian noise stddev applied to quantized weights
        trials: number of noisy trials to average
        binary_force_R2: enable strict binary ablation
        adaptive_max_n: clamp for adaptive n selection

    Returns:
        dict with success_rate, loss_delta, avg_rcp, R_effective, n_effective, mode_variant, net_mode
    """
    W_list = _to_list_matrix(W)
    x_list = [_to_list_vector(x) for x in x_batch]
    if not x_list:
        raise ValueError("x_batch must contain at least one input vector")

    baseline_outputs, baseline_preds = _forward_batch(W_list, x_list)

    total_success = 0.0
    total_loss = 0.0
    total_rcp = 0.0

    r_eff, n_eff, mode_variant, net_mode = resolve_effective_params(mode, R, sigma, binary_force_R2=binary_force_R2, adaptive_max_n=adaptive_max_n)

    for _ in range(trials):
        qW, _levels = quantize_weights(W_list, r_eff)
        noisyW = _add_weight_noise(qW, sigma)
        outputs, preds = _forward_batch(noisyW, x_list)

        matches = 0
        for pred, ref in zip(preds, baseline_preds):
            if pred == ref:
                matches += 1
        trial_success_rate = matches / len(x_list)

        loss_delta = _mean_squared_error(outputs, baseline_outputs)

        n_active_cells = len(W_list) * len(W_list[0]) if W_list and W_list[0] else len(W_list)
        rcp = _compute_rcp(r_eff, 1, n_active_cells)

        total_success += trial_success_rate
        total_loss += loss_delta
        total_rcp += rcp

    success_rate = total_success / trials if trials > 0 else 0.0
    avg_loss_delta = total_loss / trials if trials > 0 else 0.0
    avg_rcp = total_rcp / trials if trials > 0 else 0.0

    return {
        "success_rate": success_rate,
        "loss_delta": avg_loss_delta,
        "avg_rcp": avg_rcp,
        "mode": mode,
        "mode_variant": mode_variant,
        "net_mode": net_mode,
        "R": R,
        "R_effective": r_eff,
        "n_effective": n_eff,
        "sigma": sigma,
        "trials": trials,
    }
