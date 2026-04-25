import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from wbit.analog_network import AnalogWbitNetwork

def debug_xor():
    print("DEBUG: Binary H=0 XOR Test")
    # Setup H=0 net (Direct A,B -> Y)
    # R=2
    net = AnalogWbitNetwork(3, 2, mode='binary')
    A, B, Y = 0, 1, 2
    
    # Try to manually construct a solution or see what random search finds
    # Truth table:
    # 0,0 -> 0
    # 0,1 -> 1
    # 1,0 -> 1
    # 1,1 -> 0
    
    # In binary mode step():
    # best = max(range(R), key=lambda idx: updated_dist[idx])
    # bin_state[best] = 1.0
    
    # Inputs are clamped via set_local_weights.
    # bias[val] = 10.0
    
    # If connections are learned...
    # Can we solve XOR with direct connections + bias?
    # Y_input = Theta[Y,A][r_y][s_a] + Theta[Y,B][r_y][s_b] + theta[Y][r_y]
    
    # If input A is 0, s_a = [1, 0]. If A is 1, s_a = [0, 1].
    # Let's say Theta[Y,A] is W_a, Theta[Y,B] is W_b.
    # Score for Y=0: S_0 = W_a[0][A] + W_b[0][B] + b_0
    # Score for Y=1: S_1 = W_a[1][A] + W_b[1][B] + b_1
    # Y = 1 if S_1 > S_0
    
    # Delta S = S_1 - S_0
    # Delta S = (W_a[1][A] - W_a[0][A]) + (W_b[1][B] - W_b[0][B]) + (b_1 - b_0)
    # Let w_a(A) = W_a[1][A] - W_a[0][A]
    # Let w_b(B) = W_b[1][B] - W_b[0][B]
    # Let bias = b_1 - b_0
    # Condition: w_a(A) + w_b(B) + bias > 0
    
    # A, B are indices 0 or 1.
    # Case 0,0 (Target 0): w_a(0) + w_b(0) + bias < 0
    # Case 0,1 (Target 1): w_a(0) + w_b(1) + bias > 0
    # Case 1,0 (Target 1): w_a(1) + w_b(0) + bias > 0
    # Case 1,1 (Target 0): w_a(1) + w_b(1) + bias < 0
    
    # Sum ineq 2 and 3: w_a(0) + w_b(1) + w_a(1) + w_b(0) + 2*bias > 0
    # Sum ineq 1 and 4: w_a(0) + w_b(0) + w_a(1) + w_b(1) + 2*bias < 0
    # Contradiction. LHS are identical.
    # IMPOSSIBLE for deterministic binary threshold.
    
    # So why did Exp C report success?
    # Hypothesis: The 'inputs' in Exp C loop are set via set_local_weights.
    # "bias[val] = 10.0".
    # This sets the local theta for A and B.
    # The network runs for `max_steps`.
    # A and B are cells 0 and 1. They are updated too!
    # Unless they are clamped?
    # In `run_until_stable`, ALL cells are updated.
    # If A and B update, they might change state based on feedback from Y?
    # But A and B have no inputs from Y (unless symmetric?).
    # connections: (Y, A), (Y, B). Directed.
    # So A and B only depend on their self-bias.
    # If bias is strong (10.0), A and B effectively clamped.
    
    # Re-check logs. "Mean Best MSE=0.0000, Mean Acc=0.50".
    # Acc=0.50 confirms it FAILED to solve XOR (random guess).
    # Success=1.00 confirms `converged` was True.
    # `converged` = `global_best_mse < 0.02`.
    # So MSE was 0.0 but Acc was 0.5.
    # How?
    # target_dist = [0.0]*R; target_dist[target_val] = 1.0
    # output_dist = net.state[output_index]
    # err = sum((o - t)**2 ...)
    # If state is binary hard [0, 1], and target is [0, 1], err is 0.
    # If state is [1, 0], and target [0, 1], err is 2.0.
    
    # If Mean MSE = 0.0 over 4 samples, then ALL samples matched.
    # Then Acc MUST be 1.0.
    # Contradiction remains.
    
    # Hypothesis: The `population` loop in Exp C might be buggy.
    # `evaluate_analog` returns `mse, acc, steps`.
    # `scored_pop.append((mse, net))`
    # `if mse < global_best_mse: ... global_best_net = net`
    # Maybe `evaluate_analog` is stochastic (temp=0.3) and `global_best_net` got lucky once (MSE=0) but failed on re-eval?
    # `final_mse, final_acc, avg_steps = evaluate_analog(global_best_net, ...)`
    # The log reports `b_mse` (from best_trial_result which comes from result tuple).
    # result = (converged, epochs, global_best_mse, rcp, final_acc)
    # So `b_mse` is the *lucky* MSE during search.
    # `b_acc` is the *validation* Acc after search.
    # If `evaluate_analog` is noisy, search finds a lucky noise seed -> MSE~0.
    # Re-run -> Acc~0.5.
    # Binary mode uses `softmax` in `step` if `noise > 0` or `temp > 0`.
    # `evaluate_analog` sets `temperature=0.3`.
    # `AnalogWbitNetwork.step`:
    #   target_dist = softmax(..., temp)
    #   if mode == 'binary':
    #      best = max(range(R), key=lambda idx: updated_dist[idx])
    # The `updated_dist` depends on `target_dist` which depends on `temp`.
    # So Binary mode IS stochastic due to `temp`.
    # With temp=0.3, it's possible to flip bits randomly.
    # So the "solution" was just a lucky sequence of flips that matched targets?
    
    print("Investigation complete.")

if __name__ == "__main__":
    debug_xor()
