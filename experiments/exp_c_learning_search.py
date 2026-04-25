import argparse
import sys
import os
import random
import csv
import copy
import operator
import statistics

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork

def add_noise_to_matrix(matrix, scale):
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            matrix[r][c] += random.gauss(0, scale)

def add_noise_to_vector(vector, scale):
    for i in range(len(vector)):
        vector[i] += random.gauss(0, scale)

def evaluate_analog(net, inputs, targets, input_indices, output_index, max_steps):
    R = net.R
    mse_total = 0.0
    correct_count = 0
    total_steps = 0
    
    for input_vals, target_val in zip(inputs, targets):
        net.reset_local_weights()
        for idx, val in zip(input_indices, input_vals):
            bias = [-5.0] * R
            bias[val] = 10.0
            net.set_local_weights(idx, bias)
            
        net.state = [[random.uniform(0.4, 0.6) for _ in range(R)] for _ in range(net.num_cells)]

        for i in range(net.num_cells):
            s = sum(net.state[i])
            net.state[i] = [x/s for x in net.state[i]]
            
        steps = net.run_until_stable(max_steps=max_steps, temperature=0.3, noise=0.0)
        total_steps += steps
        
        output_dist = net.state[output_index]
        target_dist = [0.0] * R
        target_dist[target_val] = 1.0
        
        err = sum((o - t)**2 for o, t in zip(output_dist, target_dist))
        mse_total += err
        
        predicted = output_dist.index(max(output_dist))
        if predicted == target_val:
            correct_count += 1
            
    return mse_total / len(inputs), correct_count / len(inputs), total_steps / len(inputs)

def create_random_net(num_cells, R, connections, h_indices, Y, mode):
    net = AnalogWbitNetwork(num_cells, R, mode=mode)
    scale = 2.0
    for target, source in connections:
        mat = [[random.gauss(0, scale) for _ in range(R)] for _ in range(R)]
        net.set_interaction_weights(target, source, mat)
    for cell in [Y] + h_indices:
        net.set_local_weights(cell, [random.gauss(0, scale) for _ in range(R)])
    return net

def mutate_net(net, connections, h_indices, Y, noise_scale):
    new_net = copy.deepcopy(net)
    for target, source in connections:
        mat = new_net.Theta[(target, source)]
        add_noise_to_matrix(mat, noise_scale)
    for cell in [Y] + h_indices:
        add_noise_to_vector(new_net.theta[cell], noise_scale)
    return new_net

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        header = ['trial', 'trial_seed', 'H', 'success', 'epochs', 'final_mse', 'inference_rcp', 'active_cells', 'R', 'population', 'elite_k', 'restarts', 'allow_direct_ab_to_y', 'best_search_mse', 'best_acc', 'max_epochs', 'mode', 'mode_variant', 'n_effective', 'R_effective']
        if write_header:
            writer.writerow(header)
            
        random.seed(args.seed)
        
        if args.H is not None:
            hidden_list = args.H
        else:
            hidden_list = [0, 1, 2, 4]
            
        print(f"Starting Experiment C: Learning Search")
        print(f"H values: {hidden_list}")
        print(f"Trials per H: {args.trials}")
        print(f"Base Seed: {args.seed}")
        print(f"Population: {args.population}, Elite K: {args.elite_k}")
        print(f"Restarts: {args.restarts}")
        print(f"Direct connections A,B->Y: {args.allow_direct_ab_to_y}")
        
        if args.mode == 'binary':
            R_effective = 2 if getattr(args, 'binary_force_R2', False) else args.R
            mode_variant = 'binary_strict_R2' if getattr(args, 'binary_force_R2', False) else 'binary_quantized'
        else:
            R_effective = args.R
            mode_variant = 'wbit'
        n_effective = max(1, (R_effective - 1) // 2)
        
        inputs = [(0,0), (0,1), (1,0), (1,1)]
        targets = [0, 1, 1, 0]
        
        for H_count in hidden_list:
            success_count = 0
            mses = []
            accs = []
            epochs_list = []
            
            for trial in range(args.trials):
                # Setup structure
                A, B, Y = 0, 1, 2
                num_cells = 3 + H_count
                connections = []
                h_indices = list(range(3, 3+H_count))
                
                # A,B -> H
                for h in h_indices:
                    connections.append((h, A)); connections.append((h, B))
                
                # H -> Y
                for h in h_indices:
                    connections.append((Y, h))
                
                # A,B -> Y (Optional)
                if args.allow_direct_ab_to_y:
                    connections.append((Y, A)); connections.append((Y, B))
                
                trial_seed_base = args.seed + trial + H_count * 1000
                
                best_trial_result = None # (converged, epochs, final_mse, rcp, final_acc)
                
                for restart in range(args.restarts):
                    restart_seed = trial_seed_base + restart * 100000
                    random.seed(restart_seed)
                    
                    population = []
                    for _ in range(args.population):
                        net_mode = args.mode
                        if args.mode == 'binary' and R_effective != 2:
                            net_mode = 'wbit'
                        population.append(create_random_net(num_cells, R_effective, connections, h_indices, Y, net_mode))
                    
                    epochs = 0
                    lr = 1.0
                    max_epochs = args.max_epochs
                    converged = False
                    
                    global_best_net = None
                    global_best_mse = 999.0
                    
                    while epochs < max_epochs:
                        scored_pop = []
                        for net in population:
                            mse, _, _ = evaluate_analog(net, inputs, targets, [A, B], Y, args.max_steps)
                            scored_pop.append((mse, net))
                            
                            if mse < global_best_mse:
                                global_best_mse = mse
                                global_best_net = net
                                if global_best_mse < 0.1: lr = 0.5
                                if global_best_mse < 0.05: lr = 0.2
                        
                        if global_best_mse < 0.02:
                            converged = True
                            break
                            
                        scored_pop.sort(key=lambda x: x[0])
                        elites = [x[1] for x in scored_pop[:args.elite_k]]
                        
                        new_pop = []
                        new_pop.extend(elites)
                        
                        while len(new_pop) < args.population:
                            parent = random.choice(elites)
                            child = mutate_net(parent, connections, h_indices, Y, lr)
                            new_pop.append(child)
                            
                        population = new_pop
                        epochs += 1
                    
                    final_mse, final_acc, avg_steps = evaluate_analog(global_best_net, inputs, targets, [A, B], Y, args.max_steps)
                    rcp = global_best_net.calculate_RCP(avg_steps, i_out=1.0, n_active_cells=num_cells)
                    
                    # C1: best_search_mse vs final_mse (they are same here since global_best_net is the one with best_mse)
                    result = (converged, epochs, global_best_mse, rcp, final_acc)
                    
                    if best_trial_result is None:
                        best_trial_result = result
                    else:
                        curr_conv = result[0]
                        best_conv = best_trial_result[0]
                        if curr_conv and not best_conv:
                            best_trial_result = result
                        elif (curr_conv == best_conv) and (result[2] < best_trial_result[2]):
                            best_trial_result = result
                            
                    if best_trial_result[0]:
                         break

                b_converged, b_epochs, b_mse, b_rcp, b_acc = best_trial_result
                
                row = [trial, trial_seed_base, H_count, 1 if b_converged else 0, b_epochs, f"{b_mse:.4f}", f"{b_rcp:.2f}", num_cells, R_effective, args.population, args.elite_k, args.restarts, args.allow_direct_ab_to_y, f"{b_mse:.4f}", f"{b_acc:.2f}", args.max_epochs, args.mode, mode_variant, n_effective, R_effective]
                
                # CSV Safety Assert
                assert len(row) == len(header), f"CSV row/header mismatch: {len(row)} vs {len(header)}"
                writer.writerow(row)
                if getattr(args, 'debug_metrics', False):
                    print({
                        "mode": args.mode,
                        "H": H_count,
                        "success": 1 if b_converged else 0,
                        "epochs": b_epochs,
                        "rcp": b_rcp,
                        "best_mse": b_mse
                    })
                
                if b_converged: success_count += 1
                mses.append(b_mse)
                accs.append(b_acc)
                epochs_list.append(b_epochs)
                
            mean_best_mse = statistics.mean(mses) if mses else 0.0
            mean_best_acc = statistics.mean(accs) if accs else 0.0
            mean_epochs = statistics.mean(epochs_list) if epochs_list else 0.0
            
            print(f"H={H_count}: Success {success_count/args.trials:.2f}, Mean Best MSE={mean_best_mse:.4f}, Mean Acc={mean_best_acc:.2f}, Mean Epochs={mean_epochs:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--H', type=int, nargs='+', help='List of hidden cell counts')
    parser.add_argument('--R', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='results/expC')
    parser.add_argument('--population', type=int, default=1)
    parser.add_argument('--elite_k', type=int, default=1)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary'], help='Logic substrate mode (wbit default, binary placeholder)')
    parser.add_argument('--binary_force_R2', action='store_true', help='Force R=2 in binary mode (strict ablation)')
    
    # C1 Toggle
    # argparse boolean trick
    parser.add_argument('--allow_direct_ab_to_y', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--debug_metrics', action='store_true')
    
    args = parser.parse_args()
    run_experiment(args)
