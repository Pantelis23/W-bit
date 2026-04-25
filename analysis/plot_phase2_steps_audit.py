import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_audit(path):
    rows = []
    if not os.path.exists(path):
        print(f"Missing audit: {path}")
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main(args):
    rows = load_audit(args.audit_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    if not rows:
        for fname in ['steps_zero_rate.png', 'steps_mean.png']:
            plt.figure(figsize=(6, 4))
            plt.title('No data')
            plt.savefig(os.path.join(args.output_dir, fname))
            plt.close()
        return

    labels_zero_all = []
    values_zero_all = []
    labels_zero_success = []
    values_zero_success = []
    labels_mean = []
    values_mean = []
    for r in rows:
        label = f"{r.get('experiment','?')}-{r.get('mode','?')}"
        try:
            zero_all = float(r.get('zero_step_fraction_all', 0))
            zero_success = float(r.get('zero_step_fraction_success', 0))
            mean_steps = float(r.get('mean_steps', 0))
        except (TypeError, ValueError):
            continue
        labels_zero_all.append(label)
        values_zero_all.append(zero_all)
        labels_zero_success.append(label)
        values_zero_success.append(zero_success)
        labels_mean.append(label)
        values_mean.append(mean_steps)

    plt.figure(figsize=(10, 5))
    plt.bar(labels_zero_all, values_zero_all)
    plt.title('Zero-Step Fraction (all trials)')
    plt.ylabel('Fraction')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'steps_zero_rate.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels_zero_success, values_zero_success)
    plt.title('Zero-Step Fraction (success-only)')
    plt.ylabel('Fraction')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'steps_zero_rate_success.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels_mean, values_mean)
    plt.title('Mean Steps')
    plt.ylabel('Steps')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'steps_mean.png'))
    plt.close()

    print(f"Step audit plots written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit_csv', type=str, default='results/phase2/phase2_steps_audit.csv')
    parser.add_argument('--output_dir', type=str, default='results/phase2/plots')
    args = parser.parse_args()
    main(args)
