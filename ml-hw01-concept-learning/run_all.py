import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# =====================================================================
# 1. EnjoySport Domain Definition and Table 2.1 Training Examples
# =====================================================================

ATTRIBUTES: List[str] = [
    "Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"
]

DOMAIN: Dict[str, List[str]] = {
    "Sky": ["Sunny", "Cloudy", "Rainy"],
    "AirTemp": ["Warm", "Cold"],
    "Humidity": ["Normal", "High"],
    "Wind": ["Strong", "Weak"],
    "Water": ["Warm", "Cool"],
    "Forecast": ["Same", "Change"],
}

# Training examples from Table 2.1 (EnjoySport)
TABLE_2_1_DATA: List[Tuple[Tuple[str, ...], str]] = [
    (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), "Yes"),
    (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), "Yes"),
    (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), "No"),
    (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), "Yes"),
]

EMPTY_SYMBOL = "∅"
WILDCARD_SYMBOL = "?"


# =====================================================================
# 2. Hypothesis Representation and "More General Than" Test
# =====================================================================

def satisfies(hypothesis: Tuple[str, ...], instance: Tuple[str, ...]) -> bool:
    """
    Check whether an instance satisfies a hypothesis (i.e., h(x) == 1).
    Any hypothesis containing '∅' rejects every instance.
    """
    for h_val, x_val in zip(hypothesis, instance):
        if h_val == EMPTY_SYMBOL:
            return False
        if h_val != WILDCARD_SYMBOL and h_val != x_val:
            return False
    return True


def is_more_general_or_equal(h_j: Tuple[str, ...], h_k: Tuple[str, ...]) -> bool:
    """
    Test whether hypothesis h_j is more general than or equal to h_k (h_j >=_g h_k).
    Accounts for the fact that any hypothesis containing '∅' rejects all instances.
    """
    if EMPTY_SYMBOL in h_k:
        return True
    if EMPTY_SYMBOL in h_j:
        return False

    for val_j, val_k in zip(h_j, h_k):
        if val_j == WILDCARD_SYMBOL:
            continue
        if val_j != val_k:
            return False
    return True


def is_strictly_more_general(h_j: Tuple[str, ...], h_k: Tuple[str, ...]) -> bool:
    """
    Test whether hypothesis h_j is strictly more general than h_k (h_j >_g h_k).
    """
    return is_more_general_or_equal(h_j, h_k) and not is_more_general_or_equal(h_k, h_j)


# =====================================================================
# 3. FIND-S Implementation and Table 2.1 Trace
# =====================================================================

def init_most_specific_hypothesis(num_attributes: int = 6) -> Tuple[str, ...]:
    """Initialize the most specific hypothesis in H: <∅, ∅, ∅, ∅, ∅, ∅>."""
    return tuple(EMPTY_SYMBOL for _ in range(num_attributes))


def find_s_step(
    hypothesis: Tuple[str, ...],
    instance: Tuple[str, ...],
    label: str
) -> Tuple[str, ...]:
    """
    Perform a single update step of the FIND-S algorithm for a training example.
    Negative examples (label != 'Yes') are ignored.
    """
    if label != "Yes":
        return hypothesis

    new_h = list(hypothesis)
    for i, (h_val, x_val) in enumerate(zip(new_h, instance)):
        if h_val == EMPTY_SYMBOL:
            new_h[i] = x_val
        elif h_val != WILDCARD_SYMBOL and h_val != x_val:
            new_h[i] = WILDCARD_SYMBOL
    return tuple(new_h)


def run_find_s_on_table_2_1() -> Tuple[str, ...]:
    """
    Run FIND-S on Table 2.1 and print the step-by-step trace matching Section 2.4.
    """
    print("=" * 70)
    print("Part 1: FIND-S Trace on Table 2.1 (EnjoySport)")
    print("=" * 70)

    h = init_most_specific_hypothesis(len(ATTRIBUTES))
    print(f"h_0 = <{', '.join(h)}>")

    for idx, (instance, label) in enumerate(TABLE_2_1_DATA, start=1):
        h = find_s_step(h, instance, label)
        sign = "+" if label == "Yes" else "-"
        print(f"Example {idx} ({sign}): <{', '.join(instance)}>")
        print(f"h_{idx} = <{', '.join(h)}>")

    print()
    return h


# =====================================================================
# 4. Random Instance Generator and Exercise 2.10 Experiment
# =====================================================================

def generate_random_instance(rng: np.random.Generator) -> Tuple[str, ...]:
    """
    Uniformly sample a random instance from the EnjoySport instance space X.
    """
    instance = []
    for attr in ATTRIBUTES:
        values = DOMAIN[attr]
        chosen_idx = rng.integers(0, len(values))
        instance.append(values[chosen_idx])
    return tuple(instance)


def label_instance(instance: Tuple[str, ...], target_concept: Tuple[str, ...]) -> str:
    """Label an instance as 'Yes' or 'No' according to the target concept."""
    return "Yes" if satisfies(target_concept, instance) else "No"


def run_single_experiment(
    target_concept: Tuple[str, ...],
    seed: int,
    n_max: int = 1000
) -> Dict[str, Any]:
    """
    Run a single independent trial of FIND-S with a fixed random seed.
    Stopping criterion:
      (i) FIND-S outputs exactly the target concept;
      (ii) Maximum number of training examples n_max = 1000 is reached.
    """
    rng = np.random.default_rng(seed)
    h = init_most_specific_hypothesis(len(ATTRIBUTES))

    num_examples = 0
    num_positive = 0

    while num_examples < n_max:
        instance = generate_random_instance(rng)
        label = label_instance(instance, target_concept)
        num_examples += 1
        if label == "Yes":
            num_positive += 1

        h = find_s_step(h, instance, label)

        # Check stopping condition (i): exact target concept learned
        if h == target_concept:
            return {
                "seed": seed,
                "num_examples": num_examples,
                "num_positive": num_positive,
                "stop_condition": "(i) Exact target learned",
                "converged": True,
                "final_hypothesis": h,
            }

    # Stopping condition (ii): N_max reached without learning exact target
    return {
        "seed": seed,
        "num_examples": n_max,
        "num_positive": num_positive,
        "stop_condition": "(ii) N_max reached",
        "converged": False,
        "final_hypothesis": h,
    }


def compute_statistics(values: List[int]) -> Dict[str, Any]:
    """
    Compute mean, sample standard deviation, median, interquartile range (IQR),
    minimum, and maximum for a list of run counts.
    """
    arr = np.array(values, dtype=float)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def run_exercise_2_10_experiments(
    seeds: List[int],
    n_max: int = 1000
) -> None:
    """
    Execute the 20-run experiment for Exercise 2.10 on the main target concept
    <Sunny, Warm, ?, ?, ?, ?> and evaluate how the number of '?' constraints
    in the target concept affects the required number of training examples.
    """
    target_main = ("Sunny", "Warm", "?", "?", "?", "?")

    print("=" * 70)
    print("Part 2: Exercise 2.10 Main Experiment")
    print(f"Target Concept: <{', '.join(target_main)}>")
    print(f"Seeds: {seeds[0]} .. {seeds[-1]} ({len(seeds)} independent runs), N_max = {n_max}")
    print("=" * 70)

    results = [run_single_experiment(target_main, seed=s, n_max=n_max) for s in seeds]

    print(f"{'Seed':>4} | {'Examples':>8} | {'Positives':>9} | {'Stopping Condition'}")
    print("-" * 55)
    for r in results:
        print(f"{r['seed']:>4} | {r['num_examples']:>8} | {r['num_positive']:>9} | {r['stop_condition']}")

    cond_i_runs = [r["num_examples"] for r in results if r["converged"]]
    cond_ii_runs = [r for r in results if not r["converged"]]

    print("\n--- Summary of Stopping Conditions ---")
    print(f"Ended by condition (i) [Exact target]: {len(cond_i_runs)} / {len(results)} runs")
    print(f"Ended by condition (ii) [N_max={n_max}]:  {len(cond_ii_runs)} / {len(results)} runs")

    if cond_i_runs:
        stats = compute_statistics(cond_i_runs)
        print("\n--- Statistics for Condition (i) Runs (Total Examples Needed) ---")
        print(f"Mean ± Std : {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"Median     : {stats['median']:.2f}")
        print(f"IQR        : {stats['iqr']:.2f}")
        print(f"Min        : {stats['min']}")
        print(f"Max        : {stats['max']}")

    # =================================================================
    # Evaluate effect of the number of '?' constraints in Target Concept
    # =================================================================
    targets_by_wildcards: List[Tuple[int, Tuple[str, ...]]] = [
        (0, ("Sunny", "Warm", "Normal", "Strong", "Warm", "Same")),
        (1, ("Sunny", "Warm", "Normal", "Strong", "Warm", "?")),
        (2, ("Sunny", "Warm", "Normal", "Strong", "?", "?")),
        (3, ("Sunny", "Warm", "Normal", "?", "?", "?")),
        (4, ("Sunny", "Warm", "?", "?", "?", "?")),
        (5, ("Sunny", "?", "?", "?", "?", "?")),
        (6, ("?", "?", "?", "?", "?", "?")),
    ]

    print("\n" + "=" * 70)
    print("Part 3: Effect of Number of '?' Constraints in Target Concept")
    print("=" * 70)
    header = (
        f"{'# of ?':>6} | {'Target Concept':<42} | "
        f"{'Cond(i)':>7} | {'Cond(ii)':>8} | {'Mean ± Std':<15} | {'Median (IQR)':<14} | {'Min/Max'}"
    )
    print(header)
    print("-" * len(header))

    num_q_list = []
    means = []
    stds = []
    medians = []

    for num_q, t_concept in targets_by_wildcards:
        t_results = [run_single_experiment(t_concept, seed=s, n_max=n_max) for s in seeds]
        c_i = [r["num_examples"] for r in t_results if r["converged"]]
        c_ii_count = sum(1 for r in t_results if not r["converged"])

        t_str = f"<{', '.join(t_concept)}>"
        if c_i:
            st = compute_statistics(c_i)
            num_q_list.append(num_q)
            means.append(st["mean"])
            stds.append(st["std"])
            medians.append(st["median"])

            mean_str = f"{st['mean']:.2f} ± {st['std']:.2f}"
            med_str = f"{st['median']:.1f} ({st['iqr']:.1f})"
            minmax_str = f"{st['min']} / {st['max']}"
        else:
            mean_str = "N/A"
            med_str = "N/A"
            minmax_str = "N/A"

        print(
            f"{num_q:>6} | {t_str:<42} | "
            f"{len(c_i):>7} | {c_ii_count:>8} | {mean_str:<15} | {med_str:<14} | {minmax_str}"
        )

    if not num_q_list:
        print("\n[Warning] No run converged for any target; skipping the plot.")
        return

    # Save figure in the exact same directory as run_all.py
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "exercise_2_10_plot.png"

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        num_q_list,
        means,
        yerr=stds,
        fmt="-o",
        capsize=5,
        label=f"Mean ± 1 Std ({len(seeds)} runs)"
    )
    plt.plot(
        num_q_list,
        medians,
        "--s",
        label=f"Median ({len(seeds)} runs)"
    )
    plt.xlabel("Number of '?' Constraints in Target Concept")
    plt.ylabel("Number of Training Examples Needed (N)")
    plt.title(f"FIND-S Sample Complexity vs. Number of '?' in Target (n={len(seeds)} runs, seeds {seeds[0]}..{seeds[-1]})")
    plt.xticks(num_q_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[Info] Figure saved at: {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Make sure the '∅' symbol can be printed on consoles with a non-UTF-8 default encoding
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Reproduce the FIND-S trace on Table 2.1
    run_find_s_on_table_2_1()

    # 2. Run Exercise 2.10 experiments across 20 recorded seeds (0..19)
    recorded_seeds = list(range(20))
    run_exercise_2_10_experiments(seeds=recorded_seeds, n_max=1000)