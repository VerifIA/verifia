---
icon: fontawesome/brands/searchengin
---

# Searcher

The **Searcher** is a core VerifIA component that explores the **input subspace** defined by each original‑seed × rule pair. It uses population‑based evolutionary algorithms to both discover challenging derived inputs and quantify overall consistency.

---

## 1. Objective Function

- **Goal:** Find derived inputs that **maximize deviation** between the model’s prediction and the rule‑based expected response.  
- **Why not a single optimum?** A single “worst” violation doesn’t prove inconsistency. Instead, multiple inconsistent points must be observed to statistically reject a model.

---

## 2. Statistical Consistency Measurement

1. **Population of derived inputs**  
      - Initialize a set of candidate points within the constrained subspace.  
2. **Evolutionary optimization**  
      - Evolve the population to increase prediction deviation from the expected output direction (e.g. if rule says “output ↑”, seek predictions with stalled or decreased values).  
3. **Consistency ratio**  
      - For each seed‑rule pair, compute the fraction of derived inputs that **violate** the rule.  
      - Aggregate ratios across all seeds and rules to measure model consistency against the full domain.

---

## 3. Population‑Based Evolutionary Algorithms

VerifIA currently supports several **nature‑inspired** population-based evolutionary metaheuristics:

| Algorithm | Enum Key | Description                                                                           |
|-----------|----------|---------------------------------------------------------------------------------------|
| Random Sampler       | `RS`   | Fresh random sampling each iteration.                                                |
| Firefly Algorithm    | `FFA`  | Bioluminescent attraction & random moves. <br> *params:* `beta_min`, `gamma`, `alpha`, `theta` |
| Genetic Algorithm    | `GA`   | Crossover & mutation on elites/parents. <br> *params:* `mutation_prob`, `crossover_prob`, `elite_ratio` |
| Particle Swarm       | `PSO`  | Velocity & position updates via personal/global best. <br> *params*: `c_1`, `c_2`, `w_max`, `w_min`. |
| Grey Wolf Optimizer  | `GWO`  | Leadership hierarchy & encircling prey principles.                                    |
| Moth–Flame Optimizer | `MFO`  | Spiral movement around ranked “flames”. <br> *params*: `b`                              |
| Multi‑Verse Optimizer| `MVO`  | Universe inflation/deflation & wormhole tunneling. <br> *params*: `WEP_min`, `WEP_max`, `TDR_p`. |
| Salp Swarm Algorithm | `SSA`  | Chain‑structured population with leader & followers.                                  |
| Whale Optimization   | `WOA`  | Encircling & spiral bubble-net attacks. <br> *params*: `b`                             |

The algorithm‑specific `params` control **exploration**/**exploitation** balance.

---

## 4. Run-wise Configurable Parameters

All population-based searching algorithms accept these two parameters at runtime:

| Parameter         | Purpose                                                     | Default |
|-------------------|-------------------------------------------------------------|---------|
| `pop_size`        | Number of candidates per generation                         | user‑defined |
| `max_iters`       | Iteration budget                                            | user‑defined |

Tune `pop_size` & `max_iters` to trade off **coverage**/**search scope** vs **runtime**/**compute budget**. 

---

## 5. Workflow Summary

1. **For each** original seed × rule pair:  
   a. Define the **constrained subspace** (white-noise + variation limits + constraints).  
   b. Initialize a **random population** within that subspace.  
   c. **Evolve** the population using the selected algorithm and objective function.  
   d. **Collect** derived inputs that violate the rule.  

2. **Compute** consistency ratios:  
      - `ratio = violations_count / total_candidates`  

3. **Aggregate** across all rules and seeds to assess overall domain consistency.

---

## 6. Selecting your Searcher

Before running verification, you must seect which searcher *VerifIA* will use and its parameters:

```python
from verifia.verification import RuleConsistencyVerifier

verifier = RuleConsistencyVerifier("domain_rules.yaml")

# Provide algorithm name and optional parameters
verifier = verifier.using(
    search_algo="PSO",
    search_params={
        "c_1": 0.5,    # lower personal influence
        "c_2": 0.3,    # higher social influence
        "w_max": 0.8,  # reduced max inertia
        "w_min": 0.2   # increased exploitation
    }
)
```

!!! failure

      Do not include `pop_size` or `max_iters` in `search_params`; they belong in the top‑level `.run()` call.
