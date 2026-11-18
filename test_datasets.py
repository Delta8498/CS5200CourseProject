"""
CS5200 Project: Deadline-Aware Task Scheduling with Precedence Constraints
Shared, reproducible datasets for Greedy, Dynamic Programming, and Branch-and-Bound.

Problem rules:
- One processor; no overlap and no preemption.
- All tasks are available at time zero.
- A task earns its reward only if completion_time <= deadline.
- Every dependency must finish before its dependent task starts.
- A late prerequisite may still be executed to unlock a dependent task.
- Dependencies form a DAG (no cycles).

Contents:
1. Eight fixed, hand-designed test datasets (D1-D8).
2. Seven generated groups: 5, 10, 15, 20, 30, 40, and 50 tasks.
   Each group contains 10 instances generated from fixed seeds.
3. Helpers to validate datasets and inspect available datasets.

Run:
    python test_datasets.py

Import:
    from test_datasets import FIXED_DATASETS, GENERATED_GROUPS, ALL_DATASETS
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Any


Task = Dict[str, Any]


# ---------------------------------------------------------------------------
# PART 1: FIXED DATASETS (hand-designed scenarios)
# ---------------------------------------------------------------------------

D1_INDEPENDENT = [
    {"id": "A", "duration": 3, "deadline": 8,  "reward": 20, "dependencies": []},
    {"id": "B", "duration": 2, "deadline": 5,  "reward": 25, "dependencies": []},
    {"id": "C", "duration": 4, "deadline": 10, "reward": 30, "dependencies": []},
    {"id": "D", "duration": 1, "deadline": 3,  "reward": 15, "dependencies": []},
    {"id": "E", "duration": 5, "deadline": 12, "reward": 40, "dependencies": []},
]

D2_CHAIN = [
    {"id": "A", "duration": 2, "deadline": 3,  "reward": 15, "dependencies": []},
    {"id": "B", "duration": 3, "deadline": 7,  "reward": 25, "dependencies": ["A"]},
    {"id": "C", "duration": 2, "deadline": 9,  "reward": 20, "dependencies": ["B"]},
    {"id": "D", "duration": 4, "deadline": 14, "reward": 35, "dependencies": ["C"]},
    {"id": "E", "duration": 1, "deadline": 15, "reward": 10, "dependencies": ["D"]},
]

D3_BRANCHING = [
    {"id": "A", "duration": 2, "deadline": 4,  "reward": 20, "dependencies": []},
    {"id": "B", "duration": 3, "deadline": 8,  "reward": 30, "dependencies": ["A"]},
    {"id": "C", "duration": 2, "deadline": 7,  "reward": 25, "dependencies": ["A"]},
    {"id": "D", "duration": 2, "deadline": 11, "reward": 35, "dependencies": ["B", "C"]},
    {"id": "E", "duration": 3, "deadline": 10, "reward": 20, "dependencies": []},
    {"id": "F", "duration": 1, "deadline": 13, "reward": 15, "dependencies": ["D"]},
]

D4_GREEDY_CHALLENGE = [
    {"id": "A", "duration": 5, "deadline": 5, "reward": 50, "dependencies": []},
    {"id": "B", "duration": 2, "deadline": 2, "reward": 24, "dependencies": []},
    {"id": "C", "duration": 2, "deadline": 4, "reward": 24, "dependencies": []},
    {"id": "D", "duration": 2, "deadline": 6, "reward": 22, "dependencies": []},
    {"id": "E", "duration": 1, "deadline": 7, "reward": 10, "dependencies": []},
]

D5_LATE_PREREQUISITE = [
    {"id": "A", "duration": 4, "deadline": 2, "reward": 30, "dependencies": []},
    {"id": "B", "duration": 2, "deadline": 7, "reward": 50, "dependencies": ["A"]},
    {"id": "C", "duration": 1, "deadline": 8, "reward": 20, "dependencies": ["B"]},
    {"id": "D", "duration": 2, "deadline": 4, "reward": 25, "dependencies": []},
]

D6_TIGHT_DEADLINES = [
    {"id": "A", "duration": 2, "deadline": 3,  "reward": 20, "dependencies": []},
    {"id": "B", "duration": 3, "deadline": 6,  "reward": 45, "dependencies": ["A"]},
    {"id": "C", "duration": 1, "deadline": 2,  "reward": 15, "dependencies": []},
    {"id": "D", "duration": 2, "deadline": 7,  "reward": 30, "dependencies": []},
    {"id": "E", "duration": 2, "deadline": 9,  "reward": 35, "dependencies": ["B"]},
    {"id": "F", "duration": 1, "deadline": 10, "reward": 15, "dependencies": ["D"]},
]

D7_IMPOSSIBLE_DEADLINES = [
    {"id": "A", "duration": 5, "deadline": 2,  "reward": 40, "dependencies": []},
    {"id": "B", "duration": 3, "deadline": 4,  "reward": 30, "dependencies": ["A"]},
    {"id": "C", "duration": 2, "deadline": 3,  "reward": 25, "dependencies": []},
    {"id": "D", "duration": 1, "deadline": 10, "reward": 20, "dependencies": ["B"]},
    {"id": "E", "duration": 2, "deadline": 12, "reward": 35, "dependencies": ["D"]},
]

D8_MIXED_LARGE = [
    {"id": "A", "duration": 2, "deadline": 5,  "reward": 20, "dependencies": []},
    {"id": "B", "duration": 3, "deadline": 8,  "reward": 35, "dependencies": ["A"]},
    {"id": "C", "duration": 1, "deadline": 4,  "reward": 15, "dependencies": []},
    {"id": "D", "duration": 4, "deadline": 10, "reward": 40, "dependencies": []},
    {"id": "E", "duration": 2, "deadline": 12, "reward": 30, "dependencies": ["B"]},
    {"id": "F", "duration": 3, "deadline": 9,  "reward": 25, "dependencies": ["C"]},
    {"id": "G", "duration": 2, "deadline": 14, "reward": 35, "dependencies": ["D"]},
    {"id": "H", "duration": 1, "deadline": 13, "reward": 20, "dependencies": ["E", "F"]},
    {"id": "I", "duration": 3, "deadline": 16, "reward": 40, "dependencies": ["G"]},
    {"id": "J", "duration": 2, "deadline": 18, "reward": 30, "dependencies": ["H", "I"]},
]

FIXED_DATASETS: Dict[str, List[Task]] = {
    "D1_Independent": D1_INDEPENDENT,
    "D2_Chain": D2_CHAIN,
    "D3_Branching": D3_BRANCHING,
    "D4_Greedy_Challenge": D4_GREEDY_CHALLENGE,
    "D5_Late_Prerequisite": D5_LATE_PREREQUISITE,
    "D6_Tight_Deadlines": D6_TIGHT_DEADLINES,
    "D7_Impossible_Deadlines": D7_IMPOSSIBLE_DEADLINES,
    "D8_Mixed_Large": D8_MIXED_LARGE,
}


# ---------------------------------------------------------------------------
# PART 2: RANDOM DAG DATASET GENERATOR
# ---------------------------------------------------------------------------

def generate_dag_dataset(
    n: int,
    seed: int,
    dependency_probability: float = 0.18,
    deadline_tightness: str = "mixed",
) -> List[Task]:
    """
    Generate one reproducible scheduling instance with n tasks.

    A random topological order is created first. Dependencies are only
    assigned from earlier to later tasks in that order, guaranteeing a DAG.

    deadline_tightness:
        "tight"   - generally difficult deadlines
        "mixed"   - mixture of tight and relaxed deadlines
        "relaxed" - generally more generous deadlines
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= dependency_probability <= 1:
        raise ValueError("dependency_probability must be between 0 and 1")
    if deadline_tightness not in {"tight", "mixed", "relaxed"}:
        raise ValueError("deadline_tightness must be tight, mixed, or relaxed")

    rng = random.Random(seed)
    ids = [f"T{i:02d}" for i in range(1, n + 1)]
    topological_order = ids[:]
    rng.shuffle(topological_order)

    duration = {task_id: rng.randint(1, 10) for task_id in ids}
    reward = {task_id: rng.randint(10, 100) for task_id in ids}
    dependencies = {task_id: [] for task_id in ids}

    # Only earlier tasks in the shuffled topological order can be parents.
    for later_index, task_id in enumerate(topological_order):
        for earlier_id in topological_order[:later_index]:
            if rng.random() < dependency_probability:
                dependencies[task_id].append(earlier_id)

    # Keep deadlines varied and meaningful. The base is a fraction of total
    # processing time, then individual deadlines are distributed around it.
    total_duration = sum(duration.values())
    if deadline_tightness == "tight":
        low_factor, high_factor = 0.18, 0.60
    elif deadline_tightness == "relaxed":
        low_factor, high_factor = 0.55, 1.15
    else:
        low_factor, high_factor = 0.25, 0.95

    deadlines = {}
    for task_id in ids:
        factor = rng.uniform(low_factor, high_factor)
        deadlines[task_id] = max(
            1, int(round(total_duration * factor * rng.uniform(0.65, 1.0)))
        )

    tasks = []
    for task_id in ids:
        tasks.append({
            "id": task_id,
            "duration": duration[task_id],
            "deadline": deadlines[task_id],
            "reward": reward[task_id],
            "dependencies": sorted(dependencies[task_id]),
        })

    return tasks


def generate_dataset_group(
    n: int,
    instances: int = 10,
    base_seed: int = 520000,
    dependency_probability: float = 0.18,
    deadline_tightness: str = "mixed",
) -> Dict[str, List[Task]]:
    """Generate a named group of reproducible instances of the same size."""
    group = {}
    for instance_number in range(1, instances + 1):
        seed = base_seed + n * 100 + instance_number
        name = f"N{n:02d}_Instance_{instance_number:02d}"
        group[name] = generate_dag_dataset(
            n=n,
            seed=seed,
            dependency_probability=dependency_probability,
            deadline_tightness=deadline_tightness,
        )
    return group


# ---------------------------------------------------------------------------
# PART 3: ALL GENERATED GROUPS
# ---------------------------------------------------------------------------

TASK_COUNTS = [5, 10, 15, 20, 30, 40, 50]

GENERATED_GROUPS: Dict[str, Dict[str, List[Task]]] = {}
for task_count in TASK_COUNTS:
    GENERATED_GROUPS[f"G{task_count:02d}_Tasks"] = generate_dataset_group(
        n=task_count,
        instances=10,
        base_seed=520000,
        dependency_probability=0.18,
        deadline_tightness="mixed",
    )

# Flattened dictionary: fixed datasets + all 70 generated instances.
ALL_DATASETS: Dict[str, List[Task]] = dict(FIXED_DATASETS)
for group in GENERATED_GROUPS.values():
    ALL_DATASETS.update(group)


# ---------------------------------------------------------------------------
# PART 4: DATASET VALIDATION AND INSPECTION
# ---------------------------------------------------------------------------

def validate_dataset(tasks: List[Task]) -> bool:
    """Raise ValueError for malformed tasks, unknown dependencies, or cycles."""
    required = {"id", "duration", "deadline", "reward", "dependencies"}
    ids = [task.get("id") for task in tasks]

    if len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be unique")

    id_set = set(ids)
    graph = defaultdict(list)
    indegree = {task_id: 0 for task_id in ids}

    for task in tasks:
        missing = required - set(task)
        if missing:
            raise ValueError(f"Task {task.get('id')} is missing fields: {missing}")
        if not isinstance(task["duration"], int) or task["duration"] <= 0:
            raise ValueError(f"Task {task['id']} must have a positive integer duration")
        if not isinstance(task["deadline"], int) or task["deadline"] < 0:
            raise ValueError(f"Task {task['id']} must have a nonnegative integer deadline")
        if not isinstance(task["reward"], (int, float)) or task["reward"] < 0:
            raise ValueError(f"Task {task['id']} must have a nonnegative reward")

        for parent in task["dependencies"]:
            if parent not in id_set:
                raise ValueError(f"Task {task['id']} has unknown dependency {parent}")
            if parent == task["id"]:
                raise ValueError(f"Task {task['id']} cannot depend on itself")
            graph[parent].append(task["id"])
            indegree[task["id"]] += 1

    # Kahn's algorithm: a DAG must have all nodes processed.
    queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in graph[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(tasks):
        raise ValueError("Dependencies contain a cycle; the graph is not a DAG")

    return True


def describe_dataset(name: str, tasks: List[Task]) -> None:
    """Print a readable summary and all task rows."""
    validate_dataset(tasks)
    edge_count = sum(len(task["dependencies"]) for task in tasks)
    print(f"\n{name}: {len(tasks)} tasks, {edge_count} dependency edges")
    print("-" * 88)
    print(f"{'ID':<8}{'Duration':>10}{'Deadline':>10}{'Reward':>10}  Dependencies")
    for task in tasks:
        deps = ", ".join(task["dependencies"]) if task["dependencies"] else "-"
        print(
            f"{task['id']:<8}{task['duration']:>10}"
            f"{task['deadline']:>10}{task['reward']:>10}  {deps}"
        )


def get_dataset(name: str) -> List[Task]:
    """Return a deep-enough copy so an algorithm cannot mutate shared inputs."""
    import copy
    if name not in ALL_DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return copy.deepcopy(ALL_DATASETS[name])


def list_datasets() -> None:
    """Print every dataset name and task count."""
    print("FIXED DATASETS")
    for name, tasks in FIXED_DATASETS.items():
        print(f"  {name}: {len(tasks)} tasks")

    print("\nGENERATED GROUPS")
    for group_name, group in GENERATED_GROUPS.items():
        sizes = sorted({len(tasks) for tasks in group.values()})
        print(f"  {group_name}: {len(group)} instances, task size(s)={sizes}")

    print(f"\nTotal: {len(ALL_DATASETS)} datasets")


# ---------------------------------------------------------------------------
# PART 5: QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Validate every fixed and generated dataset.
    for dataset_name, task_list in ALL_DATASETS.items():
        validate_dataset(task_list)

    print("All datasets passed validation.")
    list_datasets()

    # Print one example from each generated group.
    print("\nONE EXAMPLE FROM EACH GENERATED GROUP")
    for group_name, group in GENERATED_GROUPS.items():
        example_name = sorted(group.keys())[0]
        describe_dataset(example_name, group[example_name])

    # Example: inspect a fixed scenario.
    describe_dataset("D5_Late_Prerequisite", D5_LATE_PREREQUISITE)
