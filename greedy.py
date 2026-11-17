"""
greedy.py

Original Author: Ashlan Garner
"""

from __future__ import annotations

import copy
import heapq
import time
import tracemalloc
from collections import defaultdict
from typing import Any, Callable
from test_datasets import ALL_DATASETS, validate_dataset

Task = dict[str, Any]

def _priority_key(task: Task, algorithm: str) -> tuple:
    # Return a deterministic priority key; lower tuple sorts first.
    if algorithm == "EDF":
        return (task["deadline"], task["duration"], str(task["id"]))
    if algorithm == "SPT":
        return (task["duration"], task["deadline"], str(task["id"]))
    if algorithm == "Reward Density":
        # Negative density sorts highest density first.
        return (
            -(task["reward"] / task["duration"]),
            task["deadline"],
            str(task["id"]),
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _unfinished_ancestor_closure(target_id: str, task_by_id: dict[str, Task], completed: set[str]) -> set[str]:
    # Return unfinished prerequisites needed to execute target_id.
    closure: set[str] = set()
    stack = [target_id]

    while stack:
        current = stack.pop()
        if current in completed or current in closure:
            continue
        closure.add(current)
        stack.extend(task_by_id[current]["dependencies"])

    return closure


def _late_prerequisite_key(
    candidate: Task,
    now: int,
    task_by_id: dict[str, Task],
    descendants: dict[str, set[str]],
    completed: set[str],
) -> tuple | None:
    # Determine whether a late eligible task is useful.
    candidate_id = candidate["id"]
    unfinished_descendants = descendants[candidate_id] - completed
    if not unfinished_descendants:
        return None

    useful = False
    for descendant_id in unfinished_descendants:
        descendant = task_by_id[descendant_id]
        if descendant["reward"] <= 0:
            continue

        closure = _unfinished_ancestor_closure(
            descendant_id, task_by_id, completed
        )
        if candidate_id not in closure:
            continue

        required_duration = sum(task_by_id[item]["duration"] for item in closure)
        if now + required_duration <= descendant["deadline"]:
            useful = True
            break

    if not useful:
        return None

    earliest_descendant_deadline = min(
        task_by_id[item]["deadline"] for item in unfinished_descendants
    )
    return (
        candidate["duration"],
        earliest_descendant_deadline,
        str(candidate_id),
    )


def _build_descendants(tasks: list[Task]) -> dict[str, set[str]]:
    # Build the transitive descendant set for each task.
    children: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for dependency in task["dependencies"]:
            children[dependency].append(task["id"])

    result: dict[str, set[str]] = {}
    for task in tasks:
        root = task["id"]
        found: set[str] = set()
        stack = list(children[root])
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(children[current])
        result[root] = found
    return result



def _run_schedule_pass(
    task_list: list[Task],
    task_by_id: dict[str, Task],
    descendants: dict[str, set[str]],
    algorithm: str,
    excluded_late_tasks: set[str],
) -> list[dict[str, Any]]:

    completed = set()
    schedule = []
    current_time = 0

    while len(completed) < len(task_list):

        # Only tasks whose dependencies are completed are eligible.
        eligible = [
            task for task in task_list
            if task["id"] not in completed
            and all(dep in completed for dep in task["dependencies"])
        ]

        if not eligible:
            break

        # FIRST: Find useful late prerequisites.
        useful_late = []

        for task in eligible:
            task_id = task["id"]

            is_late = (
                current_time + task["duration"] > task["deadline"]
            )

            if not is_late or task_id in excluded_late_tasks:
                continue

            key = _late_prerequisite_key(
                task,
                current_time,
                task_by_id,
                descendants,
                completed
            )

            if key is not None:
                useful_late.append((key, task))

        if useful_late:
            # Shortest duration, then earliest unfinished descendant deadline.
            _, selected = min(
                useful_late,
                key=lambda item: item[0]
            )

        else:
            # SECOND: Apply the selected greedy algorithm to on-time tasks.
            on_time_eligible = [
                task for task in eligible
                if current_time + task["duration"] <= task["deadline"]
            ]

            if not on_time_eligible:
                break

            selected = min(
                on_time_eligible,
                key=lambda task: _priority_key(task, algorithm)
            )

        start_time = current_time
        completion_time = start_time + selected["duration"]

        on_time = completion_time <= selected["deadline"]
        earned_reward = selected["reward"] if on_time else 0

        schedule.append({
            "task_id": selected["id"],
            "start_time": start_time,
            "completion_time": completion_time,
            "deadline": selected["deadline"],
            "earned_reward": earned_reward,
            "on_time": on_time
        })

        completed.add(selected["id"])
        current_time = completion_time

    return schedule


def _schedule(tasks: list[Task], dataset_name: str, algorithm: str) -> dict[str, Any]:

    tracemalloc.start()
    start_clock = time.perf_counter()

    try:
        task_list = copy.deepcopy(tasks)
        validate_dataset(task_list)

        task_by_id = {
            task["id"]: task for task in task_list
        }

        descendants = _build_descendants(task_list)

        excluded_late_tasks = set()

        # Rebuild the schedule until no unsupported late tasks remain.
        while True:

            schedule = _run_schedule_pass(
                task_list,
                task_by_id,
                descendants,
                algorithm,
                excluded_late_tasks
            )

            # Find tasks that actually earned a reward.
            rewarded_task_ids = {
                row["task_id"]
                for row in schedule
                if row["earned_reward"] > 0
            }

            # Find all prerequisites needed by rewarded tasks.
            required_for_reward = set()

            for task_id in rewarded_task_ids:
                required_for_reward.update(
                    _unfinished_ancestor_closure(
                        task_id,
                        task_by_id,
                        set()
                    )
                )

            # Late tasks that do not support an actually rewarded taskmust not remain in the schedule as late prerequisites.
            unsupported_late_tasks = {
                row["task_id"]
                for row in schedule
                if not row["on_time"]
                and row["task_id"] not in required_for_reward
                and row["task_id"] not in excluded_late_tasks
            }

            if not unsupported_late_tasks:
                break

            excluded_late_tasks.update(unsupported_late_tasks)

        # Calculate final reward and unscheduled tasks.
        total_reward = sum(
            row["earned_reward"] for row in schedule
        )

        scheduled_ids = {
            row["task_id"] for row in schedule
        }

        unscheduled_tasks = [
            task["id"]
            for task in task_list
            if task["id"] not in scheduled_ids
        ]

        runtime_seconds = time.perf_counter() - start_clock
        _, peak_memory_bytes = tracemalloc.get_traced_memory()

        return {
            "algorithm_name": algorithm,
            "dataset_name": dataset_name,
            "schedule_sequence": [
                row["task_id"] for row in schedule
            ],
            "schedule": schedule,
            "total_reward": total_reward,
            "runtime_seconds": runtime_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "unscheduled_tasks": unscheduled_tasks
        }

    finally:
        tracemalloc.stop()



def earliest_deadline_first(tasks: list[Task], dataset_name: str = "Unnamed") -> dict[str, Any]:
    # Run dependency-aware Earliest Deadline First.
    return _schedule(tasks, dataset_name, "EDF")


def shortest_processing_time(tasks: list[Task], dataset_name: str = "Unnamed") -> dict[str, Any]:
    # Run dependency-aware Shortest Processing Time.
    return _schedule(tasks, dataset_name, "SPT")


def reward_density(tasks: list[Task], dataset_name: str = "Unnamed") -> dict[str, Any]:
    # Run dependency-aware Reward Density (reward divided by duration).
    return _schedule(tasks, dataset_name, "Reward Density")


ALGORITHMS: dict[str, Callable[[list[Task], str], dict[str, Any]]] = {
    "EDF": earliest_deadline_first,
    "SPT": shortest_processing_time,
    "Reward Density": reward_density,
}


def run_all_algorithms(tasks: list[Task], dataset_name: str = "Unnamed") -> dict[str, dict[str, Any]]:
    # Run all three algorithms on the same task list.
    return {
        name: algorithm(tasks, dataset_name)
        for name, algorithm in ALGORITHMS.items()
    }


def run_all_datasets():
    # Run all algorithms on every dataset in test_datasets.py.
    results = {}
    for dataset_name, tasks in ALL_DATASETS.items():
        results[dataset_name] = run_all_algorithms(tasks, dataset_name)

    return results


def print_result(result: dict[str, Any]) -> None:
    print(f"\n{result['algorithm_name']} | {result['dataset_name']}")
    print("-" * 76)
    print(
        f"{'Task':<10}{'Start':>10}{'Complete':>12}"
        f"{'Deadline':>12}{'Reward':>12}{'On time':>12}"
    )
    for row in result["schedule"]:
        print(
            f"{row['task_id']:<10}{row['start_time']:>10}"
            f"{row['completion_time']:>12}{row['deadline']:>12}"
            f"{row['earned_reward']:>12}{str(row['on_time']):>12}"
        )
    print(f"Sequence: {result['schedule_sequence']}")
    print(f"Total reward: {result['total_reward']}")
    print(f"Runtime: {result['runtime_seconds']:.8f} seconds")
    print(f"Peak traced memory: {result['peak_memory_bytes']} bytes")

def print_result(result):
    print("\n" + "=" * 60)
    print(f"Algorithm: {result['algorithm_name']}")
    print(f"Dataset: {result['dataset_name']}")
    print("=" * 60)

    print("\nSchedule Sequence:")
    print(" -> ".join(result["schedule_sequence"]))

    print("\nTask Details:")
    print(
        f"{'Task':<10}"
        f"{'Start':<12}"
        f"{'Completion':<15}"
        f"{'Deadline':<12}"
        f"{'Reward':<10}"
    )
    print("-" * 59)

    for task in result["schedule"]:
        print(
            f"{task['task_id']:<10}"
            f"{task['start_time']:<12}"
            f"{task['completion_time']:<15}"
            f"{task['deadline']:<12}"
            f"{task['earned_reward']:<10}"
        )

    print("-" * 59)
    print(f"Total Reward: {result['total_reward']}")
    print(f"Runtime: {result['runtime_seconds']:.8f} seconds")

    memory = result.get("peak_memory_bytes")
    if memory is not None:
        print(f"Peak Memory: {memory:,} bytes")
    else:
        print("Peak Memory: Not measured")

    print("=" * 60)


if __name__ == "__main__":
    import argparse
    from test_datasets import ALL_DATASETS, FIXED_DATASETS, validate_dataset

    parser = argparse.ArgumentParser(
        description="Run greedy scheduling algorithms."
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["all", "fixed", "one"],
        default="all"
    )

    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Name of the dataset when using mode 'one'."
    )

    args = parser.parse_args()

    # Select datasets
    if args.mode == "all":
        selected_datasets = ALL_DATASETS

    elif args.mode == "fixed":
        selected_datasets = FIXED_DATASETS

    elif args.mode == "one":
        if not args.dataset_name:
            parser.error("Please provide a dataset name.")

        if args.dataset_name not in ALL_DATASETS:
            parser.error(f"Unknown dataset: {args.dataset_name}")

        selected_datasets = {
            args.dataset_name: ALL_DATASETS[args.dataset_name]
        }

    # Execute algorithms
    results = {}

    for dataset_name, tasks in selected_datasets.items():

        validate_dataset(tasks)

        print("\n\n" + "#" * 60)
        print(f"DATASET: {dataset_name}")
        print("#" * 60)

        results[dataset_name] = run_all_algorithms(
            tasks, dataset_name
        )

        # Print the complete results for each algorithm
        for algorithm_name, result in results[dataset_name].items():
            print_result(result)

    # Print summary
    print("FINAL SUMMARY")

    print(f"\nCompleted {len(results) * 3} algorithm executions.")
    print(f"Datasets tested: {len(results)}")

    print("\nSummary (total reward):")
    print(f"{'Dataset':<28}{'EDF':>10}{'SPT':>10}{'Reward Density':>18}")

    for dataset_name, algorithm_results in results.items():
        print(
            f"{dataset_name:<28}"
            f"{algorithm_results['EDF']['total_reward']:>10}"
            f"{algorithm_results['SPT']['total_reward']:>10}"
            f"{algorithm_results['Reward Density']['total_reward']:>18}"
        )