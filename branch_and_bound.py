from __future__ import annotations

import argparse
import json
import platform
import signal
import time
import tracemalloc
from typing import Any
from test_datasets import ALL_DATASETS, FIXED_DATASETS, get_dataset, validate_dataset


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Solver timed out")


def verify_os():
    if platform.system() != "Linux":
        raise OSError(
            "\tThis program is designed to run on Linux systems only.\n\t\tPlease run it in a Linux environment ( WSL or a Linux VM )."
        )


class BranchAndBoundSolver:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        dataset_name: str = "Unknown",
    ):
        validate_dataset(tasks)
        self.dataset_name = dataset_name
        self.tasks = {t["id"]: t for t in tasks}
        self.all_task_ids = set(self.tasks.keys())

        # Statistics tracking
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_reward = -1.0
        self.best_schedule: list[str] = []

    def _compute_upper_bound(
        self, current_time: int, scheduled_set: set, current_reward: float
    ) -> float:
        unscheduled = [t for tid, t in self.tasks.items() if tid not in scheduled_set]
        if len(unscheduled) == 0:
            return current_reward

        potential_gain = 0.0
        scored_tasks = []
        max_deadline = 0

        for t in unscheduled:
            ratio = (
                (t["reward"] / t["duration"]) if t["duration"] > 0 else float("inf")
            )  # Give it a reward ratio of reward / duration
            can_meet = (current_time + t["duration"]) <= t[
                "deadline"
            ]  # If the task can be completed by its deadline

            if can_meet:
                max_deadline = max(max_deadline, t["deadline"])
                scored_tasks.append((ratio, t["reward"], t["duration"], t["deadline"]))

        scored_tasks.sort(
            key=lambda x: x[0], reverse=True
        )  # Sort the tasks by descending ratio

        temp_time = current_time
        for ratio, reward, duration, deadline in scored_tasks:
            if (
                temp_time + duration <= max_deadline or duration == 0
            ):  # 										If the tasks fit the loose deadline
                potential_gain += reward  # 			add entire duration and reward
                temp_time += duration

            else:
                remaining_time = max_deadline - temp_time
                potential_gain += (
                    reward * remaining_time / duration
                )  # 	 Task takes longer to complete than time given ->
                break  # fractional knapsack, and knapsack is full

        return current_reward + potential_gain

    def solve(self) -> tuple[dict[str, Any], bool]:

        tracemalloc.start()
        start_time = time.perf_counter()

        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_reward = -1.0
        self.best_schedule = []
        timed_out = False

        initial_state = (
            0,
            frozenset(),
            [],
            0.0,
            self._compute_upper_bound(0, frozenset(), 0.0),
        )
        stack = [initial_state]

        try:
            while stack:
                curr_time, scheduled, schedule, reward, ub = stack.pop()
                self.nodes_explored += 1
                # If all tasks are scheduled, then check if the reward is worth
                if len(scheduled) == len(self.all_task_ids):
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_schedule = list(schedule)
                    continue

                if ub <= self.best_reward:
                    self.nodes_pruned += 1
                    continue

                eligible = []

                for tid, task in self.tasks.items():
                    if tid not in scheduled and all(
                        dep in scheduled for dep in task["dependencies"]
                    ):  #  Not scheduled And all it's dependencies have been scheduled
                        eligible.append(task)  # It is eligible

                eligible.sort(
                    key=lambda t: (
                        (t["reward"] / t["duration"])
                        if t["duration"] > 0
                        else float("inf")
                    )
                )  # sort by ratio ascending
                # 	 DFS list stack explores highest ratio unpruned node next
                for task in eligible:
                    tid = task["id"]
                    duration = task["duration"]
                    deadline = task["deadline"]
                    task_reward = task["reward"]

                    new_time = curr_time + duration
                    earned = task_reward if new_time <= deadline else 0.0

                    new_scheduled = scheduled | {tid}
                    new_schedule = schedule + [tid]
                    new_reward = reward + earned

                    child_ub = self._compute_upper_bound(
                        new_time, new_scheduled, new_reward
                    )

                    if (
                        child_ub <= self.best_reward
                    ):  # Only add nodes to the stack that beat the incumbent
                        self.nodes_pruned += 1
                        continue

                    stack.append(
                        (new_time, new_scheduled, new_schedule, new_reward, child_ub)
                    )

        except TimeoutException:
            timed_out = True

        end_time = time.perf_counter()

        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_ms = (end_time - start_time) * 1000.0

        # Compute schedule task details
        task_details = []
        curr_t = 0

        for tid in self.best_schedule:
            task = self.tasks[tid]

            start = curr_t
            completion = curr_t + task["duration"]
            curr_t = completion

            deadline = task["deadline"]
            earned = task["reward"] if completion <= deadline else 0.0

            task_details.append(
                {
                    "task_id": tid,
                    "start_time": start,
                    "completion_time": completion,
                    "deadline": deadline,
                    "earned_reward": earned,
                }
            )

        result = {
            "algorithm_name": "Branch-and-Bound",
            "dataset_name": self.dataset_name,
            "task_timing_and_rewards": task_details,
            "total_reward": self.best_reward,
            "runtime_ms": elapsed_ms,
            "peak_memory_kb": peak_mem / 1024.0,
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
        }

        return result, timed_out


if __name__ == "__main__":
    verify_os()
    parser = argparse.ArgumentParser(
        description="Run BranchAndBoundSolver across datasets with official output requirements."
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds per dataset instance (default: 20.0s).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="csv",
        help="Output as csv, JSON, text, or verbose.",
    )

    args = parser.parse_args()

    if args.output == "csv":
        print(
            "Algorithm,Dataset,Task|Start Time|Completion Time|Deadline|Earned Reward,Total Reward,Runtime (ms),Peak Memory (KB),Nodes Explored,Nodes Pruned"
        )

    elif args.output == "text":
        print(
            f"{'Dataset Name':<35} | {'Reward':<8} | {'Nodes Exp':<10} | {'Pruned':<8} | {'Time (ms)':<10} | {'Memory (KB)'}"
        )
        print("-" * 95)

    for name in sorted(ALL_DATASETS.keys()):
        try:
            tasks = get_dataset(name)

            solver = BranchAndBoundSolver(tasks, dataset_name=name)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, args.timeout)
            try:
                res, timed_out = solver.solve()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

            display_name = f"{name} (timed out)" if timed_out else name

            if args.output == "json":
                result_json = json.dumps(res, indent=2)
                print(f"{result_json}\n")

            elif args.output == "csv":
                output = "B&B," + display_name + ","

                for task in res["task_timing_and_rewards"]:
                    output += f"{task['task_id']}|"
                    output += f"{task['start_time']}|"
                    output += f"{task['completion_time']}|"
                    output += f"{task['deadline']}|"
                    output += f"{task['earned_reward']}|"

                if len(res["task_timing_and_rewards"]) > 0:
                    output = output[:-1]

                output += f",{res['total_reward']},"
                output += f"{res['runtime_ms']},"
                output += f"{res['peak_memory_kb']},"
                output += f"{res['nodes_explored']},"
                output += f"{res['nodes_pruned']}"

                print(output)

            else:
                if args.output == "verbose":
                    print(
                        f"{'Dataset Name':<35} | {'Reward':<8} | {'Nodes Exp':<10} | {'Pruned':<8} | {'Time (ms)':<10} | {'Memory (KB)'}"
                    )

                display_name = f"{name} (timed out)" if timed_out else name

                print(
                    f"{display_name:<35} | "
                    f"{res['total_reward']:<8.1f} | "
                    f"{res['nodes_explored']:<10} | "
                    f"{res['nodes_pruned']:<8} | "
                    f"{res['runtime_ms']:<10.3f} | "
                    f"{res['peak_memory_kb']:.2f}"
                )

                if args.output == "verbose":
                    print("    - Task Breakdown:")

                    for item in res["task_timing_and_rewards"]:
                        status = (
                            "LATE (0 reward)"
                            if item["completion_time"] > item["deadline"]
                            else "ON TIME"
                        )

                        print(
                            f"       Task {item['task_id']}: "
                            f"Start={item['start_time']}, "
                            f"Completion={item['completion_time']}, "
                            f"Deadline={item['deadline']}, "
                            f"Earned Reward={item['earned_reward']} [{status}]"
                        )

                    print("-" * 95)

        except Exception as exc:
            print(f"{name:<35} | ERROR: {exc}")
