# How to run `branch_and_bound.py`

## Signal timeout will not work on Windows unless you run in WSL or a VM.

Please run the program from a Linux environment.

Run it from the same directory as the `test_datasets.py` file. 

You can run it with arguments:

- `--timeout X`
- `--traversal Y`
- `--output Z`

Where:
- **X** is the time in seconds until the solver times out for a specific dataset.
- **Y** is `best`, `bfs`, or `dfs` for different ways to traverse the tree.
- **Z** is `json`, `text`, `csv`, or `verbose` for different output formats.

The default timeout is 20 seconds, and the default output is as a CSV.

---

# How does `branch_and_bound.py` work?

## State Definition

I defined the states as:

- **curr_time:** Current time of the state.
- **scheduled:** A frozenset of scheduled task IDs for O(1) lookup.
- **schedule:** The order of task choices leading to this state.
- **reward:** The reward gained from the scheduled tasks.
- **ub:** The knapsack upper bound of the current state.

## Branching

Branching builds the state space tree of the problem.

1. The knapsack upper bound of the current state is compared with the current incumbent, and the node is pruned if its upper bound is worse than the incumbent.
2. Then the process discovers eligible tasks, which are tasks that have yet to be scheduled from the previous decisions.
3. An eligible task is any task where all dependencies have already been scheduled.
4. Then eligible tasks are sorted by a ratio of reward / duration.
5. Finally, child states are added to the stack/queue/heap if their knapsack upper bound is greater than the best reward so far (the incumbent).

Sorting by reward / duration puts tasks with a higher reward : duration ratio higher on the stack, this sorting is reversed for the queue, and unnecessary for a minheap.

Child states' upper bounds are calculated before adding them to the stack/queue/heap even though they will be evaluated again when they are popped from the stack/queue/heap:

- Checking upper bound before adding to the stack/queue/heap helps to conserve memory.
- It is necessary to check upper bound after popping from the stack/queue/heap because the incumbent was checked when the node was added to the stack/queue/heap, and the current incumbent may differ.

Using a stack gives the algorithm a DFS traversal order.

Using a queue gives the algorithm a BFS traversal order.

Using a minheap sorted by negative upper bound gives the algorithm a best-first-search traversal order.

The algorithm ends when:

- **There are no remaining states in the stack/queue/heap:**
  - All branches have either been pruned or evaluated, and a solution has been found.
- **The timeout time has elapsed:**
  - The incumbent solution is returned.

## Bounding

Bounding was done using a fractional knapsack upper bound.

1. The tasks which could be completed by their deadline were sorted by ratio of reward to duration descending.
2. The latest deadline in this set is considered the `max_deadline`.
3. Then tasks with the highest ratio are added to the `potential_gain` until this new schedule reaches the `max_deadline`.
4. Finally, the `potential_gain` is added to the current reward of the state: This is our knapsack upper bound.

## Timeout

Timeout was implemented with Python's `signal` library.

## Credits

- `branch_and_bound.py` - Developed by Ethan Koehler.
- `test_datasets.py` - Created by Tarun Sai Naregudam
- `greedy.py` - Created by Ashlan Garner