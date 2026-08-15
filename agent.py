# agent.py
import random
from collections import deque
import heapq
from typing import Tuple, List, Set, Dict

class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    SearchAgent that plans to the nearest food pellet using BFS, DFS, or UCS.
    - self.plan: list of action strings (e.g. ['Up','Left',...'])
    - self.active_algo: one of 'BFS', 'DFS', 'UCS'
    """

    DELTAS = {
        'Up': (0, -1),
        'Down': (0, 1),
        'Left': (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        self.plan: List[str] = []
        self.active_algo: str = 'BFS'  # change to 'DFS' or 'UCS' to observe differences

    def sense_and_act(self, percept: dict) -> str:
        # If we already have a plan, execute the next action
        if self.plan:
            return self.plan.pop(0)

        start = tuple(percept.get('agent_pos'))
        all_food = [tuple(f) for f in percept.get('all_food', [])]
        if not all_food:
            return 'Stop'  # nothing to do

        # choose the closest food by Manhattan distance as the goal
        def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        goal = min(all_food, key=lambda f: manhattan(start, f))

        walls: Set[Tuple[int,int]] = set(tuple(w) for w in percept.get('walls', []))
        grid_size = percept.get('grid_size')  # optional (width, height) tuple

        if self.active_algo == 'BFS':
            plan = self._bfs(start, goal, walls, grid_size)
        elif self.active_algo == 'DFS':
            plan = self._dfs(start, goal, walls, grid_size)
        elif self.active_algo == 'UCS':
            plan = self._ucs(start, goal, walls, grid_size)
        else:
            plan = []

        self.plan = plan
        return self.plan.pop(0) if self.plan else 'Stop'

    def _in_bounds(self, pos: Tuple[int,int], grid_size) -> bool:
        if grid_size is None:
            return True
        w, h = grid_size
        x, y = pos
        return 0 <= x < w and 0 <= y < h

    def _neighbors(self, pos: Tuple[int,int], walls: Set[Tuple[int,int]], grid_size):
        for action, delta in self.DELTAS.items():
            nx, ny = pos[0] + delta[0], pos[1] + delta[1]
            npos = (nx, ny)
            if npos in walls:
                continue
            if not self._in_bounds(npos, grid_size):
                continue
            yield npos, action

    def _reconstruct(self, parent: Dict[Tuple[int,int], Tuple[Tuple[int,int], str]],
                     start: Tuple[int,int], goal: Tuple[int,int]) -> List[str]:
        actions = []
        cur = goal
        while cur != start:
            if cur not in parent:
                return []  # no path
            prev, action = parent[cur]
            actions.append(action)
            cur = prev
        actions.reverse()
        return actions

    def _bfs(self, start, goal, walls, grid_size) -> List[str]:
        q = deque([start])
        visited = {start}
        parent: Dict[Tuple[int,int], Tuple[Tuple[int,int], str]] = {}
        while q:
            cur = q.popleft()
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            for npos, action in self._neighbors(cur, walls, grid_size):
                if npos not in visited:
                    visited.add(npos)
                    parent[npos] = (cur, action)
                    q.append(npos)
        return []

    def _dfs(self, start, goal, walls, grid_size) -> List[str]:
        stack = [start]
        visited = set()
        parent: Dict[Tuple[int,int], Tuple[Tuple[int,int], str]] = {}
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            # push neighbors in a deterministic order (so behavior is reproducible)
            for npos, action in reversed(list(self._neighbors(cur, walls, grid_size))):
                if npos not in visited:
                    parent[npos] = (cur, action)
                    stack.append(npos)
        return []

    def _ucs(self, start, goal, walls, grid_size) -> List[str]:
        # Uniform-cost search with unit step costs (behaves like BFS on unweighted grid).
        pq = []
        heapq.heappush(pq, (0, start))
        cost_so_far = {start: 0}
        parent: Dict[Tuple[int,int], Tuple[Tuple[int,int], str]] = {}
        while pq:
            cost, cur = heapq.heappop(pq)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            if cost > cost_so_far.get(cur, float('inf')):
                continue
            for npos, action in self._neighbors(cur, walls, grid_size):
                new_cost = cost + 1  # all step costs = 1; replace if you have weighted moves
                if new_cost < cost_so_far.get(npos, float('inf')):
                    cost_so_far[npos] = new_cost
                    parent[npos] = (cur, action)
                    heapq.heappush(pq, (new_cost, npos))
        return []

# Observation:
# Change SearchAgent().active_algo between 'BFS', 'DFS', and 'UCS' and run the simulation.
# DFS will often produce winding, erratic paths; BFS and UCS give direct/optimal paths on an unweighted grid.