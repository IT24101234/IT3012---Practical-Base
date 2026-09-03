# agent.py
import random
import math
from collections import deque
import heapq
from typing import Tuple, List, Set, Dict, Any

from logic_engine import KnowledgeBase


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Pure condition-action rules based only on the immediate percept."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here') or percept.get('smells_food'):
            return random.choice(['Up', 'Down', 'Left', 'Right'])
        if percept.get('wall_ahead') or percept.get('hit_wall'):
            return random.choice(['Left', 'Right', 'Down', 'Up'])
        return random.choice(['Up', 'Down', 'Left', 'Right'])


class ModelBasedAgent:
    """A memory-based agent that avoids repeating the same failed action."""

    def __init__(self):
        self.last_percept = None
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if self.last_percept == percept and self.last_action is not None:
            options = ['Up', 'Down', 'Left', 'Right']
            for action in options:
                if action != self.last_action:
                    self.last_percept = percept.copy()
                    self.last_action = action
                    return action
            fallback = random.choice(options)
            self.last_percept = percept.copy()
            self.last_action = fallback
            return fallback

        action = random.choice(['Up', 'Down', 'Left', 'Right'])
        if percept.get('wall_ahead') or percept.get('hit_wall'):
            action = random.choice(['Left', 'Right', 'Down', 'Up'])

        self.last_percept = percept.copy()
        self.last_action = action
        return action


class SearchAgent:
    """
    SearchAgent that plans to the nearest food pellet using BFS, DFS, UCS, or A*.
    Also integrates a logic-based feasibility check using the Knowledge Base.
    """

    DELTAS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        self.plan: List[str] = []
        self.active_algo: str = 'BFS'
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _tile_fact_set(self, pos, tile_percepts=None):
        facts = set()
        if tile_percepts is None:
            tile_percepts = {}
        for fact in tile_percepts.get('facts', []):
            facts.add(fact)
        if tile_percepts.get('TargetVisible'):
            facts.add('TargetVisible')
        if tile_percepts.get('HasDust'):
            facts.add('HasDust')
        if tile_percepts.get('BloodseekerMissing'):
            facts.add('BloodseekerMissing')
        return facts

    def _is_tile_infeasible(self, pos, tile_percepts=None):
        self.kb.clear_facts()
        for fact in self._tile_fact_set(pos, tile_percepts):
            self.kb.tell_fact(fact)
        self.kb.forward_chain()
        return 'Retreat' in self.kb.facts

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', tile_percepts=None):
        heuristic = self.euclidean_distance if heuristic_type == 'euclidean' else self.manhattan_distance
        priority_queue = []
        reached_states = set()
        initial_h_cost = heuristic(start_pos, goal_pos)
        heapq.heappush(priority_queue, (initial_h_cost, 0, start_pos, []))

        while priority_queue:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(priority_queue)
            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for neighbor_pos, action in self._neighbors(current_pos, walls, grid_size):
                if self._is_tile_infeasible(neighbor_pos, tile_percepts):
                    continue
                if neighbor_pos in reached_states:
                    continue
                new_g_cost = g_cost + 1
                new_h_cost = heuristic(neighbor_pos, goal_pos)
                new_f_cost = new_g_cost + new_h_cost
                new_path = path_taken + [action]
                heapq.heappush(priority_queue, (new_f_cost, new_g_cost, neighbor_pos, new_path))
        return []

    def sense_and_act(self, percept: dict) -> str:
        if self.plan:
            return self.plan.pop(0)

        start = tuple(percept.get('agent_pos'))
        all_food = [tuple(f) for f in percept.get('all_food', [])]
        if not all_food:
            return 'Stop'

        def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        goal = min(all_food, key=lambda f: manhattan(start, f))
        walls: Set[Tuple[int, int]] = set(tuple(w) for w in percept.get('walls', []))
        grid_size = percept.get('grid_size')
        tile_percepts = percept.get('tile_percepts', {})

        if self.active_algo == 'BFS':
            plan = self.bfs_search(start, goal, walls, grid_size, tile_percepts)
        elif self.active_algo == 'DFS':
            plan = self._dfs(start, goal, walls, grid_size, tile_percepts)
        elif self.active_algo == 'UCS':
            plan = self._ucs(start, goal, walls, grid_size, tile_percepts)
        elif self.active_algo == 'AStar':
            plan = self.astar_search(start, goal, walls, grid_size, tile_percepts=tile_percepts)
        else:
            plan = []

        self.plan = plan
        return self.plan.pop(0) if self.plan else 'Stop'

    def bfs_search(self, start, goal, walls, grid_size, tile_percepts=None):
        q = deque([start])
        visited = {start}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while q:
            cur = q.popleft()
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            for npos, action in self._neighbors(cur, walls, grid_size):
                if self._is_tile_infeasible(npos, tile_percepts):
                    continue
                if npos not in visited:
                    visited.add(npos)
                    parent[npos] = (cur, action)
                    q.append(npos)
        return []

    def _in_bounds(self, pos: Tuple[int, int], grid_size) -> bool:
        if grid_size is None:
            return True
        w, h = grid_size
        x, y = pos
        return 0 <= x < w and 0 <= y < h

    def _neighbors(self, pos: Tuple[int, int], walls: Set[Tuple[int, int]], grid_size):
        for action, delta in self.DELTAS.items():
            nx, ny = pos[0] + delta[0], pos[1] + delta[1]
            npos = (nx, ny)
            if npos in walls:
                continue
            if not self._in_bounds(npos, grid_size):
                continue
            yield npos, action

    def _reconstruct(self, parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]],
                     start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        actions = []
        cur = goal
        while cur != start:
            if cur not in parent:
                return []
            prev, action = parent[cur]
            actions.append(action)
            cur = prev
        actions.reverse()
        return actions

    def _dfs(self, start, goal, walls, grid_size, tile_percepts=None) -> List[str]:
        stack = [start]
        visited = set()
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            for npos, action in reversed(list(self._neighbors(cur, walls, grid_size))):
                if self._is_tile_infeasible(npos, tile_percepts):
                    continue
                if npos not in visited:
                    parent[npos] = (cur, action)
                    stack.append(npos)
        return []

    def _ucs(self, start, goal, walls, grid_size, tile_percepts=None) -> List[str]:
        pq = []
        heapq.heappush(pq, (0, start))
        cost_so_far = {start: 0}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while pq:
            cost, cur = heapq.heappop(pq)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            if cost > cost_so_far.get(cur, float('inf')):
                continue
            for npos, action in self._neighbors(cur, walls, grid_size):
                if self._is_tile_infeasible(npos, tile_percepts):
                    continue
                new_cost = cost + 1
                if new_cost < cost_so_far.get(npos, float('inf')):
                    cost_so_far[npos] = new_cost
                    parent[npos] = (cur, action)
                    heapq.heappush(pq, (new_cost, npos))
        return []