# visual_grid_game.py
import random
import tkinter as tk

# Clockwise order used for turning: Up -> Right -> Down -> Left -> Up ...
FACING_ORDER = ['Up', 'Right', 'Down', 'Left']
DIRECTION_VECTORS = {
    'Up': (0, 1),
    'Down': (0, -1),
    'Left': (-1, 0),
    'Right': (1, 0),
}


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None, num_traps=5):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.facing = 'Up'       # Step 1.1: agent now has an orientation

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Dynamically generate toxic traps avoiding start, walls, and food
        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            pos_tuple = (tx, ty)
            if (pos_tuple != (0, 0) and
                    pos_tuple not in self.walls and
                    pos_tuple not in self.food_positions):
                self.toxic_traps.add(pos_tuple)

        # Generate adversarial opponents avoiding start, walls, and food
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if (tuple(op_pos) != (0, 0) and
                    tuple(op_pos) not in self.walls and
                    tuple(op_pos) not in self.food_positions):
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    # ---------- Step 1.1: Partial Observability ----------
    def _direction_vector(self):
        return DIRECTION_VECTORS[self.facing]

    def get_percept(self) -> dict:
        """
        Only local, egocentric information is exposed now -- no more global
        (agent_pos / opponent_positions) coordinates. The agent only learns
        what is immediately ahead of it and what is true of its own cell.
        """
        dx, dy = self._direction_vector()
        ahead = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        in_bounds = 0 <= ahead[0] < self.width and 0 <= ahead[1] < self.height
        wall_ahead = (not in_bounds) or (ahead in self.walls)

        current = tuple(self.agent_pos)

        return {
            'wall_ahead': wall_ahead,
            'food_here': current in self.food_positions,
            'toxin_here': current in self.toxic_traps,
            'facing': self.facing,       # proprioception (agent knows its own orientation)
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions),
        }

    # ---------- Actions ----------
    def _rotate(self, step):
        idx = FACING_ORDER.index(self.facing)
        self.facing = FACING_ORDER[(idx + step) % 4]

    def execute_action(self, action: str):
        self.steps += 1

        if action == 'turn_left':
            self._rotate(-1)
        elif action == 'turn_right':
            self._rotate(1)
        elif action == 'move_forward':
            dx, dy = self._direction_vector()
            new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
            in_bounds = 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height
            if not in_bounds or tuple(new_pos) in self.walls:
                self.score -= 5  # bumped into a wall / edge
            else:
                self.agent_pos = new_pos
        elif action == 'suck':
            pos_t = tuple(self.agent_pos)
            if pos_t in self.food_positions:
                self.food_positions.remove(pos_t)
                self.score += 20
        # 'Stay' / unrecognized actions do nothing besides costing a step

        # Check if standing on a toxic trap
        pos_t = tuple(self.agent_pos)
        if pos_t in self.toxic_traps:
            self.score -= 15

        # Move opponents randomly
        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


# ======================================================================
# Step 1.2: Simple Reflex Agent
# ======================================================================
class SimpleReflexAgent:
    """
    Pure condition-action rules. No memory of any kind is kept between
    calls, so this agent has no way of knowing it has been here before.
    In a corner / U-shaped wall it will cycle forever:
        turn_left -> move_forward -> hit wall -> turn_left -> ...
    """

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'suck'
        elif percept['wall_ahead']:
            return 'turn_left'
        else:
            return 'move_forward'


# ======================================================================
# Step 1.3: Model-Based Agent
# ======================================================================
class ModelBasedAgent:
    """
    Keeps an internal model of the world built purely from percepts and its
    own action history (dead reckoning) -- it is NEVER given the true global
    position by the environment. This lets it recognize cells it has already
    visited and break out of loops that trap the SimpleReflexAgent.
    """

    def __init__(self):
        # Internal (believed) state -- starts matching the environment's
        # known starting convention, but is derived only from self-tracking.
        self.believed_pos = (0, 0)
        self.believed_facing = 'Up'
        self.visited_cells = {(0, 0)}
        self.last_action = None

    def _peek_facing(self, step):
        idx = FACING_ORDER.index(self.believed_facing)
        return FACING_ORDER[(idx + step) % 4]

    def _cell_ahead(self, facing):
        dx, dy = DIRECTION_VECTORS[facing]
        return (self.believed_pos[0] + dx, self.believed_pos[1] + dy)

    def _update_state(self):
        """Transition model: integrate the effect of the last action taken."""
        if self.last_action == 'turn_left':
            self.believed_facing = self._peek_facing(-1)
        elif self.last_action == 'turn_right':
            self.believed_facing = self._peek_facing(1)
        elif self.last_action == 'move_forward':
            # We only ever issue move_forward when we believe there's no
            # wall ahead, so in this deterministic world it succeeds.
            self.believed_pos = self._cell_ahead(self.believed_facing)

        self.visited_cells.add(self.believed_pos)

    def sense_and_act(self, percept: dict) -> str:
        # 1. Update internal state/model using last action + new percept
        self._update_state()

        wall_ahead = percept['wall_ahead']
        food_here = percept['food_here']

        # 2. Decide using memory-augmented rules
        if food_here:
            action = 'suck'
        else:
            left_facing = self._peek_facing(-1)
            left_cell = self._cell_ahead(left_facing)
            ahead_cell = self._cell_ahead(self.believed_facing)

            if wall_ahead:
                # IF wall_ahead AND left_is_visited THEN turn_right
                if left_cell in self.visited_cells:
                    action = 'turn_right'
                else:
                    action = 'turn_left'
            elif ahead_cell in self.visited_cells:
                # Already been where we're about to go -- try somewhere new
                if left_cell not in self.visited_cells:
                    action = 'turn_left'
                else:
                    action = 'turn_right'
            else:
                action = 'move_forward'

        self.last_action = action
        return action


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, agent_type='model'):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")
        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents, custom_walls=walls)

        # Choose which kind of agent drives the simulation
        if agent_type == 'simple':
            self.agent = SimpleReflexAgent()
        else:
            self.agent = ModelBasedAgent()

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066", fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        # Draw base grid and walls
        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white", font=("Arial", 8, "bold"))

        # Draw food
        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b", outline="#d97706")

        # Draw toxic traps (purple triangles)
        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.25
            x1 = tx * self.cell_size + offset
            y1 = (self.env.height - 1 - ty) * self.cell_size + offset
            self.canvas.create_polygon(
                x1 + self.cell_size * 0.25, y1,                             # Top point
                x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5,       # Bottom right
                x1, y1 + self.cell_size * 0.5,                              # Bottom left
                fill="#9333ea", outline="#581c87"
            )

        # Draw opponents
        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000", outline="#7a0000")

        # Draw agent
        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066", outline="#1e3a8a")

        # Draw a small facing indicator so you can visually track orientation
        cx = ax * self.cell_size + self.cell_size / 2
        cy = (self.env.height - 1 - ay) * self.cell_size + self.cell_size / 2
        dx, dy = DIRECTION_VECTORS[self.env.facing]
        tip_x = cx + dx * self.cell_size * 0.4
        tip_y = cy - dy * self.cell_size * 0.4  # canvas y grows downward
        self.canvas.create_line(cx, cy, tip_x, tip_y, fill="#ffffff", width=3, arrow=tk.LAST)

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)
                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Facing: {self.env.facing} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    # Try agent_type='simple' to watch the SimpleReflexAgent get stuck in a
    # wall/corner loop, or agent_type='model' (default) to see it escape.
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0, agent_type='simple')
    root.mainloop()