# visual_grid_game.py
import random
import tkinter as tk
from collections import deque

class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""
    
    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None, num_traps=5):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        
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

    def get_percept(self) -> dict:
        return {
            'agent_pos': list(self.agent_pos),
            'opponent_positions': [list(op) for op in self.opponents],
            'smells_food': tuple(self.agent_pos) in self.food_positions,
            'hit_wall': tuple(self.agent_pos) in self.walls,
            'smells_toxin': tuple(self.agent_pos) in self.toxic_traps,
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions),
            'grid_size': (self.width, self.height),
            'walls': list(self.walls),
            'all_food': list(self.food_positions),
        }

    def bfs_pathfind(self, start, goal):
        """BFS to find shortest path avoiding walls."""
        queue = deque([(start, [])])
        visited = {tuple(start)}
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return path
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                new_x = current[0] + dx
                new_y = current[1] + dy
                
                # Check bounds and walls
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and 
                    (new_x, new_y) not in self.walls and
                    (new_x, new_y) not in visited):
                    
                    visited.add((new_x, new_y))
                    new_path = path + [(new_x, new_y)]
                    queue.append(((new_x, new_y), new_path))
        
        return []  # No path found

    def find_next_action(self):
        """Find next action towards nearest reachable food."""
        if not self.food_positions:
            return random.choice(['Up', 'Down', 'Left', 'Right'])
        
        current_pos = tuple(self.agent_pos)
        best_path = None
        best_food = None
        
        # Find nearest food with valid path
        for food in self.food_positions:
            path = self.bfs_pathfind(current_pos, food)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
                best_food = food
        
        if not best_path:
            return random.choice(['Up', 'Down', 'Left', 'Right'])
        
        # Get next position in path
        next_pos = best_path[0]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx > 0:
            return 'Right'
        elif dx < 0:
            return 'Left'
        elif dy > 0:
            return 'Up'
        else:
            return 'Down'

    def execute_action(self, action: str):
        self.steps += 1
        new_pos = list(self.agent_pos)
        
        if action == 'Up':
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 'Down':
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 'Left':
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 'Right':
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)
            
        # Check collision with walls
        if tuple(new_pos) in self.walls:
            self.score -= 5
        else:
            self.agent_pos = new_pos
            
        tuple_pos = tuple(self.agent_pos)
        
        # Check if eating food
        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20
            
        # Check if stepped on toxic trap
        if tuple_pos in self.toxic_traps:
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
        return len(self.food_positions) == 0 or self.steps >= 300 or self.collision

class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""
    
    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")
        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents, custom_walls=walls)
        
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

    def run_loop(self):
        self.btn.config(state="disabled")
        
        def step():
            if not self.env.is_done():
                action = self.env.find_next_action()
                self.env.execute_action(action)
                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Food Left: {len(self.env.food_positions)}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")
                
        step()

if __name__ == "__main__":
    root = tk.Tk()
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0)
    root.mainloop()