import numpy as np
from typing import Tuple, List, Dict
from custom_dataclass import CellData, Move
import random

class MinesweeperSolver:
    def __init__(self, game_state: np.ndarray, cell_data: Dict[Tuple[int, int], CellData], total_mines: int):
        self.game_state = np.array(game_state)
        self.cell_data = cell_data
        self.rows, self.cols = game_state.shape
        self.mine_probabilities = np.empty((self.rows, self.cols), dtype=int)
        self.total_mines = total_mines
        np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})
        
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cell coordinates."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    neighbors.append((new_row, new_col))
        return neighbors
    
    def get_numbered_neighbors(self, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get neighboring cells that have numbers, returns (row, col, number)."""
        numbered_neighbors = []
        for neighbor_row, neighbor_col in self.get_neighbors(row, col):
            cell_value = self.game_state[neighbor_row, neighbor_col]
            if cell_value > 0:  # If it's a numbered cell
                numbered_neighbors.append((neighbor_row, neighbor_col, cell_value))
        return numbered_neighbors
    
    def calculate_probabilities(self) -> None:
        """Calculate mine probabilities for all cells using neighbor constraints.
        
        This method uses a three-pass approach:
        1. Initialize probabilities and handle known cells
        2. Calculate direct neighbor constraints
        3. Refine probabilities using overlapping constraints
        """
        self._initialize_probabilities()
        self._apply_direct_constraints()
        self._refine_overlapping_constraints()
        self._handle_remaining_cells()
        print("Mine probabilities:")
        print(self.mine_probabilities)

    def _initialize_probabilities(self) -> None:
        """Initialize probability matrix and handle known cells."""
        self.mine_probabilities = np.full((self.rows, self.cols), -1.0)
        
        # Set known probabilities for revealed and flagged cells
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i, j] >= 0:  # Revealed number or empty
                    self.mine_probabilities[i, j] = 0
                elif self.game_state[i, j] == -2:  # Flagged
                    self.mine_probabilities[i, j] = 1

    def _get_cell_info(self, row: int, col: int) -> dict:
        """Get information about a cell's neighbors.
        
        Returns:
            dict: Contains lists of unknown and flagged neighbors, and remaining mines
        """
        neighbors = self.get_neighbors(row, col)
        unknown_neighbors = [(r, c) for r, c in neighbors if self.game_state[r, c] == -1]
        flagged_neighbors = [(r, c) for r, c in neighbors if self.game_state[r, c] == -2]
        number = self.game_state[row, col]
        remaining_mines = number - len(flagged_neighbors)
        
        return {
            'unknown_neighbors': unknown_neighbors,
            'flagged_neighbors': flagged_neighbors,
            'remaining_mines': remaining_mines
        }

    def _apply_direct_constraints(self) -> None:
        """First pass: Apply direct neighbor constraints to calculate initial probabilities."""
        self.constraint_groups = []  # Store for later refinement
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i, j] > 0:  # Numbered cell
                    cell_info = self._get_cell_info(i, j)
                    unknown_neighbors = cell_info['unknown_neighbors']
                    remaining_mines = cell_info['remaining_mines']
                    
                    if unknown_neighbors:
                        self.constraint_groups.append({
                            'cells': unknown_neighbors,
                            'mines': remaining_mines
                        })
                        
                        # Update initial probabilities
                        prob = remaining_mines / len(unknown_neighbors)
                        for r, c in unknown_neighbors:
                            if self.mine_probabilities[r, c] == -1:
                                self.mine_probabilities[r, c] = prob
                            else:
                                self.mine_probabilities[r, c] = min(
                                    self.mine_probabilities[r, c],
                                    prob
                                )

    def _refine_overlapping_constraints(self) -> None:
        """Second pass: Refine probabilities using overlapping constraints."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i, j] == -1:  # Unopened cell
                    numbered_neighbors = self.get_numbered_neighbors(i, j)
                    if len(numbered_neighbors) > 1:
                        min_prob, max_prob = self._calculate_probability_bounds(i, j, numbered_neighbors)
                        
                        # Update probability based on refined constraints
                        if self.mine_probabilities[i, j] != -1:
                            self.mine_probabilities[i, j] = np.clip(
                                self.mine_probabilities[i, j],
                                min_prob,
                                max_prob
                            )

    def _calculate_probability_bounds(
        self, row: int, col: int, numbered_neighbors: list
    ) -> tuple[float, float]:
        """Calculate minimum and maximum probability bounds for a cell.
        
        Args:
            row: Cell row
            col: Cell column
            numbered_neighbors: List of neighboring numbered cells
            
        Returns:
            tuple: (minimum probability, maximum probability)
        """
        max_possible_probability = 1.0
        min_possible_probability = 0.0
        
        for n_row, n_col, number in numbered_neighbors:
            cell_info = self._get_cell_info(n_row, n_col)
            unknown_neighbors = cell_info['unknown_neighbors']
            remaining_mines = cell_info['remaining_mines']
            
            # Calculate minimum required mines around current cell
            other_unknowns = len(unknown_neighbors) - 1  # Exclude current cell
            if other_unknowns < remaining_mines:
                min_possible_probability = max(
                    min_possible_probability,
                    (remaining_mines - other_unknowns)
                )
            
            # Calculate maximum possible mines around current cell
            max_possible_probability = min(
                max_possible_probability,
                remaining_mines
            )
        
        return min_possible_probability, max_possible_probability

    def _handle_remaining_cells(self) -> None:
        """Third pass: Handle cells with no calculated probabilities yet.
        Uses the total remaining mines divided by number of uncalculated cells."""
        # Count cells that still have -1 probability (no calculated probability yet)
        uncalculated_cells = np.sum(self.mine_probabilities == -1)
        if uncalculated_cells == 0:
            return
            
        # Count remaining mines (subtract the sum of all positive probabilities)
        assigned_mines = np.sum(self.mine_probabilities[self.mine_probabilities > 0])
        remaining_mines = self.total_mines - assigned_mines
        
        # Calculate and apply base probability
        base_probability = remaining_mines / uncalculated_cells
        self.mine_probabilities[self.mine_probabilities == -1] = base_probability

    def get_best_move(self) -> Move:
        """Determine the best next move.
        
        Priority order:
        1. Flag cells with 100% mine probability
        2. Click cells with 0% mine probability
        3. Click the cell with lowest mine probability
        
        Returns:
            Move: The best move to make next
        """
        self.calculate_probabilities()
        
        # First priority: Flag cells that definitely contain mines
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.game_state[i, j] == -1 and  # Unopened cell
                    abs(self.mine_probabilities[i, j] - 1.0) < 1e-6):  # Probability ≈ 1
                    return Move(i, j, 'flag', 1.0)
        
        # Second priority: Click safe cells
        best_probability = float('inf')
        best_pos = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i, j] == -1:  # Unopened cell
                    prob = self.mine_probabilities[i, j]
                    if prob < best_probability:
                        best_probability = self.mine_probabilities[i, j]
                        best_pos = [(i, j)]
                    elif abs(prob - best_probability) < 1e-6:
                        best_pos.append((i, j))
                    
                    # If we find a definitely safe move, return it immediately
                    if abs(prob) < 1e-6:  # Probability ≈ 0
                        return Move(best_pos[0][0], best_pos[0][1], 'click', prob)
        
        if not best_pos:
            return Move(0, 0, 'none', 0)
        random_best_pos = random.choice(best_pos)
        return Move(random_best_pos[0], random_best_pos[1], 'click', best_probability)

    def get_screen_coords(self, move: Move) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates."""
        cell_data = self.cell_data[(move.row, move.col)]
        return cell_data.position.screen_x, cell_data.position.screen_y