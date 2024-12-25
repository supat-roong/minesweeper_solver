from dataclasses import dataclass
import numpy as np
from typing import Tuple, List, Dict, Optional
from custom_dataclass import CellData, Move, CellInfo, ConstraintGroup
from config import CELL_STATE
import random


class MinesweeperSolver:
    """A solver for Minesweeper that uses probability-based analysis to make optimal moves.
        
        Uses a three-pass probability calculation approach:
        1. Initialize known cell states
        2. Apply direct neighbor constraints
        3. Refine using overlapping constraints
        
        Attributes:
            game_state: Current game board state as 2D numpy array
            cell_data: Mapping of grid coordinates to cell metadata
            rows, cols: Grid dimensions
            total_mines: Total mines in the game
            mine_probabilities: Calculated mine probabilities for each cell
            constraint_groups: Groups of cells with known mine counts
    """
    
    def __init__(self, game_state: np.ndarray, cell_data: Dict[Tuple[int, int], CellData], total_mines: int):
        """Initialize the solver with the current game state and parameters."""
        self.game_state = np.array(game_state)
        self.cell_data = cell_data
        self.rows, self.cols = game_state.shape
        self.total_mines = total_mines
        self.mine_probabilities = np.empty((self.rows, self.cols), dtype=float)
        self.constraint_groups: List[ConstraintGroup] = []
        np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cell coordinates within the game grid.
        
        Args:
            row: Row index of target cell
            col: Column index of target cell
            
        Returns:
            List[Tuple[int, int]]: List of valid (row, col) coordinates for all 8 adjacent cells
                                  within grid boundaries
        """
        neighbors = []
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                neighbors.append((new_row, new_col))
        return neighbors
    
    def get_numbered_neighbors(self, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get neighbors containing mine count numbers.
        
        Args:
            row: Row index of target cell
            col: Column index of target cell
            
        Returns:
            List[Tuple[int, int, int]]: List of (row, col, number) tuples for neighboring cells
                                       that contain mine count numbers > 0
        """
        return [(r, c, self.game_state[r, c]) 
                for r, c in self.get_neighbors(row, col)
                if self.game_state[r, c] > 0]
    
    def _get_cell_info(self, row: int, col: int) -> CellInfo:
        """Get neighbor and mine information for a cell.
        
        Args:
            row: Row index of target cell
            col: Column index of target cell
            
        Returns:
            CellInfo: Object containing:
                - unknown_neighbors: List of unopened neighbor coordinates
                - flagged_neighbors: List of flagged neighbor coordinates
                - remaining_mines: Number of unfound mines around cell
        """
        neighbors = self.get_neighbors(row, col)
        unknown_neighbors = [(r, c) for r, c in neighbors 
                           if self.game_state[r, c] == CELL_STATE.unopened]
        flagged_neighbors = [(r, c) for r, c in neighbors 
                           if self.game_state[r, c] == CELL_STATE.flag]
        number = self.game_state[row, col]
        remaining_mines = number - len(flagged_neighbors)
        
        return CellInfo(unknown_neighbors, flagged_neighbors, remaining_mines)
    
    def calculate_probabilities(self) -> None:
        """Execute all probability calculation passes.
        
        Performs three-pass probability calculation:
        1. Initialize known states
        2. Apply direct constraints
        3. Refine overlapping constraints
        
        Functions:
            Updates self.mine_probabilities matrix
        """
        self._initialize_probabilities()
        self._apply_direct_constraints()
        self._refine_overlapping_constraints()
        self._handle_remaining_cells()
        print("Mine probabilities:")
        print(self.mine_probabilities)

    def _initialize_probabilities(self) -> None:
        """Set initial probabilities based on known cell states.
        
        Functions:
            Initializes self.mine_probabilities with:
            - 0.0 for revealed cells
            - 1.0 for flagged cells
            - -1.0 for unknown cells
        """
        self.mine_probabilities = np.full((self.rows, self.cols), -1.0)
        
        mask_revealed = self.game_state >= 0
        mask_flagged = self.game_state == CELL_STATE.flag
        
        self.mine_probabilities[mask_revealed] = 0
        self.mine_probabilities[mask_flagged] = 1

    def _apply_direct_constraints(self) -> None:
        """Calculate initial probabilities from neighbor constraints.
        
        Functions:
            - Updates self.mine_probabilities for cells with direct constraints
            - Populates self.constraint_groups list
        """
        self.constraint_groups = []
        
        numbered_cells = np.argwhere(self.game_state > 0)
        for row, col in numbered_cells:
            cell_info = self._get_cell_info(row, col)
            
            if cell_info.unknown_neighbors:
                self.constraint_groups.append(ConstraintGroup(
                    cells=cell_info.unknown_neighbors,
                    mines=cell_info.remaining_mines
                ))
                
                prob = cell_info.remaining_mines / len(cell_info.unknown_neighbors)
                for r, c in cell_info.unknown_neighbors:
                    if self.mine_probabilities[r, c] == -1:
                        self.mine_probabilities[r, c] = prob
                    else:
                        self.mine_probabilities[r, c] = min(
                            self.mine_probabilities[r, c],
                            prob
                        )

    def _refine_overlapping_constraints(self) -> None:
        """Refine probabilities using overlapping neighbor information.
        
        Functions:
            Updates self.mine_probabilities based on overlapping constraint analysis
        """
        unknown_cells = np.argwhere(self.game_state == CELL_STATE.unopened)
        for row, col in unknown_cells:
            numbered_neighbors = self.get_numbered_neighbors(row, col)
            if len(numbered_neighbors) > 1:
                min_prob, max_prob = self._calculate_probability_bounds(row, col, numbered_neighbors)
                
                if self.mine_probabilities[row, col] != -1:
                    self.mine_probabilities[row, col] = np.clip(
                        self.mine_probabilities[row, col],
                        min_prob,
                        max_prob
                    )

    def _calculate_probability_bounds(
        self, 
        row: int, 
        col: int, 
        numbered_neighbors: List[Tuple[int, int, int]]
    ) -> Tuple[float, float]:
        """Calculate minimum and maximum possible mine probabilities.
        
        Args:
            row: Row index of target cell
            col: Column index of target cell
            numbered_neighbors: List of (row, col, number) tuples for neighboring cells
                              containing mine count numbers
            
        Returns:
            Tuple[float, float]: (min_probability, max_probability) bounds for target cell,
                                where both values are in range [0.0, 1.0]
        """
        min_prob, max_prob = 0.0, 1.0
        
        for n_row, n_col, number in numbered_neighbors:
            cell_info = self._get_cell_info(n_row, n_col)
            other_unknowns = len(cell_info.unknown_neighbors) - 1
            
            if other_unknowns < cell_info.remaining_mines:
                min_prob = max(min_prob, cell_info.remaining_mines - other_unknowns)
            
            max_prob = min(max_prob, cell_info.remaining_mines)
        
        return min_prob, max_prob

    def _handle_remaining_cells(self) -> None:
        """Assign probabilities to cells without direct constraints.
        
        Functions:
            Updates self.mine_probabilities for uncalculated cells by evenly
            distributing remaining mines
        """
        uncalculated_mask = self.mine_probabilities == -1
        uncalculated_count = np.sum(uncalculated_mask)
        
        if uncalculated_count == 0:
            return
            
        assigned_mines = np.sum(self.mine_probabilities[self.mine_probabilities > 0])
        remaining_mines = self.total_mines - assigned_mines
        base_probability = remaining_mines / uncalculated_count
        
        self.mine_probabilities[uncalculated_mask] = base_probability

    def get_best_move(self) -> Move:
        """Determine optimal next move based on mine probabilities.
        
        Args:
            None
            
        Returns:
            Move: Object containing:
                - row, col: Grid coordinates for the move
                - action: One of ['flag', 'click', 'none']
                - probability: Mine probability at selected position
                
        Note:
            Priority order:
            1. Flag 100% mine probability
            2. Click 0% mine probability  
            3. Click lowest non-zero probability
        """
        self.calculate_probabilities()
        
        definite_mines = np.argwhere(
            (self.game_state == CELL_STATE.unopened) & 
            (np.abs(self.mine_probabilities - 1.0) < 1e-6)
        )
        if len(definite_mines) > 0:
            row, col = definite_mines[0]
            return Move(row, col, 'flag', 1.0)
        
        unopened_mask = self.game_state == CELL_STATE.unopened
        if not np.any(unopened_mask):
            return Move(0, 0, 'none', 0)
        
        probs = np.ma.masked_array(
            self.mine_probabilities,
            ~unopened_mask
        )
        min_prob = np.min(probs)
        
        best_positions = np.argwhere(
            unopened_mask & 
            (np.abs(self.mine_probabilities - min_prob) < 1e-6)
        )
        
        if min_prob < 1e-6:
            row, col = best_positions[0]
            return Move(row, col, 'click', min_prob)
        
        row, col = random.choice(best_positions)
        return Move(row, col, 'click', min_prob)

    def get_screen_coords(self, move: Move) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates.
        
        Args:
            move: Move object containing:
                - row: Grid row index
                - col: Grid column index
            
        Returns:
            Tuple[int, int]: (screen_x, screen_y) pixel coordinates for the specified move
        """
        cell_data = self.cell_data[(move.row, move.col)]
        return cell_data.position.screen_x, cell_data.position.screen_y