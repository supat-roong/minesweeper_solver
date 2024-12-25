import numpy as np
from typing import Dict, List, Tuple, Optional
from custom_dataclass import CellData, CellPosition, ScreenRegion, Move
from debugger import Debugger
from bomb_counter import BombCounterDetector
from board_detector import BoardDetector
from grid_detector import GridLineDetector
from cell_detector import CellStateDetector

class MinesweeperDetector:
    """Main class for detecting and processing Minesweeper game state from screenshots."""
    
    def __init__(self, grid_size: Tuple[int, int], tesseract_path: Optional[str] = None, debug: bool = False):
        """Initialize the detector with given parameters."""
        self.grid_size = grid_size
        self.debugger = Debugger() if debug else None
        
        # Initialize component detectors
        self.bomb_counter = BombCounterDetector(self.debugger)
        self.board_detector = BoardDetector(grid_size, self.debugger)
        self.grid_detector = GridLineDetector(grid_size, self.debugger)
        self.cell_detector = CellStateDetector(tesseract_path, self.debugger)
        
        self.cell_data = {}
        self.screen_region = None
        self.board_offset_x = 0
        self.board_offset_y = 0

    def process_screenshot(self, screen_region: ScreenRegion) -> dict:
        """Process a Minesweeper screenshot and return game state."""
        self.screen_region = screen_region
        
        # Detect bomb counter
        total_bombs = self.bomb_counter.detect(screen_region.image)
        print(f"Detected bombs: {total_bombs}")
        
        # Detect board region and reset button
        (board, board_offset), reset_button = self.board_detector.detect(screen_region.image)
        self.board_offset_x, self.board_offset_y = board_offset
        
        print(f"Board offset relative to screen region: ({self.board_offset_x}, {self.board_offset_y})")
        
        # Process board state
        game_state, cell_data = self._process_board(board)
        
        if self.debugger is not None:
            self.debugger.visualize_game_state(game_state)
        
        return {
            'total_bombs': total_bombs,
            'remaining_bombs': total_bombs,
            'game_state': game_state.tolist(),
            'cell_data': cell_data,
            'board_image': board,
            'board_offset': board_offset,
            'screen_region': screen_region,
            'reset_button': reset_button
        }

    def _process_board(self, board: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], CellData]]:
        """Process the game board to detect cell states and positions."""
        # Detect grid lines
        horizontal_lines, vertical_lines = self.grid_detector.detect(board)
        
        # Initialize game state and cell data
        game_state = np.zeros(self.grid_size, dtype=int)
        cell_data = {}
        
        # Process each cell
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                # Get cell boundaries
                top = horizontal_lines[i]
                bottom = horizontal_lines[i + 1]
                left = vertical_lines[j]
                right = vertical_lines[j + 1]
                
                # Extract cell with margin
                margin = 3
                cell = board[top + margin:bottom - margin, left + margin:right - margin]
                
                # Detect cell state
                state = self.cell_detector.detect_state(cell, (i, j))
                game_state[i, j] = state
                
                # Calculate absolute screen coordinates
                abs_left = self.screen_region.x + self.board_offset_x + left + margin
                abs_right = self.screen_region.x + self.board_offset_x + right - margin
                abs_top = self.screen_region.y + self.board_offset_y + top + margin
                abs_bottom = self.screen_region.y + self.board_offset_y + bottom - margin
                abs_center_x = self.screen_region.x + self.board_offset_x + (left + right) // 2
                abs_center_y = self.screen_region.y + self.board_offset_y + (top + bottom) // 2
                
                # Create cell position object
                cell_pos = CellPosition(
                    letter=str(state) if state >= 0 else ('-' if state == -1 else 'F'),
                    screen_x_range=(abs_left, abs_right),
                    screen_y_range=(abs_top, abs_bottom),
                    screen_x=abs_center_x,
                    screen_y=abs_center_y,
                    grid_row=i,
                    grid_col=j
                )
                
                cell_data[(i, j)] = CellData(state=state, position=cell_pos)
        
        return game_state, cell_data

    def update_after_move(self, screen_region: ScreenRegion, last_move: Move, game_state: np.ndarray, cell_data: Dict, remaining_bombs: int) -> dict:
        """Update game state after a move."""
        updated_state = game_state.copy()
        
        if last_move.action == 'flag':
            row, col = last_move.row, last_move.col
            updated_state[row, col] = -2
            remaining_bombs -= 1
                
            cell_data[(row, col)] = CellData(
                state=updated_state[row, col],
                position=cell_data[(row, col)].position
            )
            
            return {
                'game_state': updated_state,
                'cell_data': cell_data,
                'remaining_bombs': remaining_bombs,
            }
        
        elif last_move.action == 'click':
            cells_to_check = [(last_move.row, last_move.col)]
            processed_cells = set()
            
            while cells_to_check:
                row, col = cells_to_check.pop(0)
                
                if ((row, col) in processed_cells or 
                    row < 0 or row >= self.grid_size[0] or 
                    col < 0 or col >= self.grid_size[1]):
                    continue
                    
                processed_cells.add((row, col))
                cell_pos = cell_data[(row, col)].position
                
                # Extract and process cell
                cell_img = screen_region.image[
                    cell_pos.screen_y_range[0] - screen_region.y:cell_pos.screen_y_range[1] - screen_region.y,
                    cell_pos.screen_x_range[0] - screen_region.x:cell_pos.screen_x_range[1] - screen_region.x
                ]
                
                new_state = self.cell_detector.detect_state(cell_img, (row, col))
                updated_state[row, col] = new_state
                
                cell_data[(row, col)] = CellData(
                    state=new_state,
                    position=cell_pos
                )
                
                # Check neighbors for empty cells
                if new_state == 0:
                    for n_row, n_col in self._get_neighbors(row, col):
                        if (n_row, n_col) not in processed_cells:
                            cells_to_check.append((n_row, n_col))
            
            return {
                'game_state': updated_state,
                'cell_data': cell_data,
                'remaining_bombs': remaining_bombs,
            }

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cell coordinates."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
                    neighbors.append((new_row, new_col))
        return neighbors