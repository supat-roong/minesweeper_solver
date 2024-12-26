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
        """
        Initialize the MinesweeperDetector with given grid size, OCR path, and debug options.
        
        Args:
            grid_size (Tuple[int, int]): Number of rows and columns in the Minesweeper grid.
            tesseract_path (Optional[str]): Path to the Tesseract OCR binary (if used for number detection).
            debug (bool): If True, enables debugging and visualization tools.
        """
        self.grid_size = grid_size
        self.debugger = Debugger() if debug else None
        
        # Initialize component detectors
        self.bomb_counter = BombCounterDetector(self.debugger)
        self.board_detector = BoardDetector(grid_size, self.debugger)
        self.grid_detector = GridLineDetector(grid_size, self.debugger)
        self.cell_detector = CellStateDetector(tesseract_path, self.debugger)
        
        self.cell_data = {}  # Stores data about each cell (state and position)
        self.screen_region = None  # Stores details about the current screen region being processed
        self.board_offset_x = 0  # Horizontal offset of the game board relative to the screen region
        self.board_offset_y = 0  # Vertical offset of the game board relative to the screen region

    def process_screenshot(self, screen_region: ScreenRegion) -> dict:
        """
        Process a screenshot of the Minesweeper game and extract game state information.

        Args:
            screen_region (ScreenRegion): Region of the screen containing the Minesweeper game.

        Returns:
            dict: Extracted game state, including total bombs, remaining bombs, cell data, and more.
        """
        self.screen_region = screen_region
        
        # Detect the total number of bombs from the bomb counter
        total_bombs = self.bomb_counter.detect(screen_region.image)
        print(f"Detected bombs: {total_bombs}")
        
        # Detect the board region, offsets, and reset button location
        (board, board_offset), reset_button = self.board_detector.detect(screen_region.image)
        self.board_offset_x, self.board_offset_y = board_offset
        print(f"Board offset relative to screen region: ({self.board_offset_x}, {self.board_offset_y})")
        
        # Process the board to determine the game state and cell details
        game_state, cell_data = self._process_board(board)
        
        # Visualize the game state if debugging is enabled
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
        """
        Process the Minesweeper board image to detect cell states and calculate their screen positions.

        Args:
            board (np.ndarray): Cropped image of the Minesweeper board.

        Returns:
            Tuple[np.ndarray, Dict[Tuple[int, int], CellData]]:
                - A 2D array representing the game state of each cell.
                - A dictionary containing details (state and position) for each cell.
        """
        # Detect grid lines that separate the cells
        horizontal_lines, vertical_lines = self.grid_detector.detect(board)
        
        # Initialize game state (grid of cell states) and cell data dictionary
        game_state = np.zeros(self.grid_size, dtype=int)
        cell_data = {}
        
        # Process each cell in the grid
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                # Determine cell boundaries based on grid lines
                top = horizontal_lines[i]
                bottom = horizontal_lines[i + 1]
                left = vertical_lines[j]
                right = vertical_lines[j + 1]
                
                # Add a margin to avoid boundary issues
                margin = 3
                cell = board[top + margin:bottom - margin, left + margin:right - margin]
                
                # Detect the state of the cell (e.g., empty, number, flagged)
                state = self.cell_detector.detect_state(cell, (i, j))
                game_state[i, j] = state
                
                # Calculate absolute screen coordinates for the cell
                abs_left = self.screen_region.x + self.board_offset_x + left + margin
                abs_right = self.screen_region.x + self.board_offset_x + right - margin
                abs_top = self.screen_region.y + self.board_offset_y + top + margin
                abs_bottom = self.screen_region.y + self.board_offset_y + bottom - margin
                abs_center_x = self.screen_region.x + self.board_offset_x + (left + right) // 2
                abs_center_y = self.screen_region.y + self.board_offset_y + (top + bottom) // 2
                
                # Create a CellPosition object to store the cell's position data
                cell_pos = CellPosition(
                    letter=str(state) if state >= 0 else ('-' if state == -1 else 'F'),
                    screen_x_range=(abs_left, abs_right),
                    screen_y_range=(abs_top, abs_bottom),
                    screen_x=abs_center_x,
                    screen_y=abs_center_y,
                    grid_row=i,
                    grid_col=j
                )
                
                # Store the cell's data
                cell_data[(i, j)] = CellData(state=state, position=cell_pos)
        
        return game_state, cell_data

    def update_after_move(self, screen_region: ScreenRegion, last_move: Move, game_state: np.ndarray, cell_data: Dict, remaining_bombs: int) -> dict:
        """
        Update the game state after a player makes a move.

        Args:
            screen_region (ScreenRegion): Updated screen region after the move.
            last_move (Move): Details of the last move made (e.g., flagging or clicking).
            game_state (np.ndarray): Current game state.
            cell_data (Dict): Current cell data (state and position).
            remaining_bombs (int): Number of bombs remaining to be flagged.

        Returns:
            dict: Updated game state, cell data, and remaining bomb count.
        """
        updated_state = game_state.copy()
        
        # Handle flagging a cell
        if last_move.action == 'flag':
            row, col = last_move.row, last_move.col
            updated_state[row, col] = -2  # Mark the cell as flagged
            remaining_bombs -= 1  # Decrement the bomb counter
            
            # Update cell data
            cell_data[(row, col)] = CellData(
                state=updated_state[row, col],
                position=cell_data[(row, col)].position
            )
            
            return {
                'game_state': updated_state,
                'cell_data': cell_data,
                'remaining_bombs': remaining_bombs,
            }
        
        # Handle clicking a cell
        elif last_move.action == 'click':
            cells_to_check = [(last_move.row, last_move.col)]  # Initialize the cell queue for BFS
            processed_cells = set()
            
            while cells_to_check:
                row, col = cells_to_check.pop(0)
                
                # Skip already processed or out-of-bounds cells
                if ((row, col) in processed_cells or 
                    row < 0 or row >= self.grid_size[0] or 
                    col < 0 or col >= self.grid_size[1]):
                    continue
                    
                processed_cells.add((row, col))
                cell_pos = cell_data[(row, col)].position
                
                # Extract and process the cell's image
                cell_img = screen_region.image[
                    cell_pos.screen_y_range[0] - screen_region.y:cell_pos.screen_y_range[1] - screen_region.y,
                    cell_pos.screen_x_range[0] - screen_region.x:cell_pos.screen_x_range[1] - screen_region.x
                ]
                
                # Detect the new state of the cell
                new_state = self.cell_detector.detect_state(cell_img, (row, col))
                updated_state[row, col] = new_state
                
                # Update the cell data with the new state
                cell_data[(row, col)] = CellData(
                    state=new_state,
                    position=cell_pos
                )
                
                # Add neighbors to the queue if the cell is empty
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
        """
        Get valid neighboring cells for a given cell in the grid.

        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            List[Tuple[int, int]]: List of valid neighboring cell coordinates.
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                n_row, n_col = row + dr, col + dc
                if 0 <= n_row < self.grid_size[0] and 0 <= n_col < self.grid_size[1]:
                    neighbors.append((n_row, n_col))
        return neighbors
