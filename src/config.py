from typing import Dict, List, Tuple
from custom_dataclass import CellState, VisualizerConfig, WindowConfig, DisplayColors, ProbabilityConfig

# Toggle for enabling/disabling the board visualizer.
BOARD_VISUALIZER = True

# Enable or disable debugging outputs for development and troubleshooting.
DEBUG = False

# Dimensions of the Minesweeper grid (e.g., 9x9 grid for a beginner level).
MINESWEEPER_GRID_SIZE = (9, 9)  # Adjust based on your Minesweeper variant

# Path to the Tesseract OCR executable, used for text recognition if needed.
TESSERACT_PATH = "path/to/tesseract.exe"  # Adjust path as needed

class Colors:
    """
    Stores color configurations for the Minesweeper board.
    Colors are defined as a dictionary where keys are descriptive strings,
    and values are tuples containing RGB ranges for the respective colors.
    """
    CELL_COLORS: Dict[str, Tuple[List[int], List[int]]] = {
        'white': ([240, 240, 240], [255, 255, 255]),  # Color range for unmarked cells
        'gray': ([180, 180, 180], [200, 200, 200]),   # Color range for opened cells
        'red': ([0, 0, 220], [40, 60, 255]),         # Color range for flagged cells
        'black': ([0, 0, 0], [10, 10, 10])           # Color range for undetected mines
    }
    
    NUMBER_COLORS: Dict[str, Tuple[List[int], List[int]]] = {
        # Colors for the numbers displayed on opened cells based on adjacent mines
        'blue': ([200, 0, 0], [255, 50, 50]),        # Number 1
        'green': ([0, 100, 20], [20, 200, 50]),      # Number 2
        'red': ([0, 0, 200], [50, 50, 255]),         # Number 3
        'dark_blue': ([100, 0, 0], [150, 20, 20]),   # Number 4
        'brown': ([0, 0, 100], [20, 20, 150]),       # Number 5
        'cyan': ([100, 100, 0], [150, 150, 50])      # Number 6
    }

# Definitions for the state of each cell on the Minesweeper board.
CELL_STATE = CellState(
    unopened=-1,    # Cell is not yet opened.
    flag=-2,        # Cell is flagged as a potential mine.
    mine=-3,        # Cell contains a mine.
    undetected=-4,  # Cell is undetected (used for debugging).
    empty=0         # Cell is opened and has no adjacent mines.
)

# Configuration for the visualizer that displays the Minesweeper board.
VISUALIZER_CONFIG = VisualizerConfig(
    window=WindowConfig(
        width=900,            # Width of the visualization window (in pixels).
        height=900,           # Height of the visualization window (in pixels).
        position_x=1000,      # X-coordinate of the window's top-left corner.
        position_y=50         # Y-coordinate of the window's top-left corner.
    ),
    colors=DisplayColors(
        # Define the display colors for different cell states.
        unopened='lightgray',  # Color for unopened cells.
        flag='lightcoral',     # Color for flagged cells.
        mine='red',            # Color for cells with mines.
        undetected='lightgray',# Color for undetected cells (debugging).
        opened='white'         # Color for opened cells.
    ),
    number_colors=[
        'blue',       # Number 1 (adjacent mines)
        'green',      # Number 2
        'red',        # Number 3
        'darkblue',   # Number 4
        'brown',      # Number 5
        'cyan'        # Number 6
    ],
    probability=ProbabilityConfig(
        # Probability values used for visualizing mine probabilities on the board.
        min_value=0,       # Minimum probability value.
        max_value=1,       # Maximum probability value.
        colors=[
            (0.8, 1.0, 0.8),  # Green (low probability of mine)
            (1.0, 1.0, 0.8),  # Yellow (medium probability)
            (1.0, 0.8, 0.8)   # Red (high probability of mine)
        ]
    ),
    update_interval_ms=50  # Interval (in milliseconds) for updating the visualizer.
)
