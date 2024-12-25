from typing import Dict, List, Tuple
from custom_dataclass import CellState, VisualizerConfig, WindowConfig, DisplayColors, ProbabilityConfig

BOARD_VISUALIZER = True

DEBUG = False

MINESWEEPER_GRID_SIZE = (9, 9)  # Adjust based on your Minesweeper variant

TESSERACT_PATH = "path/to/tesseract.exe"  # Adjust path as needed

class Colors:
    CELL_COLORS: Dict[str, Tuple[List[int], List[int]]] = {
        'white': ([240, 240, 240], [255, 255, 255]),
        'gray': ([180, 180, 180], [200, 200, 200]),
        'red': ([0, 0, 220], [40, 60, 255]),
        'black': ([0, 0, 0], [10, 10, 10])
    }
    
    NUMBER_COLORS: Dict[str, Tuple[List[int], List[int]]] = {
        'blue': ([200, 0, 0], [255, 50, 50]),      # 1
        'green': ([0, 100, 20], [20, 200, 50]),    # 2
        'red': ([0, 0, 200], [50, 50, 255]),       # 3
        'dark_blue': ([100, 0, 0], [150, 20, 20]), # 4
        'brown': ([0, 0, 100], [20, 20, 150]),     # 5
        'cyan': ([100, 100, 0], [150, 150, 50]),   # 6
    }

CELL_STATE = CellState(
    unopened=-1,
    flag=-2,
    mine=-3,
    undetected=-4,
    empty=0
)

VISUALIZER_CONFIG = VisualizerConfig(
    window=WindowConfig(
        width=900,
        height=900,
        position_x=1000,
        position_y=50
    ),
    colors=DisplayColors(
        unopened='lightgray',
        flag='lightcoral',
        mine='red',
        undetected='lightgray',
        opened='white'
    ),
    number_colors=[
        'blue',
        'green', 
        'red',
        'darkblue',
        'brown',
        'cyan'
    ],
    probability=ProbabilityConfig(
        min_value=0,
        max_value=1,
        colors=[
            (0.8, 1.0, 0.8),  # Green
            (1.0, 1.0, 0.8),  # Yellow
            (1.0, 0.8, 0.8)   # Red
        ]
    ),
    update_interval_ms=50
)