from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class CellInfo:
    """
    Stores information about a cell's neighbors and mine count.

    Attributes:
        unknown_neighbors: List of coordinates of unopened neighboring cells.
        flagged_neighbors: List of coordinates of flagged neighboring cells.
        remaining_mines: Number of mines remaining in the neighboring cells.
    """
    unknown_neighbors: List[Tuple[int, int]]
    flagged_neighbors: List[Tuple[int, int]]
    remaining_mines: int

@dataclass
class ConstraintGroup:
    """
    Represents constraint information for a group of cells in Minesweeper.

    Attributes:
        cells: List of cell coordinates involved in the constraint.
        mines: Number of mines within the given cells.
    """
    cells: List[Tuple[int, int]]
    mines: int

    def __post_init__(self):
        if self.mines < 0:
            raise ValueError("mines must be non-negative")

@dataclass
class CellState:
    """
    Constants representing the state of cells on the Minesweeper board.

    Attributes:
        unopened: State representing an unopened cell.
        flag: State representing a flagged cell.
        mine: State representing a mined cell.
        undetected: State for cells with undetected mines.
        empty: State representing an empty (opened) cell.
    """
    unopened: int
    flag: int
    mine: int
    undetected: int
    empty: int

@dataclass
class DisplayColors:
    """
    Defines color settings for cell visualization.

    Attributes:
        unopened: Color for unopened cells.
        flag: Color for flagged cells.
        mine: Color for cells with mines.
        undetected: Color for undetected mines.
        opened: Color for opened cells.
    """
    unopened: str
    flag: str
    mine: str
    undetected: str
    opened: str

    def __post_init__(self):
        if not all(isinstance(color, str) for color in [self.unopened, self.flag, self.mine, self.undetected, self.opened]):
            raise ValueError("All colors must be strings")

@dataclass
class WindowConfig:
    """
    Configuration for the Minesweeper visualizer window.

    Attributes:
        width: Width of the window in pixels.
        height: Height of the window in pixels.
        position_x: X-coordinate of the window's top-left corner.
        position_y: Y-coordinate of the window's top-left corner.
    """
    width: int
    height: int
    position_x: int
    position_y: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

@dataclass
class ProbabilityConfig:
    """
    Settings for probability visualization in Minesweeper.

    Attributes:
        min_value: Minimum probability value.
        max_value: Maximum probability value.
        colors: List of RGB tuples for gradient visualization.
    """
    min_value: float
    max_value: float
    colors: List[Tuple[float, float, float]]

    def __post_init__(self):
        if self.min_value < 0 or self.max_value < 0:
            raise ValueError("min_value and max_value must be non-negative")

@dataclass
class VisualizerConfig:
    """
    Complete configuration for the Minesweeper visualizer.

    Attributes:
        window: Configuration for the visualizer window.
        colors: Color settings for different cell states.
        number_colors: Colors corresponding to cell numbers (1-8).
        probability: Configuration for probability visualization.
        update_interval_ms: Interval for visual updates, in milliseconds.
    """
    window: WindowConfig
    colors: DisplayColors
    number_colors: List[str]
    probability: ProbabilityConfig
    update_interval_ms: int

    def __post_init__(self):
        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")

@dataclass
class ScreenRegion:
    """
    Represents a rectangular region on the screen with associated image data.

    Attributes:
        x: X-coordinate of the top-left corner.
        y: Y-coordinate of the top-left corner.
        width: Width of the region in pixels.
        height: Height of the region in pixels.
        image: Numpy array containing the image data (BGR format).
    """
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray

    def __post_init__(self):
        """Validate region parameters and image data."""
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if len(self.image.shape) != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be a 3-dimensional array with 3 color channels (BGR)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.image.shape[0] != self.height or self.image.shape[1] != self.width:
            raise ValueError(
                f"Image dimensions ({self.image.shape[1]}x{self.image.shape[0]}) do not match specified dimensions ({self.width}x{self.height})"
            )

@dataclass
class CellPosition:
    """
    Represents a cell's position in screen coordinates and grid coordinates.

    Attributes:
        letter: Letter associated with the cell.
        screen_x_range: Tuple representing the X-coordinate range on the screen.
        screen_y_range: Tuple representing the Y-coordinate range on the screen.
        screen_x: Center X-coordinate of the cell on the screen.
        screen_y: Center Y-coordinate of the cell on the screen.
        grid_row: Row index of the cell in the grid.
        grid_col: Column index of the cell in the grid.
    """
    letter: str
    screen_x_range: Tuple[int, int]
    screen_y_range: Tuple[int, int]
    screen_x: int
    screen_y: int
    grid_row: int
    grid_col: int

    def __post_init__(self):
        """Validate position parameters."""
        if not isinstance(self.letter, str) or len(self.letter) != 1:
            raise ValueError("letter must be a single character string")
        if self.grid_row < 0:
            raise ValueError("grid_row must be non-negative")
        if self.grid_col < 0:
            raise ValueError("grid_col must be non-negative")

@dataclass
class CellData:
    """
    Stores both state and position information for each cell.

    Attributes:
        state: State of the cell (-1 for unopened, -2 for flag, 0-8 for number of adjacent mines).
        position: Position information of the cell.
    """
    state: int
    position: CellPosition

@dataclass
class Move:
    """
    Represents a player's move in Minesweeper.

    Attributes:
        row: Row index of the move.
        col: Column index of the move.
        action: Action to be performed ('click', 'flag', or 'none').
        probability: Probability associated with the move.
    """
    row: int
    col: int
    action: str
    probability: float


@dataclass
class GameOutcome:
    """
    Represents the outcome of a Minesweeper game.

    Attributes:
        is_over: Boolean indicating whether the game is over.
        result: Outcome of the game ('win', 'loss', or empty string if ongoing).
    """
    is_over: bool
    result: str

    def __post_init__(self):
        if self.result not in ('', 'win', 'loss'):
            raise ValueError("result must be '', 'win', or 'loss'")
