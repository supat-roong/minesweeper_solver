from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class ScreenRegion: 
    """
    Represents a rectangular region on the screen with its image data.
    
    Attributes:
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the region in pixels
        height: Height of the region in pixels
        image: Numpy array containing the image data (BGR format)
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
        if len(self.image.shape) != 3:
            raise ValueError("image must be a 3-dimensional array (height, width, channels)")
        if self.image.shape[2] != 3:
            raise ValueError("image must have 3 color channels (BGR)")
        
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")
        
        if self.image.shape[0] != self.height or self.image.shape[1] != self.width:
            raise ValueError(f"Image dimensions ({self.image.shape[1]}x{self.image.shape[0]}) "
                           f"do not match specified dimensions ({self.width}x{self.height})")


@dataclass
class CellPosition:
    """
    Represents a letter's position in both screen coordinates and grid coordinates.
    
    Attributes:
        cell_state: cell s at this position
        screen_x_range: X-coordinate on screen
        screen_y_range: Y-coordinate on 
        screen_x: center coordinate
        screen_y: int
        grid_row: Row index in the game grid
        grid_col: Column index in the game grid
    """
    letter: str
    screen_x_range: tuple[int, int]
    screen_y_range: tuple[int, int]
    screen_x: int
    screen_y: int
    grid_row: int
    grid_col: int

    def __post_init__(self):
        """Validate letter and position parameters."""
        if self.grid_row < 0:
            raise ValueError("grid_row must be non-negative")
        if self.grid_col < 0:
            raise ValueError("grid_col must be non-negative")

@dataclass
class CellData:
    """Store both state and position information for each cell"""
    state: int  # -1: unopened, -2: flag, 0-8: number
    position: CellPosition


@dataclass
class Move:
    row: int
    col: int
    action: str  # 'click', 'flag' or 'none'
    probability: float

    
@dataclass
class PlayerConfig:
    """
    Configuration for mouse movement and timing in automated gameplay.
    
    Attributes:
        move_speed: Speed of mouse movement in seconds
        pause_between_words: Pause duration between words in seconds
    """
    move_speed: float
    pause_between_words: float

    def __post_init__(self):
        """Validate player configuration parameters."""
        if self.move_speed <= 0:
            raise ValueError("move_speed must be positive")
        if self.pause_between_words < 0:
            raise ValueError("pause_between_words cannot be negative")