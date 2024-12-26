from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Debugger:
    """
    A visualization tool for debugging image processing pipelines.
    
    This class provides functionality to display intermediate results of image
    processing operations, allowing step-by-step visualization with interactive
    control. Each visualization is numbered sequentially and requires user
    interaction to proceed to the next step.
    
    Attributes:
        current_step (int): Counter keeping track of the current debugging step number.
    """
    
    def __init__(self) -> None:
        """Initialize the Debugger with a step counter starting at 0."""
        self.current_step: int = 0
    
    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """
        Display a debug image and wait for user interaction before proceeding.
        
        This method creates a new figure window showing the provided image with
        a numbered step title. If the input image is in BGR format (common for
        OpenCV images), it automatically converts it to RGB for proper display.
        The window includes a grid and axes for better visualization of pixel
        coordinates.
        
        Args:
            title (str): Descriptive title for the current debugging step.
            img (np.ndarray): Image array to display. Can be either grayscale
                (2D array) or color (3D array in BGR format).
            cmap (Optional[str]): Colormap to use for displaying the image.
                Particularly useful for grayscale images or heat maps.
                Defaults to None, which uses matplotlib's default colormap.
        
        Notes:
            - The method will increment the internal step counter automatically.
            - The window will display until any keyboard button is pressed.
            - The figure window is automatically closed after the button press.
            - For BGR images (OpenCV default), automatic conversion to RGB is performed.
        """
        self.current_step += 1
        
        # Create new figure with specified size
        plt.figure(figsize=(10, 8))
        plt.title(f"Step {self.current_step}: {title}")
        
        # Convert BGR to RGB if dealing with a color image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display image with specified parameters
        plt.imshow(img, cmap=cmap)
        plt.axis('on')
        plt.grid(True)
        
        # Add instruction text at the bottom
        plt.figtext(0.5, 0.01, 'Press any key to continue...',
                   ha='center', va='bottom')
        
        # Show the plot and wait for user input
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

        
    def visualize_game_state(self, game_state: np.ndarray) -> None:
        """Visualize the game state matrix."""
        plt.figure(figsize=(10, 10))
        plt.imshow(game_state, cmap='viridis')
        plt.colorbar(label='Cell State')
        plt.title('Game State Matrix\nPress any key to continue...')
        
        grid_size = game_state.shape
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                plt.text(j, i, str(game_state[i, j]), 
                        ha='center', va='center', color='black')
        
        plt.grid(True)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
        
    def visualize_cell_analysis(self, cell: np.ndarray, position: tuple, state: int) -> None:
        """Show detailed cell analysis visualization."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original")
        
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        ax2.imshow(gray, cmap='gray')
        ax2.set_title("Grayscale")
        
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ax3.imshow(thresh, cmap='gray')
        ax3.set_title("Threshold")
        
        plt.suptitle(f"Cell Analysis - Position: {position}, State: {state}")
        plt.figtext(0.5, 0.01, 'Press any key to continue...', ha='center', va='bottom')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()