import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
import queue
from typing import Tuple, List, Optional
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import CELL_STATE, VISUALIZER_CONFIG

class MinesweeperVisualizer:
    """
    A real-time visualization tool for Minesweeper game state and probabilities.
    
    This class creates a Tkinter window with two main displays:
    1. Game State: Shows the current state of the Minesweeper board
    2. Mine Probabilities: Displays probability heatmap of mine locations
    """
    
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize the visualizer with specified grid dimensions.
        
        Args:
            grid_size: Tuple of (rows, columns) defining the grid size
        """
        self.running = True
        self.after_id: Optional[str] = None
        self.rows, self.cols = grid_size
        
        # Initialize game state arrays
        self.game_state = np.full(grid_size, CELL_STATE.unopened)
        self.mine_probs = np.zeros(grid_size)
        self.update_queue: queue.Queue = queue.Queue()
        
        # Setup visualization components
        self._setup_colormap()
        self._create_window()
        self._setup_figure()
        self._initialize_visual_elements()
        self._setup_static_elements()
        self._create_legends()
        
        # Start update loop
        self._check_updates()

    def _setup_colormap(self):
        """Create custom colormap for probability visualization with green at 0 and red at 1."""
        # Define colors for the colormap: green (0) to yellow (0.5) to red (1)
        colors = [
            (0.8, 1.0, 0.8),  # Green
            (1.0, 1.0, 0.8),  # Yellow
            (1.0, 0.8, 0.8)   # Red
        ]
        self.prob_norm = plt.Normalize(
            VISUALIZER_CONFIG.probability.min_value,
            VISUALIZER_CONFIG.probability.max_value
        )
        self.prob_cmap = LinearSegmentedColormap.from_list('custom_prob', colors)
    
    def _create_window(self):
        """Initialize the Tkinter window."""
        self.root = tk.Tk()
        self.root.title("Minesweeper Visualizer")
    
    def _setup_figure(self):
        """Setup the matplotlib figure and its subplots."""
        self.fig = Figure(figsize=(6, 6))
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[8, 2], height_ratios=[1, 1])
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Game state
        self.ax_legend = self.fig.add_subplot(gs[0, 1])  # Legend
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Probabilities
        self.ax_prob_legend = self.fig.add_subplot(gs[1, 1])  # Probabilities Legend
        
        # Adjust layout
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, 
                               wspace=-0.2, hspace=0.2)
        
        # Create and pack canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
    
    def _initialize_visual_elements(self):
        """Initialize lists to store visual elements."""
        self.cell_patches: List[patches.Rectangle] = []
        self.prob_patches: List[patches.Rectangle] = []
        self.text_objects: List[plt.Text] = []
        self.prob_text_objects: List[plt.Text] = []
    
    def _create_legend_cell(self, ax, pos: Tuple[float, float], facecolor: str, 
                          text: str, text_color: str = 'black', text_size: int = 12) -> List:
        """Create a single cell in the legend."""
        rect = patches.Rectangle(
            pos, 1, 0.5,
            facecolor=facecolor,
            edgecolor='gray'
        )
        ax.add_patch(rect)
        
        if not text:
            return [rect]
        
        y_offset = 0.1 if text == "*" else 0.2
        t = ax.text(
            pos[0] + 0.5, pos[1] + y_offset,
            text,
            ha='center',
            va='center',
            color=text_color,
            fontsize=text_size,
            fontweight='bold'
        )
        return [rect, t]
    
    def _create_legends(self):
        """Create legends for both game state and probability displays."""
        self.ax_legend.clear()
        self.ax_legend.set_title('Legend')
        self.ax_legend.set_xticks([])
        self.ax_legend.set_yticks([])
        
        # Define legend items using config colors
        legend_items = [
            ('Unopened', VISUALIZER_CONFIG.colors.unopened, '', 'black', 12),
            ('Opened', VISUALIZER_CONFIG.colors.opened, '', 'black', 12),
            ('Flag', VISUALIZER_CONFIG.colors.flag, 'F', 'darkred', 12),
            ('Mine', VISUALIZER_CONFIG.colors.mine, '*', 'black', 24),
            ('Undetected', VISUALIZER_CONFIG.colors.unopened, 'X', 'black', 12)
        ]
        
        # Calculate layout
        cell_height = 1.0
        total_height = len(legend_items) * cell_height
        start_y = total_height - 0.5
        
        # Set axes limits
        self.ax_legend.set_xlim(-1.5, 3)
        self.ax_legend.set_ylim(0.2, total_height + 0.5)
        self.ax_legend.set_axis_off()
        
        # Create legend cells
        for i, (label, color, text, text_color, text_size) in enumerate(legend_items):
            y_pos = start_y - (i * cell_height)
            self._create_legend_cell(
                self.ax_legend,
                (-1.2, y_pos),
                color,
                text,
                text_color,
                text_size
            )
            self.ax_legend.text(
                0.2,
                y_pos + 0.2,
                label,
                va='center',
                ha='left',
                fontsize=10
            )
        
        self._create_probability_legend()
    
    def _create_probability_legend(self):
        """Create colorbar legend for probability display."""
        self.ax_prob_legend.set_axis_off()
        divider = make_axes_locatable(self.ax_prob_legend)
        cax = divider.append_axes("left", size="40%", pad=0.05, axes_class=plt.Axes)
        
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=self.prob_norm, cmap=self.prob_cmap),
            cax=cax,
            format='%.2f',
            fraction=0.05
        )
        colorbar.ax.tick_params(labelsize=10)
    
    def _create_cell(self, i: int, j: int):
        """Create visual elements for a single cell."""
        y_pos = self.rows - i - 1
        
        # Game state elements
        patch = patches.Rectangle(
            (j - 0.5, y_pos - 0.5), 1, 1,
            facecolor=VISUALIZER_CONFIG.colors.unopened,
            edgecolor='gray'
        )
        self.ax1.add_patch(patch)
        self.cell_patches.append(patch)
        
        text = self.ax1.text(
            j, y_pos,
            '',
            ha='center',
            va='center',
            fontweight='bold'
        )
        self.text_objects.append(text)
        
        # Probability elements
        prob_patch = patches.Rectangle(
            (j - 0.5, y_pos - 0.5), 1, 1,
            facecolor='lightgreen',
            edgecolor='gray'
        )
        self.ax2.add_patch(prob_patch)
        self.prob_patches.append(prob_patch)
        
        prob_text = self.ax2.text(
            j, y_pos - 0.35,
            '',
            ha='center',
            va='baseline',
            fontsize=8
        )
        self.prob_text_objects.append(prob_text)
    
    def _get_cell_color(self, value: int) -> str:
        """Get the display color for a cell based on its value."""
        colors = {
            CELL_STATE.unopened: VISUALIZER_CONFIG.colors.unopened,
            CELL_STATE.flag: VISUALIZER_CONFIG.colors.flag,
            CELL_STATE.mine: VISUALIZER_CONFIG.colors.mine,
            CELL_STATE.undetected: VISUALIZER_CONFIG.colors.undetected
        }
        return colors.get(value, VISUALIZER_CONFIG.colors.opened)
    
    def _get_text_color(self, number: int) -> str:
        """Get the text color for a numbered cell."""
        if 1 <= number <= len(VISUALIZER_CONFIG.number_colors):
            return VISUALIZER_CONFIG.number_colors[number - 1]
        return 'black'

    def _setup_static_elements(self):
        """Setup the static elements of both game state and probability displays."""
        self.ax1.set_title('Game State')
        self.ax2.set_title('Mine Probabilities')
        
        for ax in [self.ax1, self.ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-0.5, self.rows - 0.5)
            ax.set_aspect('equal')
        
        for i in range(self.rows):
            for j in range(self.cols):
                self._create_cell(i, j)

    def _check_updates(self):
        """Check for and process any pending updates in the queue."""
        if not self.running:
            return
        try:
            while True:
                game_state, mine_probs = self.update_queue.get_nowait()
                while not self.update_queue.empty():
                    game_state, mine_probs = self.update_queue.get_nowait()
                self.game_state = game_state
                self.mine_probs = mine_probs
                self._update_display()
        except queue.Empty:
            pass
        finally:
            if self.running:
                self.after_id = self.root.after(
                    VISUALIZER_CONFIG.update_interval_ms,
                    self._check_updates
                )

    def _set_text_properties(self, text_obj, content: str, color: str, 
                           size: int, y_pos: float, va: str):
        """
        Set all properties of a text object at once.
        
        Args:
            text_obj: The matplotlib text object to update
            content: The text content to display
            color: The color of the text
            size: Font size
            y_pos: Vertical position
            va: Vertical alignment
        """
        text_obj.set_text(content)
        text_obj.set_color(color)
        text_obj.set_fontsize(size)
        text_obj.set_y(y_pos)
        text_obj.set_va(va)

    def _update_display(self):
        """Update both game state and probability displays."""
        self._update_game_state()
        self._update_probabilities()
        self.canvas.draw()
    
    def _update_game_state(self):
        """Update the game state display."""
        for idx, ((i, j), patch, text) in enumerate(zip(
            np.ndindex(self.rows, self.cols),
            self.cell_patches,
            self.text_objects
        )):
            value = self.game_state[i, j]
            y_pos = self.rows - i - 1
            
            # Update cell color
            patch.set_facecolor(self._get_cell_color(value))
            
            # Update text properties based on cell value
            if value > 0:  # Number
                self._set_text_properties(
                    text, str(value),
                    self._get_text_color(value),
                    12, y_pos - 0.35, 'baseline'
                )
            elif value == CELL_STATE.flag:
                self._set_text_properties(
                    text, 'F', 'darkred',
                    12, y_pos - 0.35, 'baseline'
                )
            elif value == CELL_STATE.mine:
                self._set_text_properties(
                    text, '*', 'black',
                    20, y_pos - 0.2, 'center'
                )
            elif value == CELL_STATE.undetected:
                self._set_text_properties(
                    text, 'X', 'black',
                    12, y_pos - 0.35, 'baseline'
                )
            elif value == CELL_STATE.empty:  # Empty opened cell
                self._set_text_properties(
                    text, '0', 'black',
                    12, y_pos - 0.35, 'baseline'
                )
            else:
                text.set_text('')
    
    def _update_probabilities(self):
        """Update the probability display."""
        for idx, ((i, j), patch, text) in enumerate(zip(
            np.ndindex(self.rows, self.cols),
            self.prob_patches,
            self.prob_text_objects
        )):
            y_pos = self.rows - i - 1
            prob = self.mine_probs[i, j]
            
            # Update probability cell color using normalization
            patch.set_facecolor(self.prob_cmap(self.prob_norm(prob)))
            
            # Update probability text
            text.set_y(y_pos - 0.2)
            if prob in [0, 1]:
                text.set_text(str(int(prob)))
            else:
                text.set_text(f'{prob:.2f}')
    
    def update(self, game_state: np.ndarray, mine_probs: np.ndarray):
        """
        Thread-safe update of the visualization.
        
        Args:
            game_state: Current state of the game board
            mine_probs: Current probability of mines for each cell
            
        Raises:
            ValueError: If input arrays don't match the grid size
        """
        if (game_state.shape != (self.rows, self.cols) or 
            mine_probs.shape != (self.rows, self.cols)):
            raise ValueError(
                f"Arrays must have shape ({self.rows}, {self.cols})"
            )
        
        # Put update in queue
        self.update_queue.put((game_state.copy(), mine_probs.copy()))
    
    def cleanup(self):
        """Clean up resources and close the window."""
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the visualization with configured window settings."""
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        try:
            # Set window position and size from config
            self.root.geometry(
                f"{VISUALIZER_CONFIG.window.width}x"
                f"{VISUALIZER_CONFIG.window.height}+"
                f"{VISUALIZER_CONFIG.window.position_x}+"
                f"{VISUALIZER_CONFIG.window.position_y}"
            )
            self.root.mainloop()
        except:
            self.cleanup()