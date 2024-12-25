import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import threading
import time
import queue
from typing import Tuple
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MinesweeperVisualizer:
    def __init__(self, grid_size: Tuple[int, int]):
        self.running = True
        self.after_id = None
        self.rows = grid_size[0]
        self.cols = grid_size[1]
        
        # Initialize empty state
        self.game_state = np.full(grid_size, -1)
        self.mine_probs = np.zeros(grid_size)
        self.update_queue = queue.Queue()
        
        # Create custom colormap for probabilities
        colors = [(0.8, 1, 0.8), (1, 1, 0.8), (1, 0.8, 0.8)]
        self.prob_cmap = LinearSegmentedColormap.from_list('custom_prob', colors)
        
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Minesweeper Visualizer")
        

        
        # Create main figure
        self.fig = Figure(figsize=(6, 6))
        
        # Create subfigures with 80/20 split
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[8, 2], height_ratios=[1, 1])
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Game state
        self.ax_legend = self.fig.add_subplot(gs[0, 1])  # Legend
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Probabilities
        self.ax_prob_legend = self.fig.add_subplot(gs[1, 1])  # Probabilities Legend
        
        # Adjust spacing
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=-0.2, hspace=0.2)
        
        # Create canvas and embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        
        # Store all visual elements
        self.cell_patches = []
        self.prob_patches = []
        self.text_objects = []
        self.prob_text_objects = []
        
        # Create all visual elements
        self._setup_static_elements()
        self._create_legends()
        
        # Start periodic check for updates
        self._check_updates()
        
    def _create_legend_cell(self, ax, pos, facecolor, text, text_color='black', text_size=12):
        """Helper function to create a legend cell with exact game appearance"""
        rect = patches.Rectangle(
            pos, 1, 0.5,  # Make cells slightly shorter for compact layout
            facecolor=facecolor,
            edgecolor='gray'
        )
        ax.add_patch(rect)
        
        if text:
            if text == "*":
                t = ax.text(
                    pos[0]+0.5, pos[1] + 0.1,  # Center text vertically
                    text,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=text_size,
                    fontweight='bold'
                )
            else:
                t = ax.text(
                    pos[0]+0.5, pos[1] + 0.2,  # Center text vertically
                    text,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=text_size,
                    fontweight='bold'
                )
            return [rect, t]
        return [rect]
        
    def _create_legends(self):
        # Clear and set up legend axes
        self.ax_legend.clear()
        self.ax_legend.set_title('Legend')
        self.ax_legend.set_xticks([])
        self.ax_legend.set_yticks([])
        
        # Create each legend cell
        legend_items = [
            ('Unopened', 'lightgray', '', 'black', 12),
            ('Opened', 'white', '', 'black', 12),
            ('Flag', 'lightcoral', 'F', 'darkred', 12),
            ('Mine', 'red', '*', 'black', 24),
            ('Undetected', 'lightgray', 'X', 'black', 12)
        ]
        
        # Calculate legend layout
        cell_height = 1.0  # More compact spacing
        total_height = len(legend_items) * cell_height
        start_y = total_height - 0.5  # Start from top
        
        # Set legend axes limits - even more compact horizontally
        self.ax_legend.set_xlim(-1.5, 3)
        self.ax_legend.set_ylim(0.2, total_height + 0.5)
        self.ax_legend.set_axis_off()
        
        for i, (label, color, text, text_color, text_size) in enumerate(legend_items):
            y_pos = start_y - (i * cell_height)
            artists = self._create_legend_cell(
                self.ax_legend, 
                (-1.2, y_pos),
                color,
                text,
                text_color,
                text_size
            )
            
            # Add label to the right of the cell - tighter spacing
            self.ax_legend.text(
                0.2,
                y_pos + 0.2,  # Align with cell center
                label,
                va='center',
                ha='left',
                fontsize=10
            )

        self.ax_prob_legend.set_axis_off()

        divider = make_axes_locatable(self.ax_prob_legend)
        cax = divider.append_axes("left", size="40%", pad=0.05, axes_class=plt.Axes)

        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=self.prob_cmap),
            cax=cax,
            format='%.2f',
            fraction=0.05
        )
        colorbar.ax.tick_params(labelsize=10)
        
    def _setup_static_elements(self):
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
                y_pos = self.rows - i - 1
                
                # Game state patches
                patch = patches.Rectangle(
                    (j - 0.5, y_pos - 0.5), 1, 1,
                    facecolor='lightgray',
                    edgecolor='gray'
                )
                self.ax1.add_patch(patch)
                self.cell_patches.append(patch)
                
                # Game state text
                text = self.ax1.text(
                    j, y_pos,
                    '',
                    ha='center',
                    va='center',
                    fontweight='bold'
                )
                self.text_objects.append(text)
                
                # Probability patches
                prob_patch = patches.Rectangle(
                    (j - 0.5, y_pos - 0.5), 1, 1,
                    facecolor='lightgreen',
                    edgecolor='gray'
                )
                self.ax2.add_patch(prob_patch)
                self.prob_patches.append(prob_patch)
                
                # Probability text
                prob_text = self.ax2.text(
                    j, y_pos - 0.35,
                    '',
                    ha='center',
                    va='baseline',
                    fontsize=8
                )
                self.prob_text_objects.append(prob_text)
    
    def _get_cell_color(self, value):
        if value == -1:  # Unopened
            return 'lightgray'
        elif value == -2:  # Flag
            return 'lightcoral'
        elif value == -3:  # Clicked bomb
            return 'red'
        elif value == -4:  # Detection failed
            return 'lightgray'
        return 'white'  # Opened cell
    
    def _get_text_color(self, number):
        colors = ['blue', 'green', 'red', 'darkblue', 'brown', 'cyan']
        if 1 <= number <= 6:
            return colors[number - 1]
        return 'black'
    
    def _check_updates(self):
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
                self.after_id = self.root.after(50, self._check_updates)

    def cleanup(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        try:
            # # Center window on screen with specific size
            window_width = 900
            window_height = 900
            x = 1000
            y = 50
            
            # # Set initial window position and size
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            self.root.mainloop()
        except:
            self.cleanup()
    
    def _update_display(self):
        """Update the display with current state"""
        for idx, ((i, j), patch, text) in enumerate(zip(
            np.ndindex(self.rows, self.cols),
            self.cell_patches,
            self.text_objects
        )):
            value = self.game_state[i, j]
            y_pos = self.rows - i - 1
            patch.set_facecolor(self._get_cell_color(value))
            
            if value > 0:  # Number
                text.set_text(str(value))
                text.set_color(self._get_text_color(value))
                text.set_fontsize(12)
                text.set_y(y_pos - 0.35)
                text.set_va('baseline')
            elif value == -2:  # Flag
                text.set_text('F')
                text.set_color('darkred')
                text.set_fontsize(12)
                text.set_y(y_pos - 0.35)
                text.set_va('baseline')
            elif value == -3:  # Bomb
                text.set_text('*')
                text.set_color('black')
                text.set_fontsize(20)
                text.set_y(y_pos - 0.2)
                text.set_va('center')
            elif value == -4:  # Failed
                text.set_text('X')
                text.set_color('black')
                text.set_fontsize(12)
                text.set_y(y_pos - 0.35)
                text.set_va('baseline')
            elif value == 0:  # Opened empty cell
                text.set_text('0')
                text.set_color('black')
                text.set_fontsize(12)
                text.set_y(y_pos - 0.35)
                text.set_va('baseline')
            else:
                text.set_text('')
        
        # Update probabilities using colormap
        for idx, ((i, j), patch, text) in enumerate(zip(
            np.ndindex(self.rows, self.cols),
            self.prob_patches,
            self.prob_text_objects
        )):
            y_pos = self.rows - i - 1
            prob = self.mine_probs[i, j]
            patch.set_facecolor(self.prob_cmap(prob))
            text.set_y(y_pos - 0.2)
            text.set_text(str(int(prob)) if prob in [0, 1] else f'{prob:.2f}')
        
        self.canvas.draw()
    
    def update(self, game_state, mine_probs):
        """Thread-safe update of the visualization"""
        if game_state.shape != (self.rows, self.cols) or mine_probs.shape != (self.rows, self.cols):
            raise ValueError(f"Arrays must have shape ({self.rows}, {self.cols})")
        
        # Put update in queue
        self.update_queue.put((game_state.copy(), mine_probs.copy()))
