import sys
import os
from pathlib import Path

# Add src directory to the Python module search path
SRC_DIR = os.path.join(str(Path(__file__).resolve().parent), "src")
sys.path.append(SRC_DIR)

import logging
import threading
from config import MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG, BOARD_VISUALIZER
from minesweeper_detector import MinesweeperDetector
from player import MinesweeperPlayer
from visualizer import MinesweeperVisualizer

def main():
    """
    Main entry point for the Minesweeper solver.
    
    This function configures logging, initializes the Minesweeper components, 
    and either runs the solver with or without a visualizer.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize the Minesweeper detector and player
    detector = MinesweeperDetector(MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG)
    player = MinesweeperPlayer(logger)
    
    if BOARD_VISUALIZER:
        # Use the visualizer in a separate thread
        visualizer = MinesweeperVisualizer(MINESWEEPER_GRID_SIZE)
        running = True

        # Start a thread to update the game while keeping the visualizer responsive
        update_thread = threading.Thread(
            target=player.play_game,
            args=(detector, visualizer, running),
            daemon=True  # Thread will stop when the main program exits
        )
        update_thread.start()

        # Run the GUI in the main thread
        try:
            visualizer.run()
        finally:
            # Ensure the update thread stops when the visualizer is closed
            running = False
    else:
        # Run the solver in the main thread without a visualizer
        player.play_game(detector)

if __name__ == "__main__":
    main()
