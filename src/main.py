import logging
import threading
from config import MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG, BOARD_VISUALIZER
from minesweeper_detector import MinesweeperDetector
from player import MinesweeperPlayer
from visualizer import MinesweeperVisualizer


def main():
    """Main entry point for Minesweeper bot."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    detector = MinesweeperDetector(MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG)
    player = MinesweeperPlayer(logger)
    
    if BOARD_VISUALIZER:
        # Run with visualizer in separate thread
        visualizer = MinesweeperVisualizer(MINESWEEPER_GRID_SIZE)
        running = True
        
        # Start update thread
        update_thread = threading.Thread(
            target=player.play_game, 
            args=(detector, visualizer, running), 
            daemon=True
        )
        update_thread.start()

        # Run GUI in main thread
        try:
            visualizer.run()
        finally:
            running = False
    else:
        # Run without visualizer in main thread
        player.play_game(detector)

if __name__ == "__main__":
    main()