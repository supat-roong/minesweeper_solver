import logging
import numpy as np
import time
import pyautogui
import cv2
import threading
from config import MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG
from screenshot import capture_screen_region
from board_detector import MinesweeperDetector
from solver import MinesweeperSolver
from player import MinesweeperPlayer
from visualizer import MinesweeperVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_game_over(game_state: np.ndarray, remaining_bombs: int) -> tuple[bool, str]:
    """Check if the game is over."""
    # Count unopened and flagged cells
    unopened_count = np.sum(game_state == -1)
    
    # Win condition: all non-bomb cells are opened
    if unopened_count == 0 and remaining_bombs == 0:
        return True, "win"
        
    # Look for revealed bomb (indicated by a special value, e.g., -3)
    if np.any(game_state == -3):  # Assuming -3 indicates revealed bomb
        return True, "loss"
    
    return False, ""

def play_game(detector, player, visualizer, running):
    try:
        logger.info("Starting Minesweeper bot...")
        print("Select the Minesweeper game window...")
        # Capture the game area
        screen_region = capture_screen_region()
        if screen_region is None:
            logger.error("Failed to capture screen region")
            return
        while running:
            # Initialize components
            logger.info(f"\nStarting game")
            
            # Initial board detection
            result = detector.process_screenshot(screen_region)
            game_state = np.array(result['game_state'])
            total_bombs = result['total_bombs']
            remaining_bombs = result['remaining_bombs']
            cell_data = result['cell_data']
            reset_button = result['reset_button']
            
            logger.info(f"Total bombs: {total_bombs}")
            logger.info(f"Remaining bombs: {remaining_bombs}")
            logger.info("\nInitial game state:")
            print(game_state)
            
            last_move = None

            # Calculate next move
            solver = MinesweeperSolver(game_state, cell_data, total_bombs)
            move = solver.get_best_move()

            # Visualizer
            visualizer.update(game_state, solver.mine_probabilities)
            
            if not move:
                logger.error("No valid moves found")
                break
            
            logger.info(f"\nMaking move:")
            logger.info(f"Position: ({move.row}, {move.col})")
            logger.info(f"Action: {move.action}")
            logger.info(f"Mine probability: {move.probability:.2%}")
            
            # Execute move
            screen_x, screen_y = solver.get_screen_coords(move)
            player.execute_move(screen_x, screen_y, move.action)
            last_move = move

            # Wait for animation
            time.sleep(0.5)
            
            # Game playing loop
            while running:                
                # Take new screenshot
                new_screen = pyautogui.screenshot(region=(
                    screen_region.x,
                    screen_region.y,
                    screen_region.width,
                    screen_region.height
                ))

                screen_region.image = cv2.cvtColor(np.array(new_screen), cv2.COLOR_RGB2BGR)
                
                # Update game state (only analyze affected cells)
                if last_move:
                    result = detector.update_after_move(
                        screen_region,
                        last_move,
                        game_state,
                        cell_data,
                        remaining_bombs
                    )
                else:
                    # Fallback to full board detection if needed
                    result = detector.process_screenshot(screen_region)
                
                game_state = np.array(result['game_state'])
                cell_data = result['cell_data']
                remaining_bombs = result['remaining_bombs']
                
                logger.info(f"Remaining bombs: {remaining_bombs}")
                logger.info(f"Total bombs: {total_bombs}")
                logger.info("\nCurrent game state:")
                print(game_state)

                # Calculate next move
                solver = MinesweeperSolver(game_state, cell_data, total_bombs)
                move = solver.get_best_move()

                # Visualizer
                visualizer.update(game_state, solver.mine_probabilities)
                
                if move.action != 'none':
                    logger.info(f"\nMaking move:")
                    logger.info(f"Position: ({move.row}, {move.col})")
                    logger.info(f"Action: {move.action}")
                    logger.info(f"Mine probability: {move.probability:.2%}")
                    
                    # Execute move
                    screen_x, screen_y = solver.get_screen_coords(move)
                    player.execute_move(screen_x, screen_y, move.action)
                    last_move = move

                    # Wait for animation
                    time.sleep(0.5)
                else: 
                    logger.info(f"No valid move found")

                # Check if game is over
                game_over, outcome = is_game_over(game_state, remaining_bombs)
                if game_over:
                    if outcome == "win":
                        logger.info("Game Won! 🎉")
                    else:
                        logger.info("Game Lost! 💥")

                    logger.info("Clicking reset button")
                    time.sleep(0.5)
                    screen_x, screen_y =  reset_button
                    player.execute_move(screen_region.x+screen_x, screen_region.y+screen_y, "click")

                    # Wait for animation
                    time.sleep(0.5)

                    logger.info("Starting new game")
                    
                    # Take new screenshot
                    new_screen = pyautogui.screenshot(region=(
                        screen_region.x,
                        screen_region.y,
                        screen_region.width,
                        screen_region.height
                    ))

                    screen_region.image = cv2.cvtColor(np.array(new_screen), cv2.COLOR_RGB2BGR)
                    break
            
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


def main():
    """Main entry point for Minesweeper bot."""
    running = True
    detector = MinesweeperDetector(MINESWEEPER_GRID_SIZE, TESSERACT_PATH, DEBUG)
    player = MinesweeperPlayer()
    visualizer = MinesweeperVisualizer(MINESWEEPER_GRID_SIZE)
    # Start update thread
    update_thread = threading.Thread(target=play_game, args=(detector, player, visualizer, running), daemon=True)
    update_thread.start()

    # Run GUI in main thread
    try:
        visualizer.run()
    finally:
        running = False
            
    
if __name__ == "__main__":
    main()