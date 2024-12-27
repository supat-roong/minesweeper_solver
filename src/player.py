import time
import pyautogui
import numpy as np
from typing import Tuple
from custom_dataclass import GameOutcome
from screenshot import capture_screen_region, capture_new_screenshot
from solver import MinesweeperSolver
from minesweeper_detector import MinesweeperDetector
from visualizer import MinesweeperVisualizer

class MinesweeperPlayer:
    """
    Automated Minesweeper game player that uses computer vision and AI to play the game.
    
    This class handles:
    - Game state detection and management
    - Move execution via mouse control
    - Win/loss condition checking
    - Continuous gameplay loop with visualization support
    """
    
    def __init__(self, logger) -> None:
        """
        Initialize the Minesweeper player.

        Args:
            logger: Logger instance for tracking game progress and debugging
        """
        self.logger = logger
        self.total_games = 0
        self.games_won = 0
        self.games_lost = 0

    def is_game_over(self, game_state: np.ndarray, remaining_bombs: int) -> GameOutcome:
        """
        Check if the game has ended and determine the outcome.

        Args:
            game_state: Current game board state as a numpy array
            remaining_bombs: Number of unflagged bombs remaining

        Returns:
            GameOutcome with is_over flag and result string ("win"/"loss"/"")
        """
        unopened_count = np.sum(game_state == -1)
        
        if unopened_count == 0 and remaining_bombs == 0:
            return GameOutcome(is_over=True, result="win")
            
        if np.any(game_state == -3):  # -3 indicates revealed bomb
            return GameOutcome(is_over=True, result="loss")
        
        return GameOutcome(is_over=False, result="")
        
    def execute_move(self, x: int, y: int, action: str) -> None:
        """
        Execute a mouse action at the specified screen coordinates.

        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            action: Type of move ("click" for left click, "flag" for right click)
        """
        pyautogui.moveTo(x, y)
        time.sleep(0.1)  # Stability delay
        
        if action == 'click':
            pyautogui.click(x, y)
        elif action == 'flag':
            pyautogui.rightClick(x, y)

    def play_game(self, detector: MinesweeperDetector, visualizer: MinesweeperVisualizer = None, running: bool = True) -> None:
        """
        Main game loop that continuously plays Minesweeper until stopped.

        Args:
            detector: Object that processes screenshots to detect game state
            visualizer: Optional visualizer for displaying game state
            running: Flag to control game loop continuation

        The loop follows this sequence:
        1. Capture and analyze game board
        2. Calculate optimal move
        3. Execute move and wait for animation
        4. Check game outcome
        5. Reset if game over, continue if not
        """
        try:
            self.logger.info("Starting Minesweeper bot...")
            print("Select the Minesweeper game window...")
            
            screen_region = capture_screen_region()
            if screen_region is None:
                self.logger.error("Failed to capture screen region")
                return

            while running:
                # Game initialization phase
                self.logger.info("\nStarting game")
                result = detector.process_screenshot(screen_region)
                game_state = np.array(result['game_state'])
                total_bombs = result['total_bombs']
                remaining_bombs = result['remaining_bombs']
                cell_data = result['cell_data']
                reset_button = result['reset_button']
                
                self._log_game_state(total_bombs, remaining_bombs, game_state)
                
                # Main game loop
                last_move = None
                while running:
                    # Move calculation and execution
                    solver = MinesweeperSolver(game_state, cell_data, total_bombs)
                    move = solver.get_best_move()
                    
                    if visualizer:
                        visualizer.update(game_state, solver.mine_probabilities)
                    
                    if not move or move.action == 'none':
                        self.logger.error("No valid moves found")
                        break
                    
                    self._log_move_info(move)
                    screen_x, screen_y = solver.get_screen_coords(move)
                    self.execute_move(screen_x, screen_y, move.action)
                    last_move = move
                    time.sleep(0.5)  # Animation delay
                    
                    # Game state update
                    screen_region = capture_new_screenshot(screen_region)
                    result = (detector.update_after_move(screen_region, last_move, game_state, 
                                                       cell_data, remaining_bombs)
                             if last_move else detector.process_screenshot(screen_region))
                    
                    game_state = np.array(result['game_state'])
                    cell_data = result['cell_data']
                    remaining_bombs = result['remaining_bombs']
                    
                    self._log_game_state(total_bombs, remaining_bombs, game_state)
                    
                    # Game over check
                    game_outcome = self.is_game_over(game_state, remaining_bombs)
                    if game_outcome.is_over:
                        self._handle_game_over(game_outcome.result, screen_region, reset_button)
                        self._update_game_stats(game_outcome.result)
                        self._log_win_percentage()
                        screen_region = capture_new_screenshot(screen_region)
                        break

        except KeyboardInterrupt:
            self.logger.info("\nProgram terminated by user")
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}", exc_info=True)

    def _log_game_state(self, total_bombs: int, remaining_bombs: int, game_state: np.ndarray) -> None:
        """Log current game state information."""
        self.logger.info(f"Total bombs: {total_bombs}")
        self.logger.info(f"Remaining bombs: {remaining_bombs}")
        self.logger.info("\nCurrent game state:")
        print(game_state)

    def _log_move_info(self, move) -> None:
        """Log information about the next move to be executed."""
        self.logger.info(f"\nMaking move:")
        self.logger.info(f"Position: ({move.row}, {move.col})")
        self.logger.info(f"Action: {move.action}")
        self.logger.info(f"Mine probability: {move.probability:.2%}")

    def _handle_game_over(self, outcome: str, screen_region, reset_button: Tuple[int, int]) -> None:
        """Handle game over condition and reset for new game."""
        self.logger.info("Game Won! 🎉" if outcome == "win" else "Game Lost! 💥")
        self.logger.info("Clicking reset button")
        time.sleep(0.5)
        screen_x, screen_y = reset_button
        self.execute_move(screen_region.x + screen_x, screen_region.y + screen_y, "click")
        time.sleep(0.5)
        self.logger.info("Starting new game")

    def _update_game_stats(self, outcome: str) -> None:
        """
        Update the game statistics after a game ends.

        Args:
            outcome: Result of the game ("win" or "loss")
        """
        self.total_games += 1
        if outcome == "win":
            self.games_won += 1
        elif outcome == "loss":
            self.games_lost += 1

    def _log_win_percentage(self) -> None:
        """
        Log the current win percentage.
        """
        if self.total_games > 0:
            win_percentage = (self.games_won / self.total_games) * 100
            self.logger.info(f"Games Played: {self.total_games}")
            self.logger.info(f"Games Won: {self.games_won}")
            self.logger.info(f"Games Lost: {self.games_lost}")
            self.logger.info(f"Win Percentage: {win_percentage:.2f}%")
        else:
            self.logger.info("No games played yet.")