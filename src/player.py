import time
import pyautogui

class MinesweeperPlayer:    
    def execute_move(self, x: int, y: int, action: str) -> None:
        """Execute a move at the given screen coordinates."""
        # Move mouse to target
        pyautogui.moveTo(x, y)
        time.sleep(0.1)  # Small delay for stability
        
        # Perform click based on action type
        if action == 'click':
            pyautogui.click(x, y)
        elif action == 'flag':
            pyautogui.rightClick(x, y)