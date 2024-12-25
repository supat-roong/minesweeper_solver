import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional

class Debugger:
    def __init__(self):
        self.current_step = 0
        
    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """Display debug image and wait for keypress."""
        self.current_step += 1
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Step {self.current_step}: {title}")
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        plt.imshow(img, cmap=cmap)
        plt.axis('on')
        plt.grid(True)
        plt.figtext(0.5, 0.01, 'Press any key to continue...', ha='center', va='bottom')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()