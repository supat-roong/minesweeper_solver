import numpy as np
import cv2
import pyautogui
from tkinter import *
import tkinter as tk
import win32gui
import win32con
from custom_dataclass import ScreenRegion
from typing import Optional, List

class SnippingTool:
    """
    Screen region selection tool with semi-transparent overlay.
    
    Features:
    - Semi-transparent fullscreen overlay
    - Click and drag selection
    - Escape to cancel
    - Clickthrough capability using win32gui
    """
    def __init__(self):
        """Initialize the snipping tool window and canvas."""
        self.root = tk.Tk()
        self._setup_window()
        self._setup_canvas()
        self._setup_variables()
        self._bind_events()

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.root.attributes('-alpha', 0.3)  # Semi-transparent
        self.root.attributes('-fullscreen', True)  # Cover whole screen
        self.root.attributes('-topmost', True)  # Stay on top
        self.root.configure(background='grey')
        
        # Make window clickthrough
        hwnd = win32gui.GetParent(self.root.winfo_id())
        extended_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd, 
            win32con.GWL_EXSTYLE, 
            extended_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
        )

    def _setup_canvas(self) -> None:
        """Create and configure the drawing canvas."""
        self.canvas = Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=BOTH, expand=YES)

    def _setup_variables(self) -> None:
        """Initialize instance variables."""
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.current_rect: Optional[int] = None
        self.rect_coords: List[int] = []

    def _bind_events(self) -> None:
        """Bind mouse and keyboard events."""
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Escape>", self.on_escape)

    def on_press(self, event) -> None:
        """
        Handle mouse button press.
        
        Args:
            event: Tkinter event containing coordinates
        """
        self.start_x = event.x
        self.start_y = event.y
        
        # Clear any existing rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            
        # Create new selection rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, 
            self.start_y, 
            self.start_x, 
            self.start_y, 
            outline='red', 
            width=2
        )
    
    def on_drag(self, event) -> None:
        """
        Handle mouse drag to resize selection.
        
        Args:
            event: Tkinter event containing coordinates
        """
        if self.current_rect:
            self.canvas.coords(
                self.current_rect, 
                self.start_x, 
                self.start_y, 
                event.x, 
                event.y
            )
    
    def on_release(self, event) -> None:
        """
        Handle mouse button release to finalize selection.
        
        Args:
            event: Tkinter event containing coordinates
        """
        if self.start_x is not None and self.start_y is not None:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            self.rect_coords = [x1, y1, x2, y2]
        self.root.quit()
    
    def on_escape(self, event) -> None:
        """
        Handle escape key to cancel selection.
        
        Args:
            event: Tkinter event (unused)
        """
        self.rect_coords = []
        self.root.quit()

def capture_screen_region() -> Optional[ScreenRegion]:
    """
    Capture a user-selected region of the screen.
    
    Returns:
        ScreenRegion object containing coordinates and image data,
        or None if selection was cancelled
        
    Notes:
        - Adds small delay to ensure screen is ready
        - Converts screenshot from RGB to BGR for OpenCV compatibility
    """
    try:
        # Small delay to ensure screen is ready
        pyautogui.sleep(0.1)
        
        # Create and run snipping tool
        snip = SnippingTool()
        snip.root.mainloop()
        snip.root.destroy()
        
        # Process selection if one was made
        if snip.rect_coords:
            x1, y1, x2, y2 = snip.rect_coords
            
            # Validate selection size
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                print("Invalid selection size: width and height must be positive")
                return None
            
            try:
                # Capture and process screenshot
                screenshot = pyautogui.screenshot(region=(x1, y1, x2-x1, y2-y1))
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                return ScreenRegion(x1, y1, x2-x1, y2-y1, frame)
            except Exception as e:
                print(f"Error capturing screenshot: {str(e)}")
                return None
        
        return None
        
    except Exception as e:
        print(f"Error in screen capture: {str(e)}")
        return None