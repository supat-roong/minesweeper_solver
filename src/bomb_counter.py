import cv2
import numpy as np
from typing import Optional, List, Tuple
from debugger import Debugger
from seven_segment_ocr import SevenSegmentOCR

class BombCounterDetector:
    """
    Class responsible for detecting and reading the bomb counter LED display in Minesweeper.
    
    This class identifies the red LED segments of the bomb counter using color filtering, 
    isolates the bounding box for the counter, and then uses an OCR system to read the digits.
    """

    def __init__(self, debugger: Optional[Debugger] = None):
        """
        Initialize the BombCounterDetector.
        
        Args:
            debugger (Optional[Debugger]): An optional debugger instance for visualization.
        """
        self.debugger = debugger
        self.seven_segment_ocr = SevenSegmentOCR(debugger)

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """
        Display debug images through the debugger, if available.
        
        Args:
            title (str): Title of the debug window.
            img (np.ndarray): The image to display.
            cmap (Optional[str]): Optional color map (e.g., 'gray').
        """
        if self.debugger is not None:
            self.debugger.show_debug(title, img, cmap)

    def detect(self, img: np.ndarray) -> Optional[int]:
        """
        Detect and read the bomb counter value from the input image.
        
        Args:
            img (np.ndarray): Input image containing the bomb counter.
        
        Returns:
            Optional[int]: The detected bomb counter value, or None if detection fails.
        """
        # Extract the top-left quadrant of the image where the bomb counter is expected.
        height, width = img.shape[:2]
        top_left = img[:height // 2, :width // 2]
        self.show_debug("Top Left Quadrant", top_left)
        
        # Convert the image to HSV color space and create a mask for the red LED segments.
        hsv = cv2.cvtColor(top_left, cv2.COLOR_BGR2HSV)
        red_mask = self._create_led_mask(hsv)
        self.show_debug("Red LED Mask", red_mask, 'gray')
        
        # Find contours in the red mask.
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Get the bounding box for the detected LED display.
        bounds = self._get_led_bounds(contours)
        if bounds is None:
            return None
            
        # Extract the counter region and mask for further processing.
        min_x, min_y, max_x, max_y = bounds
        counter_region = top_left[min_y:max_y, min_x:max_x]
        led_mask = red_mask[min_y:max_y, min_x:max_x]
        
        # Process the counter image and recognize the digits.
        return self._process_counter_image(counter_region, led_mask)

    def _create_led_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for the red LED segments in the HSV image.
        
        Args:
            hsv (np.ndarray): Input image in HSV color space.
        
        Returns:
            np.ndarray: Binary mask highlighting the red LED segments.
        """
        # Define red color ranges in HSV space (for both lower and upper hues).
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        
        # Create masks for both red ranges and combine them.
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def _get_led_bounds(self, contours: List[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        """
        Determine the bounding box for the LED display based on contours.
        
        Args:
            contours (List[np.ndarray]): List of contours detected in the image.
        
        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box as (min_x, min_y, max_x, max_y), 
                                                 or None if no valid box is found.
        """
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        found_valid = False
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small contours that are likely noise.
            if w * h > 20:  
                found_valid = True
                min_x, max_x = min(min_x, x), max(max_x, x + w)
                min_y, max_y = min(min_y, y), max(max_y, y + h)
                
        return (min_x, min_y, max_x, max_y) if found_valid else None

    def _process_counter_image(self, region: np.ndarray, mask: np.ndarray) -> Optional[int]:
        """
        Process the extracted counter region to recognize the digits.
        
        Args:
            region (np.ndarray): The image region containing the counter.
            mask (np.ndarray): Binary mask for the counter's LED segments.
        
        Returns:
            Optional[int]: The recognized counter value, or None if recognition fails.
        """
        # Create a blank white image and overlay the LED mask.
        background = np.full(region.shape[:2], 255, dtype=np.uint8)
        result = background.copy()
        result[mask > 0] = 0
        
        self.show_debug("Final OCR Image", result, 'gray')
        
        # Recognize digits using the seven-segment OCR system.
        digits = self.seven_segment_ocr.recognize_digits(result)
        return int(''.join(str(d) for d in digits)) if digits else None
