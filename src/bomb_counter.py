import cv2
import numpy as np
from typing import Optional, List, Tuple
from debugger import Debugger
from seven_segment_ocr import SevenSegmentOCR

class BombCounterDetector:
    """Class responsible for detecting and reading the bomb counter LED display."""
    
    def __init__(self, debugger: Optional[Debugger] = None):
        self.debugger = debugger
        self.seven_segment_ocr = SevenSegmentOCR(debugger)

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        if self.debugger is not None:
            self.debugger.show_debug(title, img, cmap)

    def detect(self, img: np.ndarray) -> Optional[int]:
        """Detect and read the bomb counter value."""
        height, width = img.shape[:2]
        top_left = img[:height // 2, :width // 2]
        self.show_debug("Top Left Quadrant", top_left)
        
        hsv = cv2.cvtColor(top_left, cv2.COLOR_BGR2HSV)
        red_mask = self._create_led_mask(hsv)
        self.show_debug("Red LED Mask", red_mask, 'gray')
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        bounds = self._get_led_bounds(contours)
        if bounds is None:
            return None
            
        min_x, min_y, max_x, max_y = bounds
        counter_region = top_left[min_y:max_y, min_x:max_x]
        led_mask = red_mask[min_y:max_y, min_x:max_x]
        
        return self._process_counter_image(counter_region, led_mask)

    def _create_led_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create mask for red LED segments."""
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def _get_led_bounds(self, contours: List[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box for LED display."""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        found_valid = False
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 20:  # Filter noise
                found_valid = True
                min_x, max_x = min(min_x, x), max(max_x, x + w)
                min_y, max_y = min(min_y, y), max(max_y, y + h)
                
        return (min_x, min_y, max_x, max_y) if found_valid else None

    def _process_counter_image(self, region: np.ndarray, mask: np.ndarray) -> Optional[int]:
        """Process counter image to extract digits."""
        background = np.full(region.shape[:2], 255, dtype=np.uint8)
        result = background.copy()
        result[mask > 0] = 0
        
        self.show_debug("Final OCR Image", result, 'gray')
        digits = self.seven_segment_ocr.recognize_digits(result)
        return int(''.join(str(d) for d in digits)) if digits else None