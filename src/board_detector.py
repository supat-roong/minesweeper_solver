import cv2
import numpy as np
from typing import Tuple, List, Optional
from debugger import Debugger

class BoardDetector:
    """Class responsible for detecting the Minesweeper game board."""
    
    def __init__(self, grid_size: Tuple[int, int], debugger: Optional[Debugger] = None):
        self.grid_size = grid_size
        self.debugger = debugger

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        if self.debugger is not None:
            self.debugger.show_debug(title, img, cmap)

    def detect(self, img: np.ndarray) -> Tuple[Tuple[np.ndarray, Tuple[int, int]], Tuple[float, float]]:
        """Detect the game board and reset button."""
        self.show_debug("Original Input", img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.show_debug("HSV Conversion", hsv)
        
        gray_mask = self._create_gray_mask(hsv)
        self.show_debug("Gray Mask", gray_mask, 'gray')
        
        cell_candidates, reset_button = self._find_cell_candidates(img, gray_mask)
        
        if cell_candidates:
            return self._extract_grid_region(img, cell_candidates), reset_button
        
        return (img, (0, 0)), (0, 0)

    def _create_gray_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create a mask for gray colored regions."""
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 30, 220])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _find_cell_candidates(self, img: np.ndarray, gray_mask: np.ndarray) -> Tuple[List[np.ndarray], Optional[Tuple[float, float]]]:
        """Find potential cell regions and reset button location."""
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        self._visualize_contours(img, contours)
        
        cell_candidates = []
        reset_button = None
        filtered_viz = img.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 25:
                x, y, w, h = cv2.boundingRect(contour)
                
                if self._is_valid_cell(contour, w, h, area, gray_mask):
                    cell_candidates.append(contour)
                    cv2.drawContours(filtered_viz, [contour], -1, (0, 255, 0), 2)
                elif self._is_reset_button(contour, w, h, area, img):
                    reset_button = (x + w/2, y + h/2)
        
        self.show_debug("Filtered Cells", filtered_viz)
        return cell_candidates, reset_button

    def _visualize_contours(self, img: np.ndarray, contours: List[np.ndarray]) -> None:
        """Create debug visualizations for contours."""
        if self.debugger is None:
            return
            
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        all_contours_viz = img.copy()
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            cv2.drawContours(all_contours_viz, [contour], -1, color, 2)
            
            single_viz = img.copy()
            cv2.drawContours(single_viz, [contour], -1, color, 2)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(single_viz, f"Area: {area:.0f}, W: {w}, H: {h}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self.show_debug(f"Contour {i}", single_viz)
        
        self.show_debug("All Contours (Colored)", all_contours_viz)

    def _is_reset_button(self, contour: np.ndarray, w: int, h: int, area: float, img: np.ndarray) -> bool:
        """Check if a region is the reset button."""
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
            
        canvas = img.copy()
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
        
        lower_yellow = np.array([0, 200, 200])
        upper_yellow = np.array([20, 255, 255])
        yellow_mask = cv2.inRange(canvas, lower_yellow, upper_yellow)
        
        mask = np.zeros_like(yellow_mask)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        yellow_pixels = np.sum((mask == 255) & (yellow_mask == 255))
        return yellow_pixels / area > 0.1

    def _is_valid_cell(self, contour: np.ndarray, w: int, h: int, area: float, gray_mask: np.ndarray) -> bool:
        """Check if a region is a valid cell candidate."""
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
            
        mask = np.zeros_like(gray_mask)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        gray_pixels = np.sum((mask == 255) & (gray_mask == 255))
        return gray_pixels / area > 0.7

    def _extract_grid_region(self, img: np.ndarray, cell_candidates: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract the gameplay grid region and calculate its offset."""
        x_coords, y_coords = [], []
        for cell in cell_candidates:
            for point in cell:
                [[x, y]] = point
                x_coords.append(x)
                y_coords.append(y)

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        grid = img[min_y:max_y, min_x:max_x]
        self.show_debug("Final Grid Detection", grid)
        
        return grid, (min_x, min_y)