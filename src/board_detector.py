import cv2
import numpy as np
from typing import Tuple, List, Optional
from debugger import Debugger

class BoardDetector:
    """
    Class responsible for detecting the Minesweeper game board and the reset button.

    This class uses image processing techniques to identify the grid cells of the game board 
    and locate the reset button by analyzing the color and shape features in the provided image.
    """
    
    def __init__(self, grid_size: Tuple[int, int], debugger: Optional[Debugger] = None):
        """
        Initialize the BoardDetector.

        Args:
            grid_size (Tuple[int, int]): Size of the Minesweeper grid as (rows, columns).
            debugger (Optional[Debugger]): Instance of a debugger for visualizing intermediate steps.
        """
        self.grid_size = grid_size
        self.debugger = debugger

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """
        Show debug images if a debugger is provided.

        Args:
            title (str): Title for the debug window.
            img (np.ndarray): Image to be displayed.
            cmap (Optional[str]): Colormap to be applied (e.g., 'gray').
        """
        if self.debugger is not None:
            self.debugger.show_debug(title, img, cmap)

    def detect(self, img: np.ndarray) -> Tuple[Tuple[np.ndarray, Tuple[int, int]], Tuple[float, float]]:
        """
        Detect the Minesweeper game board and the reset button in the given image.

        Args:
            img (np.ndarray): Input image containing the Minesweeper game.

        Returns:
            Tuple[Tuple[np.ndarray, Tuple[int, int]], Tuple[float, float]]:
                - Extracted grid region and its top-left offset (relative to the input image).
                - Center coordinates of the reset button as (x, y). Returns (0, 0) if not found.
        """
        self.show_debug("Original Input", img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.show_debug("HSV Conversion", hsv)
        
        # Create a mask for detecting gray regions (board cells).
        gray_mask = self._create_gray_mask(hsv)
        self.show_debug("Gray Mask", gray_mask, 'gray')
        
        # Identify potential cell regions and reset button.
        cell_candidates, reset_button = self._find_cell_candidates(img, gray_mask)
        
        if cell_candidates:
            # Extract the grid region and calculate its offset.
            return self._extract_grid_region(img, cell_candidates), reset_button
        
        # Default return values if no cells are detected.
        return (img, (0, 0)), (0, 0)

    def _create_gray_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Create a mask to detect gray-colored regions in the image.

        Args:
            hsv (np.ndarray): HSV representation of the input image.

        Returns:
            np.ndarray: Binary mask highlighting gray regions.
        """
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 30, 220])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Refine the mask using morphological operations.
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _find_cell_candidates(self, img: np.ndarray, gray_mask: np.ndarray) -> Tuple[List[np.ndarray], Optional[Tuple[float, float]]]:
        """
        Identify potential cell regions and the reset button.

        Args:
            img (np.ndarray): Original input image.
            gray_mask (np.ndarray): Binary mask highlighting gray regions.

        Returns:
            Tuple[List[np.ndarray], Optional[Tuple[float, float]]]:
                - List of contours representing valid cell regions.
                - Center coordinates of the reset button (x, y), or None if not found.
        """
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Debug visualization of all contours.
        self._visualize_contours(img, contours)
        
        cell_candidates = []
        reset_button = None
        filtered_viz = img.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 25:  # Filter out small noise contours.
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the contour matches the criteria for a cell.
                if self._is_valid_cell(contour, w, h, area, gray_mask):
                    cell_candidates.append(contour)
                    cv2.drawContours(filtered_viz, [contour], -1, (0, 255, 0), 2)
                # Check if the contour matches the criteria for the reset button.
                elif self._is_reset_button(contour, w, h, area, img):
                    reset_button = (x + w/2, y + h/2)
        
        self.show_debug("Filtered Cells", filtered_viz)
        return cell_candidates, reset_button

    def _visualize_contours(self, img: np.ndarray, contours: List[np.ndarray]) -> None:
        """
        Create debug visualizations for all detected contours.

        Args:
            img (np.ndarray): Original input image.
            contours (List[np.ndarray]): List of detected contours.
        """
        if self.debugger is None:
            return
            
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        all_contours_viz = img.copy()
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            cv2.drawContours(all_contours_viz, [contour], -1, color, 2)
            
            # Annotate each contour with its properties for debugging.
            single_viz = img.copy()
            cv2.drawContours(single_viz, [contour], -1, color, 2)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(single_viz, f"Area: {area:.0f}, W: {w}, H: {h}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self.show_debug(f"Contour {i}", single_viz)
        
        self.show_debug("All Contours (Colored)", all_contours_viz)

    def _is_reset_button(self, contour: np.ndarray, w: int, h: int, area: float, img: np.ndarray) -> bool:
        """
        Determine if a given contour corresponds to the reset button.

        Args:
            contour (np.ndarray): Contour to evaluate.
            w (int): Width of the bounding rectangle.
            h (int): Height of the bounding rectangle.
            area (float): Area of the contour.
            img (np.ndarray): Original input image.

        Returns:
            bool: True if the contour matches the reset button criteria, False otherwise.
        """
        # Check for a roughly square shape and sufficient area.
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
            
        canvas = img.copy()
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
        
        # Look for yellow regions within the contour to confirm it as a reset button.
        lower_yellow = np.array([0, 200, 200])
        upper_yellow = np.array([20, 255, 255])
        yellow_mask = cv2.inRange(canvas, lower_yellow, upper_yellow)
        
        mask = np.zeros_like(yellow_mask)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        yellow_pixels = np.sum((mask == 255) & (yellow_mask == 255))
        return yellow_pixels / area > 0.1

    def _is_valid_cell(self, contour: np.ndarray, w: int, h: int, area: float, gray_mask: np.ndarray) -> bool:
        """
        Determine if a given contour corresponds to a valid cell region.

        Args:
            contour (np.ndarray): Contour to evaluate.
            w (int): Width of the bounding rectangle.
            h (int): Height of the bounding rectangle.
            area (float): Area of the contour.
            gray_mask (np.ndarray): Binary mask of gray regions.

        Returns:
            bool: True if the contour matches the cell criteria, False otherwise.
        """
        # Check for a roughly square shape and sufficient area.
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
            
        mask = np.zeros_like(gray_mask)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Ensure most pixels within the contour are part of the gray mask.
        match_pixels = np.sum((mask == 255) & (gray_mask == 255))
        return match_pixels / area > 0.8

    def _extract_grid_region(self, img: np.ndarray, cell_candidates: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract the game board region based on detected cell candidates.

        Args:
            img (np.ndarray): Original input image.
            cell_candidates (List[np.ndarray]): List of contours representing valid cells.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - Cropped grid region as an image.
                - Top-left offset of the grid relative to the input image.
        """
        # Merge all contours into one bounding box for cropping.
        x, y, w, h = cv2.boundingRect(np.vstack(cell_candidates))
        grid_img = img[y:y+h, x:x+w]
        self.show_debug("Grid Region", grid_img)
        return grid_img, (x, y)
