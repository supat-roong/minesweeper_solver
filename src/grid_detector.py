import cv2
import numpy as np
from typing import List, Tuple, Optional
from debugger import Debugger

class GridLineDetector:
    """Class responsible for detecting grid lines in the Minesweeper board."""
    
    def __init__(self, grid_size: Tuple[int, int], debugger: Optional[Debugger] = None):
        """Initialize the grid line detector.
        
        Args:
            grid_size: Tuple of (rows, columns) for the Minesweeper grid
            debugger: Optional debugger for visualization
        """
        self.grid_size = grid_size
        self.debugger = debugger

    def detect(self, img: np.ndarray) -> Tuple[List[int], List[int]]:
        """Detect horizontal and vertical grid lines.
        
        Args:
            img: Input image of the Minesweeper board
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines) where each element is a list
            of pixel positions where lines were detected
        """
        # Preprocess image
        binary = self._preprocess_image(img)
        self.show_debug("Binary Image", binary, 'gray')
        
        # Get line projections
        h_proj, v_proj = self._get_line_projections(binary)
        
        # Find grid lines
        horizontal_lines = self._find_grid_lines(h_proj, self.grid_size[0] + 1)
        vertical_lines = self._find_grid_lines(v_proj, self.grid_size[1] + 1)
        
        # Visualize result
        grid_vis = self.visualize_grid(img, (horizontal_lines, vertical_lines))
        self.show_debug("Detected Grid Lines", grid_vis)
        
        return horizontal_lines, vertical_lines

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for grid line detection.
        
        Args:
            img: Input image
            
        Returns:
            Binary image ready for line detection
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        
        # Normalize image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        return binary

    def _process_lines(self, binary_img: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Process image to detect lines using morphological operations.
        
        Args:
            binary_img: Binary input image
            kernel_size: Size of the morphological kernel
            
        Returns:
            Processed image highlighting detected lines
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        processed = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
        return processed

    def _get_line_projections(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get horizontal and vertical projections of the binary image.
        
        Args:
            binary_img: Binary input image
            
        Returns:
            Tuple of (horizontal_projections, vertical_projections)
        """
        # Create kernels for horizontal and vertical lines
        h_kernel_size = (int(binary_img.shape[1]/self.grid_size[1]), 1)
        v_kernel_size = (1, int(binary_img.shape[0]/self.grid_size[0]))
        
        # Process image for horizontal and vertical lines
        h_lines = self._process_lines(binary_img, h_kernel_size)
        v_lines = self._process_lines(binary_img, v_kernel_size)
        
        # Calculate projections
        h_proj = np.sum(h_lines, axis=1)
        v_proj = np.sum(v_lines, axis=0)
        
        return h_proj, v_proj

    def _find_grid_lines(self, projections: np.ndarray, expected_lines: int) -> List[int]:
        """Find grid line positions from projection data.
        
        Args:
            projections: 1D array of projection values
            expected_lines: Expected number of grid lines
            
        Returns:
            List of pixel positions where lines were detected
        """
        # Normalize projections
        projections = projections / projections.max()
        peaks = np.where(projections > 0.1)[0]
        
        def group_peaks(peaks: np.ndarray, threshold: int = 20) -> List[int]:
            """Group nearby peaks and calculate their mean positions."""
            if len(peaks) == 0:
                return []
            groups = [[peaks[0]]]
            for peak in peaks[1:]:
                if peak - groups[-1][-1] <= threshold:
                    groups[-1].append(peak)
                else:
                    groups.append([peak])
            return [int(np.mean(group)) for group in groups]
        
        lines = group_peaks(peaks)
        
        # If we don't have enough lines, interpolate
        if len(lines) < expected_lines:
            lines = np.linspace(0, len(projections)-1, expected_lines).astype(int).tolist()
        
        return sorted(lines)

    def visualize_grid(self, img: np.ndarray, lines: Tuple[List[int], List[int]]) -> np.ndarray:
        """Visualize detected grid lines on the image.
        
        Args:
            img: Original input image
            lines: Tuple of (horizontal_lines, vertical_lines)
            
        Returns:
            Image with visualized grid lines
        """
        vis_img = img.copy()
        h_lines, v_lines = lines
        
        # Draw horizontal lines in green
        for y in h_lines:
            cv2.line(vis_img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        
        # Draw vertical lines in blue
        for x in v_lines:
            cv2.line(vis_img, (x, 0), (x, img.shape[0]), (255, 0, 0), 1)
        
        return vis_img

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """Display debug visualization if debugger is enabled.
        
        Args:
            title: Title for the debug visualization
            img: Image to display
            cmap: Optional colormap for visualization
        """
        if self.debugger is not None:
            self.debugger.show_debug(title, img, cmap)