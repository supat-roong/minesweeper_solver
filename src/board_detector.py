import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from seven_segment_ocr import SevenSegmentOCR
from debugger import Debugger
from config import Colors
from custom_dataclass import CellData, CellPosition, ScreenRegion, Move

class MinesweeperDetector:   
    def __init__(self, grid_size: Tuple[int, int], tesseract_path: Optional[str] = None, debug: bool = False):
        """Initialize the detector with given parameters."""
        self.grid_size = grid_size
        self.debug = debug
        self.debugger = Debugger() if debug else None
        self.seven_segment_ocr = SevenSegmentOCR(debug=debug)
        self.cell_data = {}  # Dictionary to store CellData objects
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            

    def show_debug(self, title: str, img: np.ndarray, cmap: Optional[str] = None) -> None:
        """Wrapper for debug visualization."""
        if self.debug and self.debugger:
            self.debugger.show_debug(title, img, cmap)

    def detect_bomb_counter(self, img: np.ndarray) -> Optional[int]:
        """Detect and read the bomb counter LED display."""
        height, width = img.shape[:2]
        top_left = img[:height // 2, :width // 2]
        self.show_debug("Top Left Quadrant", top_left)
        
        # Create LED mask
        hsv = cv2.cvtColor(top_left, cv2.COLOR_BGR2HSV)
        red_mask = self._create_led_mask(hsv)
        self.show_debug("Red LED Mask", red_mask, 'gray')
        
        # Find LED region
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        bounds = self._get_led_bounds(contours)
        if bounds is None:
            return None
            
        # Process counter region
        min_x, min_y, max_x, max_y = bounds
        counter_region = top_left[min_y:max_y, min_x:max_x]
        led_mask = red_mask[min_y:max_y, min_x:max_x]
        
        return self._process_counter_image(counter_region, led_mask)

    def _create_led_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create mask for red LED segments."""
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
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
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                
        return (min_x, min_y, max_x, max_y) if found_valid else None

    def _process_counter_image(self, region: np.ndarray, mask: np.ndarray) -> Optional[int]:
        """Process counter image to extract digits."""
        background = np.full(region.shape[:2], 255, dtype=np.uint8)
        result = background.copy()
        result[mask > 0] = 0
        
        self.show_debug("Final OCR Image", result, 'gray')
        digits = self.seven_segment_ocr.recognize_digits(result)
        return int(''.join(str(d) for d in digits)) if digits else None

    def detect_board_region(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract the Minesweeper gameplay grid and return with its offset."""
        self.show_debug("Original Input", img)
        
        # Create gray mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.show_debug("HSV Conversion", hsv)
        
        gray_mask = self._create_gray_mask(hsv)
        self.show_debug("Gray Mask", gray_mask, 'gray')
        
        # Get cell candidates
        cell_candidates, reset_button = self._find_cell_candidates(img, gray_mask)
        
        if cell_candidates:
            return self._extract_grid_region(img, cell_candidates), reset_button
        
        return (img, (0, 0)), (0, 0)


    def _create_gray_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create a mask for gray colored regions."""
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 30, 220])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def _find_cell_candidates(self, img: np.ndarray, gray_mask: np.ndarray) -> list:
        """
        Find potential cell regions using detailed contour analysis with individual visualization.
        """
        # Find ALL contour points without approximation
        contours, _ = cv2.findContours(
            gray_mask, 
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )
        
        print(f"Initial contours detected: {len(contours)}")
        
        # Create separate visualizations for each contour
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Create a visualization with all contours in different colors
        all_contours_viz = img.copy()
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            cv2.drawContours(all_contours_viz, [contour], -1, color, 2)
            
            # Create individual visualization for this contour
            single_contour_viz = img.copy()
            cv2.drawContours(single_contour_viz, [contour], -1, color, 2)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(single_contour_viz, 
                    f"Area: {area:.0f}, W: {w}, H: {h}", 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2)
            self.show_debug(f"Contour {i}", single_contour_viz)
        
        self.show_debug("All Contours (Colored)", all_contours_viz)
                
        cell_candidates = []
        filtered_viz = img.copy()

        reset_button_candidate = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 25:  # Minimal threshold
                x, y, w, h = cv2.boundingRect(contour)
                                
                if self._is_valid_cell(contour, w, h, area, gray_mask):
                    cell_candidates.append(contour)
                    cv2.drawContours(filtered_viz, [contour], -1, (0, 255, 0), 2)
                elif self._is_reset_button(contour, w, h, area, img):
                    reset_button_candidate = x + w/2, y + h/2
        
        print(f"Final cell candidates: {len(cell_candidates)}")
        self.show_debug("Filtered Cells", filtered_viz)
        
        return cell_candidates, reset_button_candidate

    def _extract_grid_region(self, img: np.ndarray, cell_candidates: list) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract the gameplay grid region and calculate its offset.
        
        Returns:
            Tuple of (grid_image, (offset_x, offset_y))
        """
        # Find min/max coordinates from all cell contour points
        x_coords = []
        y_coords = []
        for cell in cell_candidates:
            for point in cell:
                [[x, y]] = point  # Extract coordinates from contour point
                x_coords.append(x)
                y_coords.append(y)

        # Calculate boundaries
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Extract grid region
        grid = img[min_y:max_y, min_x:max_x]
        
        # Calculate offset relative to input image
        offset = (min_x, min_y)
        
        self.show_debug("Final Grid Detection", grid)
        
        return grid, offset
    
    def _is_reset_button(self, contour, w, h, area, img):
        """Check if a region is a valid cell candidate."""
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        plt.imsave(f"aaaaa{w}.png", img)
        lower_yellow = np.array([0, 200, 200])
        upper_yellow = np.array([20, 255, 255])
        yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

        # Create a blank mask the same size as the input image
        mask = np.zeros_like(yellow_mask)
        
        # Draw the contour filled on the mask
        cv2.drawContours(mask, [contour], 0, 255, -1)

        # Calculate percentage of gray pixels only within the contour area
        yellow_pixels = np.sum((mask == 255) & (yellow_mask == 255))
        
        yellow_percentage = yellow_pixels / area
        return yellow_percentage > 0.1
    
        

    def _is_valid_cell(self, contour: np.ndarray, w: int, h: int, area: int, 
                      gray_mask: np.ndarray) -> bool:
        """Check if a region is a valid cell candidate."""
        if not (0.9 <= w/h <= 1.1 and area > 100):
            return False
            
        # Create a blank mask the same size as the input image
        mask = np.zeros_like(gray_mask)
        
        # Draw the contour filled on the mask
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Calculate percentage of gray pixels only within the contour area
        gray_pixels = np.sum((mask == 255) & (gray_mask == 255))
        
        gray_percentage = gray_pixels / area
        return gray_percentage > 0.7

    def detect_cell_state(self, cell: np.ndarray, position: tuple) -> int:
        """Detect the state of a single cell."""
        state = -4
        if self._is_color_match(cell, Colors.CELL_COLORS['gray']) > 0.85:
            state = 0
        elif self._is_color_match(cell, Colors.CELL_COLORS['red']) > 0.4 and self._is_color_match(cell, Colors.CELL_COLORS['black']) > 0.2:
            state = -3
        elif self._is_color_match(cell, Colors.CELL_COLORS['red']) > 0.01 and self._is_color_match(cell, Colors.CELL_COLORS['black']) > 0.05:
            state = -2
        elif self._is_color_match(cell, Colors.CELL_COLORS['gray']) > 0.4 and self._is_color_match(cell, Colors.CELL_COLORS['white']) > 0.05:
            state = -1
        else:
            number = self._detect_number_by_color(cell)
            state = number if number is not None else self._detect_number_ocr(cell)
        
        if self.debug:
            self.debug_cell(cell, position, state)
        
        return state

    def process_screenshot(self, screen_region: ScreenRegion) -> dict:
        """Process a Minesweeper screenshot and return game state with cell positions.
        
        Args:
            screen_region: ScreenRegion object containing the captured game area
            
        Returns:
            dict containing game state, cell positions, and board image
        """
        self.show_debug("Original Image", screen_region.image)
        
        # Store screen region for coordinate translation
        self.screen_region = screen_region
        
        total_bombs = self.detect_bomb_counter(screen_region.image)
        print(f"Detected bombs: {total_bombs}")
        
        # Get board region and its offset relative to screen_region
        (board, board_offset), reset_button = self.detect_board_region(screen_region.image)
        self.board_offset_x, self.board_offset_y = board_offset
        
        print(f"Board offset relative to screen region: ({self.board_offset_x}, {self.board_offset_y})")
        
        game_state, cell_data = self._process_board(board)
        
        if self.debug:
            self._visualize_game_state(game_state)
            self._visualize_cell_positions(board, cell_data)
        
        remaining_bombs = total_bombs
        
        return {
            'total_bombs': total_bombs,
            'remaining_bombs': remaining_bombs,
            'game_state': game_state.tolist(),
            'cell_data': cell_data,
            'board_image': board,
            'board_offset': board_offset,
            'screen_region': screen_region,
            'reset_button': reset_button
        }

    def _process_board(self, board: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], CellData]]:
        """Process the game board to detect cell states and positions."""
        # Get grid lines
        horizontal_lines, vertical_lines = self.detect_grid_lines(board)
        self.show_debug("Grid Line Detection", self.visualize_grid(board, (horizontal_lines, vertical_lines)))
        
        # Initialize game state array and cell data dictionary
        game_state = np.zeros(self.grid_size, dtype=int)
        cell_data = {}
        
        # Process each cell
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                # Get cell boundaries
                top = horizontal_lines[i]
                bottom = horizontal_lines[i + 1]
                left = vertical_lines[j]
                right = vertical_lines[j + 1]
                
                # Extract cell with margin
                margin = 3
                cell = board[top + margin:bottom - margin, left + margin:right - margin]
                
                # Detect cell state
                state = self.detect_cell_state(cell, (i, j))
                game_state[i, j] = state
                
                # Convert to absolute screen coordinates
                # Add both screen_region offset and board offset
                abs_left = self.screen_region.x + self.board_offset_x + left + margin
                abs_right = self.screen_region.x + self.board_offset_x + right - margin
                abs_top = self.screen_region.y + self.board_offset_y + top + margin
                abs_bottom = self.screen_region.y + self.board_offset_y + bottom - margin
                abs_center_x = self.screen_region.x + self.board_offset_x + (left + right) // 2
                abs_center_y = self.screen_region.y + self.board_offset_y + (top + bottom) // 2
                
                # Create CellPosition object with absolute screen coordinates
                cell_pos = CellPosition(
                    letter=str(state) if state >= 0 else ('-' if state == -1 else 'F'),
                    screen_x_range=(abs_left, abs_right),
                    screen_y_range=(abs_top, abs_bottom),
                    screen_x=abs_center_x,
                    screen_y=abs_center_y,
                    grid_row=i,
                    grid_col=j
                )
                
                # Store cell data
                cell_data[(i, j)] = CellData(state=state, position=cell_pos)
        
        return game_state, cell_data

    
    def detect_grid_lines(self, img):
        """Detect grid lines in the image."""
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Enhance contrast
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive threshold to handle varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        
        # Increase kernel sizes for better line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                    (int(img.shape[1]/self.grid_size[1]), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                  (1, int(img.shape[0]/self.grid_size[0])))
        
        # Detect lines with morphological operations
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Calculate projections
        horizontal_projections = np.sum(horizontal_lines, axis=1)
        vertical_projections = np.sum(vertical_lines, axis=0)
        
        def find_grid_lines(projections, expected_lines):
            # Normalize projections
            projections = projections / projections.max()
            
            # Find peaks with lower threshold
            peaks = np.where(projections > 0.1)[0]
            
            def group_peaks(peaks, threshold=20):
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
            
            # If we still don't have enough lines, try to interpolate
            if len(lines) < expected_lines:
                # Use image dimensions to create equally spaced lines
                lines = np.linspace(0, len(projections)-1, expected_lines).astype(int)
            
            return sorted(lines)
        
        horizontal_lines = find_grid_lines(horizontal_projections, self.grid_size[0] + 1)
        vertical_lines = find_grid_lines(vertical_projections, self.grid_size[1] + 1)
        
        return horizontal_lines, vertical_lines
    
    def visualize_grid(self, img, lines):
        """Visualize detected grid lines on the image."""
        vis_img = img.copy()
        horizontal_lines, vertical_lines = lines
        
        # Draw horizontal lines
        for y in horizontal_lines:
            cv2.line(vis_img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        
        # Draw vertical lines
        for x in vertical_lines:
            cv2.line(vis_img, (x, 0), (x, img.shape[0]), (255, 0, 0), 1)
        
        return vis_img
    


    def _visualize_game_state(self, game_state: np.ndarray) -> None:
        """Visualize the final game state matrix."""
        plt.figure(figsize=(10, 10))
        plt.imshow(game_state, cmap='viridis')
        plt.colorbar(label='Cell State')
        plt.title('Final Game State\nPress any key to continue...')
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                plt.text(j, i, str(game_state[i, j]), 
                        ha='center', va='center', color='white')
        
        plt.grid(True)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

    def _visualize_cell_positions(self, board: np.ndarray, cell_data: Dict[Tuple[int, int], CellData]) -> None:
        """Visualize cell positions on the board image."""
        if not self.debug:
            return
            
        vis_img = board.copy()
        for pos, data in cell_data.items():
            # Draw cell center
            cv2.circle(vis_img, 
                      (data.position.screen_x, data.position.screen_y),
                      3, (0, 255, 0), -1)
            
            # Draw cell boundaries
            cv2.rectangle(vis_img,
                        (data.position.screen_x_range[0], data.position.screen_y_range[0]),
                        (data.position.screen_x_range[1], data.position.screen_y_range[1]),
                        (255, 0, 0), 1)
            
            # Add grid coordinates
            cv2.putText(vis_img,
                       f"({data.position.grid_row},{data.position.grid_col})",
                       (data.position.screen_x - 20, data.position.screen_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.3,
                       (0, 255, 0),
                       1)
        
        self.show_debug("Cell Positions", vis_img)

    def debug_cell(self, cell: np.ndarray, position: tuple, state: int) -> None:
        """Show detailed cell analysis in debug mode."""
        if not self.debug:
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original cell
        ax1.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original")
        
        # Grayscale
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        ax2.imshow(gray, cmap='gray')
        ax2.set_title("Grayscale")
        
        # Threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ax3.imshow(thresh, cmap='gray')
        ax3.set_title("Threshold")
        
        plt.suptitle(f"Cell Analysis - Position: {position}, State: {state}")
        plt.figtext(0.5, 0.01, 'Press any key to continue...', 
                   ha='center', va='bottom')
        
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

    def _detect_number_ocr(self, cell: np.ndarray) -> Optional[int]:
        """Detect number using OCR."""       
        # Convert to grayscale
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 5, 75, 75)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        contrast = clahe.apply(denoised)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(contrast, (3,3), 0)
        
        # Use Otsu's thresholding to get black digits
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components on inverted image to detect black components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - thresh, connectivity=8)
        
        # Create white background
        cleaned = np.ones_like(thresh) * 255
        
        # Add non-border-touching components as black
        for label in range(1, num_labels):
            component = labels == label
            if not (np.any(component[0, :]) or    # top edge
                    np.any(component[-1, :]) or    # bottom edge
                    np.any(component[:, 0]) or     # left edge
                    np.any(component[:, -1])):     # right edge
                cleaned[component] = 0
        
        result = pytesseract.image_to_string(
                cleaned,
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'
            ).strip()
        
        if not result.isdigit():
            result = pytesseract.image_to_string(
                cleaned,
                config='--psm 10 --oem 3'
            ).strip()
        
        if result and result.isdigit() and 1 <= int(result) <= 8:
            return int(result)

        return -4

    def _detect_number_by_color(self, cell: np.ndarray) -> int:
        """Detect number by cell color."""            
        color_to_number = {
            'blue': 1, 'green': 2, 'red': 3, 'dark_blue': 4,
            'brown': 5, 'cyan': 6, 'black': 7, 'gray': 8
        }
            
        color_prob_dict = {color_name: self._is_color_match(cell, color_range) 
                  for color_name, color_range in Colors.NUMBER_COLORS.items()}

        # Get color name with highest probability
        color_name = max(color_prob_dict, key=color_prob_dict.get) if any(prob > 0 for prob in color_prob_dict.values()) else None

        return color_to_number.get(color_name, -4)

    def _is_color_match(self, cell: np.ndarray, color_range: Tuple[List[int], List[int]]) -> float:
        """Check if cell matches a color range."""
        lower = np.array(color_range[0])
        upper = np.array(color_range[1])
        mask = cv2.inRange(cell, lower, upper)
        return np.sum(mask) / (cell.shape[0] * cell.shape[1] * 255)
    

    def update_after_move(self, screen_region: ScreenRegion, last_move: Move, game_state: np.ndarray, cell_data: Dict, remaining_bombs: int) -> dict:
        """
        Update game state after a move by only analyzing affected cells.
        """
        updated_state = game_state.copy()
        
        if last_move.action == 'flag':
            row, col = last_move.row, last_move.col
            updated_state[row, col] = -2
            remaining_bombs -= 1
                
            cell_data[(row, col)] = CellData(
                state=updated_state[row, col],
                position=cell_data[(row, col)].position
            )
            
            return {
                'game_state': updated_state,
                'cell_data': cell_data,
                'remaining_bombs': remaining_bombs,
            }
        
        elif last_move.action == 'click':
            cells_to_check = [(last_move.row, last_move.col)]
            processed_cells = set()  # Track processed cells
            
            while cells_to_check:
                row, col = cells_to_check.pop(0)
                
                # Skip if cell was already processed or out of bounds
                if ((row, col) in processed_cells or 
                    row < 0 or row >= self.grid_size[0] or 
                    col < 0 or col >= self.grid_size[1]):
                    continue
                    
                processed_cells.add((row, col))
                cell_pos = cell_data[(row, col)].position
                
                # Extract cell image from screen
                cell_img = screen_region.image[
                    cell_pos.screen_y_range[0] - screen_region.y:cell_pos.screen_y_range[1] - screen_region.y,
                    cell_pos.screen_x_range[0] - screen_region.x:cell_pos.screen_x_range[1] - screen_region.x
                ]
                
                # Detect cell state
                new_state = self.detect_cell_state(cell_img, (row, col))
                updated_state[row, col] = new_state
                
                cell_data[(row, col)] = CellData(
                    state=new_state,
                    position=cell_pos
                )
                
                # If clicked cell revealed a 0, check neighbors
                if new_state == 0:
                    for neighbor_row, neighbor_col in self.get_neighbors(row, col):
                        if (neighbor_row, neighbor_col) not in processed_cells:
                            cells_to_check.append((neighbor_row, neighbor_col))
            
            return {
                'game_state': updated_state,
                'cell_data': cell_data,
                'remaining_bombs': remaining_bombs,
            }
    

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cell coordinates."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
                    neighbors.append((new_row, new_col))
        return neighbors