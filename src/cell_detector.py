import cv2
import numpy as np
import pytesseract
from typing import Optional, List, Tuple
from config import Colors, CELL_STATE
from debugger import Debugger
import os
import logging

class CellStateDetector:
    """Class responsible for detecting the state of individual Minesweeper cells."""
    
    def __init__(self, tesseract_path: Optional[str] = None, debugger: Optional[Debugger] = None):
        self.debugger = debugger
        self.ocr_available = False
        
        if tesseract_path:
            try:
                # Check if the path exists
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    # Test if Tesseract works by trying to read a simple image
                    test_img = np.zeros((50, 50, 3), dtype=np.uint8)
                    try:
                        pytesseract.image_to_string(test_img)
                        self.ocr_available = True
                        logging.info("Tesseract successfully initialized and available for OCR")
                    except Exception as e:
                        logging.warning(f"Tesseract installation found but failed to initialize: {str(e)}")
                else:
                    logging.warning(f"Specified Tesseract path does not exist: {tesseract_path}")
            except Exception as e:
                logging.warning(f"Failed to set Tesseract path: {str(e)}")
        else:
            logging.info("No Tesseract path provided - OCR detection will be unavailable")

    def detect_state(self, cell: np.ndarray, position: tuple) -> int:
        """Detect the state of a single cell."""
        state = CELL_STATE.undetected
        if self._is_color_match(cell, Colors.CELL_COLORS['gray']) > 0.85:
            state = CELL_STATE.empty
        elif self._is_color_match(cell, Colors.CELL_COLORS['red']) > 0.4 and self._is_color_match(cell, Colors.CELL_COLORS['black']) > 0.2:
            state = CELL_STATE.mine
        elif self._is_color_match(cell, Colors.CELL_COLORS['red']) > 0.01 and self._is_color_match(cell, Colors.CELL_COLORS['black']) > 0.05:
            state = CELL_STATE.flag
        elif self._is_color_match(cell, Colors.CELL_COLORS['gray']) > 0.4 and self._is_color_match(cell, Colors.CELL_COLORS['white']) > 0.05:
            state = CELL_STATE.unopened
        else:
            number = self._detect_number_by_color(cell)
            if number == CELL_STATE.undetected and self.ocr_available:
                state = self._detect_number_ocr(cell)
            else:
                state = number
        
        if self.debugger is not None:
            self.debugger.visualize_cell_analysis(cell, position, state)
        
        return state

    def _is_color_match(self, cell: np.ndarray, color_range: Tuple[List[int], List[int]]) -> float:
        """Check if cell matches a color range."""
        lower = np.array(color_range[0])
        upper = np.array(color_range[1])
        mask = cv2.inRange(cell, lower, upper)
        return np.sum(mask) / (cell.shape[0] * cell.shape[1] * 255)

    def _detect_number_by_color(self, cell: np.ndarray) -> Optional[int]:
        """Detect number by cell color."""            
        color_to_number = {
            'blue': 1, 'green': 2, 'red': 3, 'dark_blue': 4,
            'brown': 5, 'cyan': 6, 'black': 7, 'gray': 8
        }
            
        color_prob_dict = {color_name: self._is_color_match(cell, color_range) 
                          for color_name, color_range in Colors.NUMBER_COLORS.items()}

        color_name = max(color_prob_dict, key=color_prob_dict.get) if any(prob > 0 for prob in color_prob_dict.values()) else None
        return color_to_number.get(color_name, CELL_STATE.undetected)

    def _detect_number_ocr(self, cell: np.ndarray) -> int:
        """Detect number using OCR if available."""
        if not self.ocr_available:
            return CELL_STATE.undetected
            
        try:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            contrast = clahe.apply(denoised)
            
            blurred = cv2.GaussianBlur(contrast, (3,3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - thresh, connectivity=8)
            
            cleaned = np.ones_like(thresh) * 255
            for label in range(1, num_labels):
                component = labels == label
                if not (np.any(component[0, :]) or np.any(component[-1, :]) or 
                       np.any(component[:, 0]) or np.any(component[:, -1])):
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
            
            return int(result) if result and result.isdigit() and 1 <= int(result) <= 8 else CELL_STATE.undetected
            
        except Exception as e:
            logging.error(f"OCR detection failed: {str(e)}")
            return CELL_STATE.undetected