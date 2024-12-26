from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from debugger import Debugger

class SevenSegmentOCR:
    """
    Optical Character Recognition (OCR) system for seven-segment displays.
    
    This class implements a specialized OCR system designed to recognize digits
    displayed on seven-segment displays. It processes images containing exactly
    three digits and identifies each digit based on the state of its seven segments.
    
    The system uses a probability-based approach for segment detection and digit
    recognition, making it robust to variations in image quality and segment
    appearance.
    """
    
    def __init__(self, debugger: Optional[Debugger] = None) -> None:
        """
        Initialize the SevenSegmentOCR system.
        
        Args:
            debugger (Optional[Debugger]): Debugger instance for visualization.
                If None, no debug visualizations will be shown.
        """
        self.debugger = debugger
    
    def recognize_digits(self, clean_image: np.ndarray) -> List[str]:
        """
        Recognize all three digits in the input image.
        
        This is the main entry point for the OCR process. It handles the complete
        pipeline from splitting the image into individual digits to recognizing
        each digit and generating debug visualizations if enabled.
        
        Args:
            clean_image (np.ndarray): Pre-processed grayscale image containing
                exactly three seven-segment digits.
        
        Returns:
            List[str]: List of three recognized digits. Unrecognized digits are
                represented as 'X'.
        """
        digit_regions, bounding_boxes = self.split_into_three_digits(clean_image)
        
        recognized_digits = []
        debug_images = []
        
        for i, digit_image in enumerate(digit_regions):
            segments = self.extract_segments_binary(digit_image)
            digit = self.recognize_digit(segments)
            recognized_digits.append(digit if digit is not None else 'X')
            
            if self.debugger:
                debug_img = self.visualize_segments(digit_image, segments)
                debug_images.append(debug_img)
                print(f"Digit {i+1} segments: {segments}")
        
        return recognized_digits
    
    def split_into_three_digits(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Split the input image into three individual digit regions.
        
        Uses a two-pass approach to ensure consistent digit separation:
        1. First pass detects actual digit widths
        2. Second pass extracts digits with equal spacing
        
        Args:
            image (np.ndarray): Input grayscale image containing three digits.
            
        Returns:
            Tuple containing:
                - List[np.ndarray]: Three separate digit images
                - List[Tuple[int, int, int, int]]: Bounding boxes (x, y, w, h)
                  for each digit
        """
        h, w = image.shape
        digit_width = w // 3
        temp_digits = []
        max_width = 0
        
        # First pass: detect digit widths
        for i in range(3):
            start_x = i * digit_width
            end_x = (i + 1) * digit_width
            digit = image[:, start_x:end_x]
            
            rows = np.any(digit == 0, axis=1)
            cols = np.any(digit == 0, axis=0)
            
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                current_width = xmax - xmin + 1
                max_width = max(max_width, current_width)
                temp_digits.append((ymin, ymax, current_width))
            else:
                temp_digits.append((0, h-1, digit_width))
        
        # Second pass: extract with equal spacing
        spacing = (w - max_width * 3) // 4
        digit_regions = []
        bounding_boxes = []
        
        for i in range(3):
            ymin, ymax, _ = temp_digits[i]
            start_x = spacing + i * (max_width + spacing)
            
            digit = image[ymin:ymax+1, start_x:start_x + max_width]
            digit_regions.append(digit)
            bounding_boxes.append((start_x, ymin, max_width, ymax - ymin + 1))
            
        if self.debugger:
            debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.debugger.show_debug("Digit Separation", debug_image)
        
        return digit_regions, bounding_boxes
    
    def extract_segments_binary(self, image: np.ndarray) -> List[float]:
        """
        Extract the state of each segment in a seven-segment digit.
        
        Analyzes seven predefined regions in the image corresponding to the
        segments of a seven-segment display. For each segment, calculates
        the probability that the segment is active based on the density
        of dark pixels in its region.
        
        Args:
            image (np.ndarray): Grayscale image of a single digit.
            
        Returns:
            List[float]: Seven probability values (0-1) indicating the state
                of each segment (top, top-right, bottom-right, bottom,
                bottom-left, top-left, middle).
        """
        h, w = image.shape
        
        regions = [
            (slice(h//16, 3*h//16), slice(w//4, 3*w//4)),      # Top
            (slice(1*h//16, 7*h//16), slice(5*w//8, 7*w//8)),  # Top-right
            (slice(9*h//16, 15*h//16), slice(5*w//8, 7*w//8)), # Bottom-right
            (slice(13*h//16, 15*h//16), slice(w//4, 3*w//4)),  # Bottom
            (slice(9*h//16, 15*h//16), slice(w//8, 3*w//8)),   # Bottom-left
            (slice(1*h//16, 7*h//16), slice(w//8, 3*w//8)),    # Top-left
            (slice(7*h//16, 9*h//16), slice(w//4, 3*w//4))     # Middle
        ]
        
        segments = [min((np.sum(image[r_slice, c_slice] == 0) / 
                    image[r_slice, c_slice].size) / 0.2, 1) 
                for r_slice, c_slice in regions]
        
        if self.debugger:
            debug_img = self.visualize_segments(image, segments)
            self.debugger.show_debug("Segment Detection", debug_img)
        
        return segments
    
    def recognize_digit(self, segments: List[float]) -> Optional[int]:
        """
        Match segment probabilities to digit templates.
        
        Uses a probabilistic approach to match the detected segment pattern
        against templates for digits 0-9. Calculates a confidence score
        for each possible digit and returns the best match if it exceeds
        a minimum confidence threshold.
        
        Args:
            segments (List[float]): Seven probability values indicating
                segment states.
            
        Returns:
            Optional[int]: The recognized digit (0-9) or None if no
                confident match is found.
        """
        templates: Dict[int, List[int]] = {
            0: [1,1,1,1,1,1,0],
            1: [0,1,1,0,0,0,0],
            2: [1,1,0,1,1,0,1],
            3: [1,1,1,1,0,0,1],
            4: [0,1,1,0,0,1,1],
            5: [1,0,1,1,0,1,1],
            6: [1,0,1,1,1,1,1],
            7: [1,1,1,0,0,0,0],
            8: [1,1,1,1,1,1,1],
            9: [1,1,1,1,0,1,1]
        }
        
        best_match = None
        best_score = float('-inf')
        
        for digit, template in templates.items():
            score = sum((segment * template_val) + 
                       ((1 - segment) * (1 - template_val))
                       for segment, template_val in zip(segments, template))
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        threshold = 7 * 0.8  # 80% confidence threshold
        return best_match if best_score >= threshold else None
    
    def visualize_segments(self, image: np.ndarray, segments: List[float]) -> np.ndarray:
        """
        Create a visual representation of detected segments.
        
        Generates a color-coded visualization where each segment is highlighted
        in green if detected as active or red if inactive. The visualization
        is overlaid on the original digit image.
        
        Args:
            image (np.ndarray): Original grayscale digit image.
            segments (List[float]): Seven segment probability values.
            
        Returns:
            np.ndarray: Color visualization image.
        """
        h, w = image.shape
        debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        regions = [
            np.array([[w//4, h//16], [3*w//4, h//16], 
                     [3*w//4, 3*h//16], [w//4, 3*h//16]]),  # Top
            np.array([[5*w//8, 2*h//16], [7*w//8, 2*h//16],
                     [7*w//8, 7*h//16], [5*w//8, 7*h//16]]),  # Top-right
            np.array([[5*w//8, 9*h//16], [7*w//8, 9*h//16],
                     [7*w//8, 14*h//16], [5*w//8, 14*h//16]]),  # Bottom-right
            np.array([[w//4, 13*h//16], [3*w//4, 13*h//16],
                     [3*w//4, 15*h//16], [w//4, 15*h//16]]),  # Bottom
            np.array([[w//8, 9*h//16], [3*w//8, 9*h//16],
                     [3*w//8, 14*h//16], [w//8, 14*h//16]]),  # Bottom-left
            np.array([[w//8, 2*h//16], [3*w//8, 2*h//16],
                     [3*w//8, 7*h//16], [w//8, 7*h//16]]),  # Top-left
            np.array([[w//4, 7*h//16], [3*w//4, 7*h//16],
                     [3*w//4, 9*h//16], [w//4, 9*h//16]])  # Middle
        ]
        
        for segment, region in zip(segments, regions):
            color = (0, 255, 0) if segment == 1 else (0, 0, 255)
            cv2.fillPoly(debug_image, [region], color)
        
        cv2.addWeighted(debug_image, 0.3, 
                       cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, 
                       0, debug_image)
        
        return debug_image