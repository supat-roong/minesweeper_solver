import numpy as np
import cv2
import matplotlib.pyplot as plt

class SevenSegmentOCR:
    def __init__(self, debug=False):
        self.debug = debug
        
    def show_debug(self, title, image, cmap=None):
        if self.debug:
            plt.figure(figsize=(8, 4))
            plt.title(title)
            plt.imshow(image, cmap=cmap)
            plt.axis('off')
            plt.show()
    
    def recognize_digits(self, clean_image):
        """Main OCR function to recognize all three digits"""
        # Split image into three digits
        digit_regions, bounding_boxes = self.split_into_three_digits(clean_image)
        
        # Process each digit
        recognized_digits = []
        debug_images = []
        
        for i, digit_image in enumerate(digit_regions):
            # Extract and analyze segments
            segments = self.extract_segments_binary(digit_image)
            digit = self.recognize_digit(segments)
            recognized_digits.append(digit if digit is not None else 'X')
            
            if self.debug:
                debug_img = self.visualize_segments(digit_image, segments)
                debug_images.append(debug_img)
                print(f"Digit {i+1} segments: {segments}")
        
        if self.debug:
            self.show_debug_visualization(clean_image, digit_regions, 
                                       bounding_boxes, debug_images, 
                                       recognized_digits)
        
        return recognized_digits
    
    def split_into_three_digits(self, image):
        """Split the image into exactly three digit regions"""
        h, w = image.shape
        digit_width = w // 3
        digit_regions = []
        bounding_boxes = []
        temp_digits = []
        max_width = 0
        
        # First pass: detect digit widths
        for i in range(3):
            start_x = i * digit_width
            end_x = (i + 1) * digit_width
            digit = image[:, start_x:end_x]
            
            # Find digit boundaries
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
        
        for i in range(3):
            ymin, ymax, _ = temp_digits[i]
            start_x = spacing + i * (max_width + spacing)
            
            digit = image[ymin:ymax+1, start_x:start_x + max_width]
            digit_regions.append(digit)
            bounding_boxes.append((start_x, ymin, max_width, ymax - ymin + 1))
        
        return digit_regions, bounding_boxes
    
    def extract_segments_binary(self, image):
        """Extract 7-segment states"""
        h, w = image.shape
        
        # Define segment regions
        regions = [
            (slice(h//16, 3*h//16), slice(w//4, 3*w//4)),      # Top
            (slice(2*h//16, 7*h//16), slice(5*w//8, 7*w//8)),  # Top-right
            (slice(9*h//16, 14*h//16), slice(5*w//8, 7*w//8)), # Bottom-right
            (slice(13*h//16, 15*h//16), slice(w//4, 3*w//4)),  # Bottom
            (slice(9*h//16, 14*h//16), slice(w//8, 3*w//8)),   # Bottom-left
            (slice(2*h//16, 7*h//16), slice(w//8, 3*w//8)),    # Top-left
            (slice(7*h//16, 9*h//16), slice(w//4, 3*w//4))     # Middle
        ]
        
        return [min((np.sum(image[r_slice, c_slice] == 0) / image[r_slice, c_slice].size)/0.2, 1) for r_slice, c_slice in regions]
    
    def recognize_digit(self, segments):
        """
        Match probability-based segments pattern to digit templates.
        
        Args:
            segments: List of 7 float values between 0 and 1 representing segment probabilities
            
        Returns:
            int or None: The best matching digit (0-9) or None if no good match found
        """
        templates = {
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
            # Calculate score using log probability
            score = sum(
                (segment * template_val) + ((1 - segment) * (1 - template_val))
                for segment, template_val in zip(segments, template)
            )
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        # Threshold for accepting a match
        # The maximum possible score is 7 (all segments match perfectly)
        # We accept matches that are at least 80% confident overall
        threshold = 7 * 0.8
        return best_match if best_score >= threshold else None
    
    def visualize_segments(self, image, segments):
        """Create debug visualization of detected segments"""
        h, w = image.shape
        debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Define segment polygons
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
    
    def show_debug_visualization(self, original_image, digit_regions, 
                               bounding_boxes, debug_images, results):
        """Show comprehensive debug visualization"""
        if not self.debug:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Show digit separation
        debug_separation = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR)
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(debug_separation, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        plt.subplot(141)
        plt.title('Digit Separation')
        plt.imshow(cv2.cvtColor(debug_separation, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        
        # Show individual digit analysis
        for i, debug_digit in enumerate(debug_images):
            plt.subplot(1, 4, i+2)
            plt.title(f'Digit {i+1}: {results[i]}')
            plt.imshow(cv2.cvtColor(debug_digit, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()