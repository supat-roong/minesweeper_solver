# Minesweeper Solver

An intelligent automated solver that plays Minesweeper using computer vision and probability-based decision-making.

## Features

- **Computer Vision Detection**: Uses OpenCV and Tesseract OCR to detect and analyze the Minesweeper game board
- **Seven-Segment Display OCR**: Custom OCR system for reading the mine counter display
- **Intelligent Decision Making**: Employs probability-based algorithms to make optimal moves
- **Real-time Game Analysis**: Continuously monitors the game state and adapts its strategy
- **Automated Player**: Fully automates gameplay by integrating screen capture, decision-making, and mouse control for seamless interaction with the Minesweeper game.
- **Real-time Visualization**: Displays dynamic updates of the solver's progress, including grid detection, cell states, and probability calculations, for an interactive experience.
- **Visual Debugging Tools**: Comprehensive visualization options for development and debugging

## Gameplay Examples

### 1. Full Solver with Visualizer

Watch the solver in action with debug visualization enabled. The solver displays:

- Detected grid lines
- Cell states and numbers
- Mine probability heatmaps

**Outcome**: The solver successfully completes the game, showcasing its probability-based decision-making and constraint satisfaction algorithms.

### 2. Solver Without Visualizer (Intermediate & Expert Modes)

Run the solver in a "headless" mode for a clean, visualizer-free experience:

Intermediate mode:

Expert mode:

**Outcome**: The solver demonstrates high efficiency and accuracy, finishing both modes without user intervention.

### 3. Example of Failure (Due to Luck-Based Guessing)

Unfortunately, some scenarios in Minesweeper are unsolvable without guessing. For example:


**Outcome**: The solver clicks the wrong cell, triggering a mine. This showcases the inherent challenge of luck-based situations in Minesweeper.

## Requirements

- Python 3.7+

## Installation

1. Clone the repository:

```bash
git clone https://github.com/supat-roong/minesweeper_solver.git
cd minesweeper_solver
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (optional for better gameboard OCR):

   - Windows: Download and install from [GitHub Tesseract Releases](https://github.com/tesseract-ocr/tesseract.git)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

1. Open your favorite Minesweeper game.
2. Run the solver:

```bash
python main.py
```

3. Use the selection tool to select the game board region.
4. The program will automatically solve the game!


## Configuration

Edit `config.py` to customize:

- Minesweeper grid size
- Visualization options
- Tesseract OCR path
- Other solver parameters

## How It Works

1. **Board Detection**:

   - Captures the screen region containing the game
   - Uses computer vision to detect grid lines and cells
   - Recognizes cell states (unopened, numbers, flags, mines)

2. **Solver Logic**:

   - Maintains a game state matrix
   - Calculates mine probabilities using neighbor constraints
   - Uses three-pass probability refinement:
     - Initial probability assignment
     - Direct neighbor constraints
     - Overlapping constraint analysis

3. **Move Execution**:

   - Prioritizes definite safe moves
   - Falls back to lowest probability cells when no safe moves exist
   - Executes moves through mouse control

## Debug and Real-time Visualization

The program includes two types of visualization options:

- **Real-time Visualization**: Displays dynamic updates of the solver's progress, including grid detection, cell states, and probability calculations.
- **Debug Visualization**: Provides detailed analysis tools such as grid line detection, cell state recognition, mine probability heatmaps, and seven-segment OCR analysis.

**Note**: Only one visualization mode can be activated at a time due to main thread limitations. Enable your desired mode in `config.py`.

## Project Structure
The project is organized as follows:

```
minesweeper_solver/
├── media/                        # Contains media assets (e.g., gameplay videos)
├── src/                          # Source code directory
│   ├── board_detector.py         # Detects the Minesweeper game board using computer vision
│   ├── bomb_counter.py           # Counts the number of total mines using visual recognition
│   ├── cell_detector.py          # Identifies cell states (e.g., unopened, flagged, numbered)
│   ├── config.py                 # Configuration file for solver parameters and visualization options
│   ├── custom_dataclass.py       # Defines custom data structures used throughout the project
│   ├── debugger.py               # Provides debugging tools for grid detection and analysis
│   ├── grid_detector.py          # Detects the grid layout of the Minesweeper board
│   ├── minesweeper_detector.py   # Core module for integrating board detection logic
│   ├── player.py                 # Handles automated gameplay (mouse control and decision execution)
│   ├── screenshot.py             # Captures the screen region containing the Minesweeper game
│   ├── seven_segment_ocr.py      # Recognizes numbers in the seven-segment display
│   ├── solver.py                 # Implements the probability-based Minesweeper solving algorithm
│   └── visualizer.py             # Handles real-time visualization of grid detection and probability calculations   
└── main.py                   # Main entry point for running the solver
```   

## License

This project is licensed under the MIT License - see the LICENSE(LICENSE) file for details.

