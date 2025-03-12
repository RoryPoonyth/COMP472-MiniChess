# Mini Chess Project

This repository contains the implementation of the **Mini Chess** game as part of the COMP 472 Winter 2025 project. The project focuses on implementing adversarial search algorithms, including Minimax and Alpha-Beta pruning, as well as developing and evaluating heuristics for the game.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [How to Run](#how-to-run)
4. [Input and Output](#input-and-output)
5. [Game Rules](#game-rules)
6. [Heuristics](#heuristics)
7. [Evaluation Criteria](#evaluation-criteria)
8. [Deliverables](#deliverables)

---

## Project Overview
Mini Chess is a simplified 2-player chess game played on a 5x5 board. The goal is to implement AI strategies to play the game effectively using adversarial search techniques. The main objectives include:

- Implementing Minimax and Alpha-Beta pruning.
- Developing and testing custom heuristics.
- Supporting multiple play modes (human vs AI, AI vs AI, etc.).

The game starts with a predefined board configuration and allows for turns where players can move their pieces according to specified rules.

---

## Features
- **Game Modes:**
  - Human vs Human (H-H)
  - Human vs AI (H-AI, AI-H)
  - AI vs AI (AI-AI)
- **Adversarial Search Algorithms:**
  - Minimax
  - Alpha-Beta Pruning
- **Heuristics:**
  - Baseline heuristic (e0)
  - Two custom heuristics (e1 and e2)
- **Trace Generation:**
  - Output files recording game parameters, actions, and statistics.
- **Command-Line Interface (CLI):**
  - Easy-to-use, text-based input and output.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Mini-Chess-Project
   ```
2. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the game:
   ```bash
   python main.py
   ```
4. Follow the on-screen prompts to configure game parameters and play the game.

---

## Input and Output
### Input
- **Game Parameters:**
  - Maximum allowed time for AI moves.
  - Maximum number of turns.
  - Play mode (H-H, H-AI, AI-AI).
  - Algorithm (Minimax or Alpha-Beta).
- **Move Coordinates:**
  - Specify moves using source and target coordinates (e.g., `B2 B3`).

### Output
- Trace files (e.g., `gameTrace-true-5-100.txt`) that include:
  - Game parameters.
  - Initial and updated board configurations.
  - Move details (e.g., turn number, heuristic scores, time taken).
  - Cumulative statistics (e.g., states explored, branching factor).
  - Winner and total turns.

---

## Game Rules
- **Objective:** Capture the opponent's king to win.
- **Board:** 5x5 grid.
- **Pieces:**
  - King, Queen, Bishop, Knight, and two Pawns per player.
- **Movement:**
  - King: 1 square in any direction.
  - Queen: Any number of squares in any direction.
  - Bishop: Diagonal movements.
  - Knight: L-shaped movements.
  - Pawn: 1 square forward; diagonal captures; with Queen promotion when reaching the other end.
- **End Conditions:**
  - Win: Opponent's king captured.
  - Draw: No pieces captured for 10 consecutive turns.

---

## Heuristics
1. **Baseline Heuristic (e0):**
   ```
   e0 = (#wp + 3·#wB + 3·#wN + 9·#wQ + 999·wK)
        - (#bp + 3·#bB + 3·#bN + 9·#bQ + 999·bK)
   ```
2. **Custom Heuristics (e1, e2):**
   - Designed to optimize piece valuation, board positioning, and attack/defense strategies.

---

## Evaluation Criteria
- **Code Functionality:**
  - Correct implementation of game rules, algorithms, and heuristics.
- **Programming Quality:**
  - Code structure, readability, and efficiency.
- **Output Quality:**
  - Accuracy and completeness of trace files.
- **Presentation:**
  - Clarity during demos and effective team collaboration.

---

## Deliverables
1. **D1 (February 13, 2025):**
   - Manual game mode (H-H).
   - Initial implementation without AI.
2. **D2 (March 13, 2025):**
   - AI with Minimax and Alpha-Beta.
   - Heuristics (e0, e1, e2).
3. **D3 (April 3, 2025):**
   - Final report and analysis.
   - Complete functionality with output trace files.

---

## Authors
- AI Gambit
  - Rory Poonyth [Team Leader] | 40226938
  - Omar Eduardo Sanchez Lopez | 40027467
  - Mansi Gairola | 40107694
