# Code Walkthrough & Line-by-Line Explanation: `engine.py`

This document provides a comprehensive, line-by-line guide to `engine.py`. It explains the algorithmic theory, mathematical justification, and exact code mechanics behind the search engine.

---

## 1. High-Level Role of `engine.py`

`engine.py` is the intelligence center of the project. Given any board position, its task is to compute the single best move within human response time constraints ($< 2$ seconds).

It accomplishes this via four tightly coupled subsystems:
1. **Hybrid Board Evaluator (`evaluate_position`)**: Blends static material heuristics with deep neural network assessments.
2. **Tactical Move Sorter (`order_moves` & `score_move`)**: Employs MVV-LVA (Most Valuable Victim - Least Valuable Attacker) to prioritize high-leverage moves.
3. **Quiescence Search Engine (`quiescence_search`)**: Searches captures beyond the horizon to prevent tactical blindness.
4. **Negamax $\alpha$-$\beta$ Tree Searcher (`negamax` & `get_best_move`)**: Prunes subtrees using Alpha-Beta cutoffs.

---

## 2. Detailed Method Breakdown

### 2.1 Initialization: `ChessEngine.__init__` (Lines 8–24)
```python
class ChessEngine:
    def __init__(self, model_path="chess_model.pth", default_depth=3, use_neural=True, device="cpu"):
        self.default_depth = default_depth
        self.use_neural = use_neural
        self.device = device
        self.nodes_visited = 0
        self.model = load_model(model_path, device=self.device) if self.use_neural else None
```
- **`default_depth`**: The search horizon in plies (half-moves). Depth 3 searches: (AI move $\to$ Player reply $\to$ AI countermove).
- **`nodes_visited`**: A diagnostics counter tracking how many board states were evaluated. Allows calculating Nodes Per Second (NPS).
- **`self.model`**: Loads the trained PyTorch `ChessNet` model. If no model file exists, it gracefully initializes without breaking.

---

### 2.2 Dynamic Evaluation: `evaluate_position` (Lines 26–57)
```python
def evaluate_position(self, board: chess.Board) -> float:
```
In traditional Minimax, evaluation is computed from White's perspective ($+10$ means White is ahead).
In **Negamax**, the evaluation is always returned from the perspective of the **active player whose turn it is to move**:
- If it is White's turn, a positive score means White has an advantage.
- If it is Black's turn, a positive score means Black has an advantage.

```python
    if board.is_checkmate():
        return -10000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0.0
```
- Terminal checkmate evaluation: If it is the current player's turn and they have no legal moves and are in check, they have lost $\implies -10000.0$.
- Draws evaluate to exactly $0.0$.

```python
    heuristic_score = compute_heuristic_score(board)
    if self.use_neural and self.model is not None:
        nn_score = evaluate_board(board, self.model, device=self.device)
        combined_white_eval = 0.65 * heuristic_score + 0.35 * (nn_score * 8.0)
    else:
        combined_white_eval = heuristic_score

    perspective = 1.0 if board.turn == chess.WHITE else -1.0
    return combined_white_eval * perspective
```
- Blends classical material/positional evaluation ($65\%$) with neural network evaluation ($35\%$).
- Multiplying by `perspective` ($+1$ for White, $-1$ for Black) fulfills the Negamax contract.

---

### 2.3 Move Ordering: `score_move` & `order_moves` (Lines 59–97)
```python
def score_move(self, board: chess.Board, move: chess.Move) -> int:
```
**Why Move Ordering is the Single Most Important Optimization in Game Trees**:
If the best move is examined first, all other alternatives can be discarded immediately via a Beta cutoff ($\alpha \ge \beta$).
We compute a tactical priority score for each candidate move:

1. **Pawn Promotions**:
   ```python
   if move.promotion:
       score += 900
   ```
   Promoting a pawn to a Queen is game-deciding.

2. **Captures (MVV-LVA)**:
   ```python
   if board.is_capture(move):
       victim = board.piece_at(move.to_square)
       attacker = board.piece_at(move.from_square)
       victim_val = int(MATERIAL_WEIGHTS.get(victim.piece_type, 1.0) * 100)
       attacker_val = int(MATERIAL_WEIGHTS.get(attacker.piece_type, 1.0) * 10)
       score += 1000 + (victim_val - attacker_val)
   ```
   Capturing a Queen ($9.0$) with a Pawn ($1.0$) gives:
   $$1000 + (900 - 10) = 1890 \text{ points}$$
   Capturing a Pawn ($1.0$) with a Queen ($9.0$) gives:
   $$1000 + (100 - 90) = 1010 \text{ points}$$
   High-value captures with low-value attackers always sort first!

3. **Checks**:
   ```python
   board.push(move)
   if board.is_check():
       score += 200
   board.pop()
   ```
   Forcing moves that check the enemy King restrict their counterplay and sort ahead of quiet moves.

---

### 2.4 Quiescence Search: `quiescence_search` (Lines 99–133)
```python
def quiescence_search(self, board: chess.Board, alpha: float, beta: float, max_depth: int = 4) -> float:
```
- **The Horizon Problem**: If search stops at depth 3 immediately after White captures Black's bishop, White thinks it gained a bishop. But if Black's queen can capture White's queen on move 4, White walked into a trap.
- **Stand-Pat Score**:
  `stand_pat = self.evaluate_position(board)`
  A player always has the option to "stand pat" (decline to make risky speculative captures if the static position is already sufficient).
  - If `stand_pat >= beta`: Immediate beta cutoff.
  - If `alpha < stand_pat`: Raise the alpha floor.
- Recursively examines only **legal captures** (`board.is_capture(move)`) until the position is calm and stable.

---

### 2.5 Negamax Search: `negamax` (Lines 135–182)
```python
def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> float:
```
The heart of the search algorithm:

```python
if board.is_checkmate():
    return -10000.0 + ply
```
- **The Ply Adjustment**:
  Notice `+ ply`! If the engine finds a checkmate in 1 move ($\text{ply} = 1$), the score is $-10000 + 1 = -9999$.
  If it finds a checkmate in 5 moves ($\text{ply} = 5$), the score is $-10000 + 5 = -9995$.
  Because the opponent negates the score, the engine will strictly prefer **shorter checkmates** over delayed checkmates!

```python
for move in ordered_moves:
    board.push(move)
    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
    board.pop()

    if score >= beta:
        return beta  # Beta Cutoff (Prune!)

    if score > alpha:
        alpha = score
return alpha
```
- Note the recursive parameters: `alpha` and `beta` are swapped and negated: `(-beta, -alpha)`.
- If `score >= beta`, the branch is immediately pruned (`return beta`), skipping dozens or hundreds of remaining legal moves!

---

### 2.6 Root Decision: `get_best_move` (Lines 184–231)
```python
def get_best_move(self, board: chess.Board, depth: Optional[int] = None) -> Tuple[Optional[chess.Move], float]:
```
- Invoked by the GUI or user.
- If only 1 legal move exists (e.g. forced king evasion), it returns immediately without wasting compute.
- Resets `nodes_visited = 0` and records start time.
- Loops through ordered root moves, evaluates them via `negamax`, tracks the highest scoring move `best_move`, and logs diagnostic search performance (depth, best move, eval score, nodes visited, and NPS).
