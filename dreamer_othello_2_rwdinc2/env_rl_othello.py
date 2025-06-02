import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
import random


class OthelloSimulator:
    # // Game constants
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    BLOCKED = 3
    BOARD_SIZE = 8

    # // [KEEP ALL STANDARD GAME FUNCTIONS]
    # // initializeBoard, updateBoardDisplay, countDiscs, isValidMove, getValidMoves, makeMove...

    # // Initialize the game board
    def __init__(self, display_board=False, display_log=False, display_scores=False, verbose=True):

        self.display_board=display_board
        self.display_log=display_log
        self.display_scores=display_scores
        self.verbose=verbose

        # // Create empty board
        self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]

        # // Game state
        self.is_valid = False
        self.currentPlayer = self.BLACK
        self.gameRunning = False
        self.moveLog = []
        self.winner = None
        
        # // Place initial pieces
        self.board[3][3] = self.WHITE
        self.board[3][4] = self.BLACK
        self.board[4][3] = self.BLACK
        self.board[4][4] = self.WHITE
        
        # // Update the display
        self.scores = self.countDiscs()

    # // Count discs of each color
    def countDiscs(self):
        black = 0
        white = 0
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if (self.board[row][col] == self.BLACK):
                    black += 1
                elif (self.board[row][col] == self.WHITE):
                    white += 1
        
        return [black, white]


    # // Check if a move is valid
    def isValidMove(self, row, col, player):
        # // Must be an empty cell
        if (self.board[row][col] != self.EMPTY):
            return False
        
        opponent = self.BLACK if player == self.WHITE else self.WHITE
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ]
        
        # // Check in each direction
        for dr, dc in directions:
            r = row + dr
            c = col + dc
            foundOpponent = False
            
            # Follow line of opponent pieces
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                foundOpponent = True
                r += dr
                c += dc
            
            # If line ends with our piece, it's a valid move
            if foundOpponent and 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == player:
                return True
            
        return False

    # // Get all valid moves for a player
    def getValidMoves(self, player):
        moves = []
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.isValidMove(row, col, player):
                    moves.append({'row': row, 'col': col})
        
        return moves

    # // Make a move
    def makeMove(self, row, col, player):
        # // Place the piece
        self.board[row][col] = player
        
        # // Log the move
        playerName = "Black" if player == self.BLACK else "White"
        colLetter = chr(97 + col)  # 'a' through 'h'
        rowNumber = row + 1  # 1 through 8
        moveText = f"{colLetter}{rowNumber}"
        self.moveLog.append(moveText)
        
        # // Flip opponent pieces
        opponent = self.BLACK if player == self.WHITE else self.WHITE
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ]
        
        for dr, dc in directions:
            r = row + dr
            c = col + dc
            piecesToFlip = []
            
            # Collect opponent pieces in this direction
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                piecesToFlip.append([r, c])
                r += dr
                c += dc
            
            # If line ends with our piece, flip all collected pieces
            if piecesToFlip and 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == player:
                for fr, fc in piecesToFlip:
                    self.board[fr][fc] = player

    # // Modify makeAIMove function to handle custom strategies from localStorage
    def makeAIMove(self, strategy_black, strategy_white):
        if not self.gameRunning:
            return
        
        # try:
        validmoves = self.getValidMoves(self.currentPlayer)
        # // Get move
        if self.currentPlayer == self.BLACK:
            move = strategy_black.move(validmoves, self.moveLog, self.currentPlayer, self.board)
        elif self.currentPlayer == self.WHITE:
            move = strategy_white.move(validmoves, self.moveLog, self.currentPlayer, self.board)
        
        if not move:
            # No valid moves, check if game is over
            opponent = self.BLACK if self.currentPlayer == self.WHITE else self.WHITE
            opponentMoves = self.getValidMoves(opponent)

            if len(opponentMoves) == 0:
                if self.verbose:
                    print("no moves for", "Black" if opponent == self.BLACK else "White")
                # Game over
                self.endGame()
                return

            # Pass turn to opponent
            playerName = "Black" if self.currentPlayer == self.BLACK else "White"
            self.moveLog.append(f"")

            self.currentPlayer = opponent
            self.updateStatus()

            # Schedule next AI move
            self.makeAIMove(strategy_black, strategy_white)
            return
        
        # Make the move
        self.makeMove(move['row'], move['col'], self.currentPlayer)

        # Switch players
        self.currentPlayer = self.BLACK if self.currentPlayer == self.WHITE else self.WHITE
        self.updateStatus()

        if self.display_board:
            for i in self.board:
                print(i)
        if self.display_scores and self.display_log:
            print([self.scores, self.moveLog])
        else:
            if self.display_scores:
                print(self.scores)
            if self.display_log:
                print(self.moveLog)
        

        # Schedule next AI move
        self.makeAIMove(strategy_black, strategy_white)

        # except Exception as error:
        #     print(f"Error in AI move: {str(error)}")
        #     self.endGame()

    # // [KEEP OTHER GAME FUNCTIONS]
    # // updateStatus, endGame, startGame, resetGame
    # // Update game status
    def updateStatus(self):
        self.scores = self.countDiscs()

    # // End the game
    def endGame(self):
        self.gameRunning = False
        self.updateStatus()
        if self.scores[0] > self.scores[1]:
            self.winner = self.BLACK
        elif self.scores[0] < self.scores[1]:
            self.winner = self.WHITE
        else:
            self.winner = self.EMPTY  # Draw
        if self.verbose:
            print("Game End: ", self.scores)

    # // Start a new game
    def startGame(self, strategy_black, strategy_white):
        # // Always reinitialize the board before starting a new game
        self.__init__(self.display_board, self.display_log, self.display_scores, self.verbose)
        
        self.gameRunning = True
        self.currentPlayer = self.BLACK
        self.moveLog = []

        self.updateStatus()
        
        # // Start AI moves
        self.makeAIMove(strategy_black, strategy_white)

    def getBoardRecord(self, playlog, is_index=False, display_result=False):
        self.__init__(self.display_board, self.display_log, self.display_scores, self.verbose)

        self.gameRunning = True
        self.currentPlayer = self.BLACK
        self.moveLog = []

        self.playRecord(playlog, is_index, display_result)
    
    def playRecord(self, playlog, is_index, display_result):
        is_valid = False

        if not self.gameRunning:
            return

        for move in playlog:
            if is_index:
                pos = [move // 8, move % 8]
            else:
                pos = self.move_to_pos(move)
            is_valid |= not self.isValidMove(pos[0], pos[1], self.currentPlayer)

            self.makeMove(pos[0], pos[1], self.currentPlayer)
            # 건너뛰기는 고려하지 않음
            self.updateStatus()
            
            self.currentPlayer = self.BLACK if self.currentPlayer == self.WHITE else self.WHITE

            if self.display_board:
                for i in self.board:
                    print(i)
            if self.display_scores and self.display_log:
                print([self.scores, self.moveLog])
            else:
                if self.display_scores:
                    print(self.scores)
                if self.display_log:
                    print(self.moveLog)
        
        self.is_valid = not is_valid
        if display_result:
            print("moves are all valid:", self.is_valid)
        
        self.endGame()

    def move_to_pos(self, move):
        # col is letter, row is number
        col, row = move[0], move[1]
        return [int(row) - 1, ord(col) - ord('a')]



# TODO
# othello 시뮬레이터에 param 넣어서 보드 사이즈나 blocked 상태 같은 거 바꿔서 나도 가변 환경 테스트 할 수 있어야 함 (사실 신규 othello 코드 다시 가져와도 될 듯)

# TODO
# 이 환경은 사전 학습만을 위한 것
# 실제로 웹에서 데이터 받아서 fine-tuning 하려면 룰이 바뀔 수 있기 때문에 이런 파이썬 환경이랑 상호작용시키기는 힘들고, 받은 데이터를 '학습에서 사용하는 form으로' 가공해서 던져주는 함수가 따로 필요
# 가공해서 던지는 거는 그냥 world model 학습 form(obs, reward, terminated 등)으로 모아서 던져주면 될 듯? buffer를 js로 구현해서 던져줘야 하려나...
# 검정과 하양이 대국하며 각각 학습하는 형태로 구현되면 좋음

class DreamerOthelloEnv:
    def __init__(self, player=1):
        self.sim = OthelloSimulator(display_board=False, display_log=False, verbose=False)
        self.weight_mobility_mine = 4
        self.weight_mobility_opponent = 2
        self.weight_frontier = -2
        self.penalty_nonvalid = -15

        self.agent = player

        self.reset()

    def reset(self):
        self.sim.__init__()  # 보드 초기화

        # TODO 시뮬레이터 내부 함수로 직접 초기화하고 나서도 아래처럼 뭔가를 더 만져줘야 함?
        self.sim.gameRunning = True
        self.sim.currentPlayer = self.sim.BLACK
        self.valid_moves = self.sim.getValidMoves(self.sim.currentPlayer)
        self.board = self.get_board_tensor()
        self.done = False

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.board.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(self.board.shape[-1] * self.board.shape[-2])

        return self.board, self.sim.currentPlayer

    def get_board_tensor(self):
        board_array = np.array(self.sim.board, dtype=np.float32)
        tensor = np.zeros((7, board_array.shape[0], board_array.shape[1]), dtype=np.float32)
        
        # TODO board 크기는 가변적임을 고려
        tensor[0] = (board_array == 1).astype(np.float32)  # black
        tensor[1] = (board_array == 2).astype(np.float32)  # white
        tensor[2] = (board_array == 0).astype(np.float32)  # empty
        tensor[3] = (board_array == 3).astype(np.float32)  # blocked
        tensor[4] = 0.0
        for move in self.valid_moves:
            tensor[4, move['row'], move['col']] = 1.0  # valid moves
        tensor[5] = float(self.sim.currentPlayer == 1)  # 흑이면 1.0
        tensor[6] = float(self.sim.currentPlayer == 2)  # 백이면 1.0

        return torch.tensor(tensor, dtype=torch.float32)#.to(self.device)  # (7, 8, 8) channel 7, height 8, width 8

    def step(self, action):
        flag_random = False
        reward = 0.0

        self.valid_moves = self.sim.getValidMoves(self.sim.currentPlayer)

        if not self.valid_moves:
            self.sim.currentPlayer = self.sim.BLACK if self.sim.currentPlayer == self.sim.WHITE else self.sim.WHITE
            return self.board, 0.0, self.done, self.done, self.sim.currentPlayer # 여기서의 info는 다음 플레이어

        if isinstance(action, int):
            action_index = action
        else:
            action_index = action.argmax().item()
        row, col = divmod(action_index, self.board.shape[-1])
        if not any(move['row'] == row and move['col'] == col for move in self.valid_moves):
            flag_random = True
            reward = self.penalty_nonvalid
            move = random.choice(self.valid_moves)
            row, col = move['row'], move['col']
        else:
            reward = len(self.valid_moves) * self.weight_mobility_mine  # mobility reward

        self.sim.makeMove(row, col, self.sim.currentPlayer)
        self.sim.updateStatus()

        # action으로 야기된 다음 상태
        self.sim.currentPlayer = self.sim.BLACK if self.sim.currentPlayer == self.sim.WHITE else self.sim.WHITE
        self.board = self.get_board_tensor()

        # 게임 종료 조건
        if (not self.sim.getValidMoves(self.sim.BLACK)) and (not self.sim.getValidMoves(self.sim.WHITE)):
            self.done = True
            discs = self.sim.countDiscs()
            if self.agent == self.sim.BLACK: # 흑 점수
                if discs[0] > discs[1]:
                    reward = float(discs[0] * 2 - discs[1])*2 # 자신이 승리
                elif discs[0] < discs[1]:
                    reward = float(discs[1] - discs[0])*2 # 상대가 승리

            elif self.agent == self.sim.WHITE: # 백 점수
                if discs[1] > discs[0]:
                    reward = float(discs[1] * 2 - discs[0])*2 # 자신이 승리
                elif discs[1] < discs[0]:
                    reward = float(discs[0] - discs[1])*2 # 상대가 승리
        # 추가 보상
        else:
            self.done = False
            if (not flag_random) and not (self.sim.currentPlayer is self.agent):
                reward -= len(self.valid_moves) * self.weight_mobility_opponent  # mobility penalty
                reward = max(reward, 0)

        return self.board, reward, self.done, self.done, self.sim.currentPlayer # 여기서의 info는 다음 플레이어
