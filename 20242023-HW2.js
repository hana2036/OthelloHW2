function analyzeStage(stageConfig, initialBoard, validMoves, api) {
    // console.log(`stageConfig: ${stageConfig}`);
    console.log('initialBoard:', initialBoard);

    const EMPTY = 0;
    const BLACK = 1;
    const WHITE = 2;
    const BLOCKED = 3;
    const BOARD_SIZE = stageConfig.boardSize;

    let weightUpdatePeriod = 10000;
    let MAX_DEPTH = 3;
    let flagAnalyze = true;

    let weights = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));


    // ============================================
    // studentStrategy()
    // ============================================
    const studentStrategy = function (board, player, validMoves, makeMove) {
        // console.time("코드 실행 시간");

        if (player === BLACK) {
            ME = BLACK;
            YOU = WHITE;
        }
        else {
            ME = WHITE;
            YOU = BLACK;
        }

        if (typeof makeMove === 'number') {
            if (makeMove === 741) {
                flagAnalyze = true;
                MAX_DEPTH = 3;
            }
        }
        else {
            flagAnalyze = false;
            MAX_DEPTH = 3;
        }
        
        // Run minimax to find the best move
        let bestScore = -Infinity;
        let bestMove = null;
        for (const move of validMoves) {
            let boardCopy = board.map(row => [...row]);
            boardCopy = api.simulateMove(boardCopy, ME, move.row, move.col).resultingBoard;
            const score = minimax(boardCopy, MAX_DEPTH, -Infinity, Infinity, false);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        // console.log('weights:\n', weights);

        // console.timeEnd("코드 실행 시간");

        return bestMove;
    };


    // ============================================
    // functions for symmetry
    // ============================================
    function detectSymmetriesFromBlocked() {
        const blockedSet = new Set(stageConfig.initialBlocked.map(({ r, c }) => `${r},${c}`));

        // blockedCells가 없으면 모든 대칭 허용
        if (blockedSet.size === 0) {
            return {
            horizontal: true,
            vertical: true,
            diagonal: true,
            reverseDiagonal: true,
            rotational180: true,
            rotational90: true
            };
        }

        // 대칭성 판별 함수
        function isSymmetric(mapFn) {
            const forwardMapped = new Set();
            const reverseMapped = new Set();

            // forward: 변환 후 좌표들이 모두 blockedSet에 포함되어야 함
            for (const { r, c } of stageConfig.initialBlocked) {
                const { row, col } = mapFn(r, c);
                forwardMapped.add(`${row},${col}`);
            }

            // reverse: blockedSet에 있는 좌표들도 mapFn을 거꾸로 적용한 결과여야 함
            for (const coord of blockedSet) {
                const [r, c] = coord.split(',').map(Number);
                const { row, col } = mapFn(r, c);
                reverseMapped.add(`${row},${col}`);
            }

            // 양방향 모두 blockedSet에 포함되어야 진정한 대칭
            return (
                [...forwardMapped].every(coord => blockedSet.has(coord)) &&
                [...reverseMapped].every(coord => blockedSet.has(coord))
            );
        }

        // 실제 대칭 판별 반환
        return {
            horizontal: isSymmetric((r, c) => ({ row: r, col: BOARD_SIZE - 1 - c })),
            vertical: isSymmetric((r, c) => ({ row: BOARD_SIZE - 1 - r, col: c })),
            diagonal: isSymmetric((r, c) => ({ row: c, col: r })),
            reverseDiagonal: isSymmetric((r, c) => ({ row: BOARD_SIZE - 1 - c, col: BOARD_SIZE - 1 - r })),
            rotational180: isSymmetric((r, c) => ({ row: BOARD_SIZE - 1 - r, col: BOARD_SIZE - 1 - c })),
            rotational90: isSymmetric((r, c) => ({ row: c, col: BOARD_SIZE - 1 - r }))
        };
    }


    function generateGroupsWithSymmetry(symmetry) {
        const visited = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(false));
        const groups = [];

        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (visited[r][c]) continue;

                const group = [[r, c]];

                if (symmetry.horizontal) group.push([r, BOARD_SIZE - 1 - c]);
                if (symmetry.vertical) group.push([BOARD_SIZE - 1 - r, c]);
                if (symmetry.diagonal) group.push([c, r]);
                if (symmetry.reverseDiagonal) group.push([BOARD_SIZE - 1 - c, BOARD_SIZE - 1 - r]);
                if (symmetry.rotational180) group.push([BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c]);
                if (symmetry.rotational90) group.push([c, BOARD_SIZE - 1 - r]);

                const uniqueGroup = group.filter(([x, y], i, arr) =>
                    x >= 0 && x < BOARD_SIZE &&
                    y >= 0 && y < BOARD_SIZE &&
                    arr.findIndex(([a, b]) => a === x && b === y) === i
                );

                groups.push(uniqueGroup);
                uniqueGroup.forEach(([x, y]) => visited[x][y] = true);
            }
        }

        return groups;
    }
    
    const symmetries = detectSymmetriesFromBlocked();
    const groups = generateGroupsWithSymmetry(symmetries);


    // ============================================
    // functions for studentStrategy()
    // ============================================
    function evaluateBoard(board, player) {
        let score = 0;

        if (flagAnalyze) {
            score += evaluatePosition(board, player);
        }
        else {
            // score += evaluateStability(board, player) * 5;
            score += evaluateMobility(board, player) * 5;
            score += evaluatePosition(board, player);
        }
        
        return score;
    }

    function evaluatePosition(board, player) {
        let score = 0;
        let opponent = player === BLACK ? WHITE : BLACK;

        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === player) {
                    score += weights[row][col];
                } else if (board[row][col] === opponent) {
                    score -= weights[row][col];
                }
            }
        }
        return score;
    }

    function evaluateStability(board, player) {
        let stableDiscs = countStableDiscs(board, player);
        let opponent = player === BLACK ? WHITE : BLACK;
        let opponentStableDiscs = countStableDiscs(board, opponent);

        return stableDiscs - opponentStableDiscs;
    }

    function countStableDiscs(board, player) {
        let stable = 0;
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === player && isStable(board, player, row, col)) {
                    stable++;
                }
            }
        }
        return stable;
    }

    function isStable(board, player, row, col) {
        let count = [0,0,0,0];
        let i = 0;

        const directions_linear = [
            [[-1, -1],[1, 1]],
            [[-1, 0],[1, 0]],
            [[-1, 1],[1, -1]],
            [[0, -1],[0, 1]]
        ];
        for (const line of directions_linear) {
            for (const [dr, dc] of line) {
                let r = row + dr;
                let c = col + dc;
                while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                    count[i] = 0;
                    if ((board[r][c] !== player) && (board[r][c] !== BLOCKED)) {
                        break;
                    }
                    r += dr;
                    c += dc;
                    count[i] = 1;
                }
                if (count[i] === 1) {
                    break;
                }
            }
            if (count[i] === 0) {
                return false;
            }
            i += 1;
        }
        return true;
    }

    function evaluateMobility(board, player) {
        const playerMoves = api.getValidMoves(board, player).length;
        const opponentMoves = api.getValidMoves(board, player === BLACK ? WHITE : BLACK).length;
        return playerMoves - opponentMoves;
    }

    function minimax(board, depth, alpha, beta, maximizingPlayer) {
        if (depth === 0) {
            return evaluateBoard(board, maximizingPlayer ? ME : YOU);
        }
        const player = maximizingPlayer ? ME : YOU;
        const currentValidMoves = api.getValidMoves(board, player);

        // If no valid moves, pass turn to opponent
        if (currentValidMoves.length === 0) {
            // Recursive call with opponent player
            return minimax(board, depth - 1, alpha, beta, !maximizingPlayer);
        }

        if (maximizingPlayer) {
            let maxEval = -Infinity;
            for (const move of currentValidMoves) {
                // Copy the board
                let boardCopy = board.map(row => [...row]);

                // Simulate the move
                boardCopy= api.simulateMove(boardCopy, ME, move.row, move.col).resultingBoard;

                // Recursive evaluation
                const eval = minimax(boardCopy, depth - 1, alpha, beta, false);
                maxEval = Math.max(maxEval, eval);

                // Alpha-beta pruning
                alpha = Math.max(alpha, eval);
                if (beta <= alpha)
                    break;
            }
            return maxEval;
        } else {
            let minEval = Infinity;
            for (const move of currentValidMoves) {
                // Copy the board
                let boardCopy = board.map(row => [...row]);

                // Simulate the move
                boardCopy= api.simulateMove(boardCopy, YOU, move.row, move.col).resultingBoard;

                // Recursive evaluation
                const eval = minimax(boardCopy, depth - 1, alpha, beta, true);
                minEval = Math.min(minEval, eval);

                // Alpha-beta pruning
                beta = Math.min(beta, eval);
                if (beta <= alpha)
                    break;
            }
            return minEval;
        }
    }

    // ============================================
    // functions for analyzing
    // ============================================
    function runOthelloSimulationFor60Seconds() {
        const startTime = Date.now();
        const endTime = startTime + 59000;

        let blackWins = 0, whiteWins = 0, draws = 0, totalGames = 0;
        let gameSinceLastUpdate = 0;
        const moveStats = {}; // position: { blackMoves, whiteMoves, blackWins, whiteWins }

        function initStats() {
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    const key = `${r},${c}`;
                    moveStats[key] = {
                        blackMoves: 0,
                        whiteMoves: 0,
                        blackWins: 0,
                        whiteWins: 0
                    };
                }
            }
        }

        function playOneGame() {
            // let board = initialBoard;
            let board = initialBoard.map(row => [...row]);
            let currentPlayer = BLACK;
            let passCount = 0;
            const gameMoves = [];

            while (passCount < 2) {
                const moves = api.getValidMoves(board, currentPlayer);
                if (moves.length === 0) {
                    passCount++;
                    currentPlayer = currentPlayer === BLACK ? WHITE : BLACK;
                    continue;
                }

                let move = moves[0];
                if (currentPlayer === BLACK) {
                    // move = studentStrategy(board, BLACK, moves, 741);
                    move = moves[Math.floor(Math.random() * moves.length)];
                }
                else if (currentPlayer === WHITE) {
                    move = moves[Math.floor(Math.random() * moves.length)];
                }

                board = api.simulateMove(board, currentPlayer, move.row, move.col).resultingBoard;
                gameMoves.push({ player: currentPlayer, row: move.row, col: move.col });

                currentPlayer = currentPlayer === BLACK ? WHITE : BLACK;
                passCount = 0;
            }

            const result_black = api.evaluateBoard(board, BLACK);
            const winner = result_black.pieceScore === 0 ? 0
                            : result_black.pieceScore > 0 ? BLACK : WHITE;
            
            if (winner === BLACK) blackWins++;
            else if (winner === WHITE) whiteWins++;
            else draws++;

            // 위치별 통계 갱신
            gameMoves.forEach(move => {
                const key = `${move.row},${move.col}`;
                if (move.player === BLACK) {
                    moveStats[key].blackMoves++;
                    if (winner === BLACK) moveStats[key].blackWins++;
                } else {
                    moveStats[key].whiteMoves++;
                    if (winner === WHITE) moveStats[key].whiteWins++;
                }
            });

            totalGames++;
            gameSinceLastUpdate++;

            if (gameSinceLastUpdate >= weightUpdatePeriod) {
                updateWeightsFromStats();
                gameSinceLastUpdate = 0;
                weightUpdatePeriod += 10;
            }
        }


        function applySymmetryAveragedWeights(rawWeights) {
            const result = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));

            for (const group of groups) {
                const valid = group.filter(([r, c]) => rawWeights[r][c] !== 0);

                const avg = valid.length > 0
                ? Math.round(valid.reduce((sum, [r, c]) => sum + rawWeights[r][c], 0) / valid.length)
                : 0;

                group.forEach(([r, c]) => {
                    result[r][c] = rawWeights[r][c] === 0 ? 0 : avg;
                });
            }

            return result;
        }


        // 정규화 후 평균 적용
        function normalizeAndApplySymmetry(rawWeights) {
            // 비어있지 않은 값만 정규화 대상
            const flat = [];
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (rawWeights[r][c] !== 0) {
                        flat.push(rawWeights[r][c]);
                    }
                }
            }

            const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
            const maxDev = Math.max(...flat.map(v => Math.abs(v - mean))) || 1;

            // 정규화
            const normalized = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (rawWeights[r][c] !== 0) {
                        normalized[r][c] = Math.round(((rawWeights[r][c] - mean) / maxDev) * 100);
                    }
                }
            }

            // 대칭성 기반 평균 적용
            return applySymmetryAveragedWeights(normalized);
        }


        function updateWeightsFromStats() {
            // 가중치 계산: valueScore = (승률 × 샘플 수 보정)
            const tempWeights = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));

            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    const key = `${r},${c}`;
                    const stat = moveStats[key];
                    const total = stat.blackMoves + stat.whiteMoves;
                    const wins = stat.blackWins + stat.whiteWins;
                    const winRate = total > 0 ? wins / total : 0;
                    const score = winRate * Math.min(1, total / 10) * 100;

                    tempWeights[r][c] = score;
                }
            }

            // 평균 중심 정규화 [-100, 100]
            // const avg = flat.reduce((a, b) => a + b, 0) / flat.length;
            // const maxDev = Math.max(...flat.map(v => Math.abs(v - avg))) || 1;
            // const normWeights = tempWeights.map(row =>
            //     row.map(v => Math.round(((v - avg) / maxDev) * 100))
            // );

            // 대칭성 기반 평균화
            // const symmetrized = applySymmetricWeightAveraging(normWeights);
            // weights = symmetrized;

            weights = normalizeAndApplySymmetry(tempWeights);
        }

        function loop() {
            initStats();

            while (Date.now() < endTime) {
                playOneGame();
            }

            updateWeightsFromStats();

            console.log('weights:\n', weights);

            console.log(`
                총 시뮬레이션 수: ${totalGames}
                흑 승리: ${blackWins}
                백 승리: ${whiteWins}
                무승부: ${draws}
                `);
        }

        setTimeout(loop, 0);
    }

    runOthelloSimulationFor60Seconds();


    return studentStrategy;
}


// 가능한 변칙 룰 //
// - 직사각형 보드 사이즈
// - 초기 돌 위치(비대칭, 다른 곳 등), 갯수
// - 비대칭 blocked cell
// - 하나라도 뒤집으면 해당 줄 전부 뒤집기
