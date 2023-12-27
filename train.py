#!/usr/bin/env python3
import os
import chess.pgn

for fn in os.listdir("data"):
    pgn = open(os.path.join("data", fn))
    while 1:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            break
        # print(game)
        result = game.headers["Result"]
        board = game.board()
        print("Game result: ", result)
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            print(i)
            print(board)
        exit(0)
    # break    

