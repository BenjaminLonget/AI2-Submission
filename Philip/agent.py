import numpy as np

# STATE DEFINITIONS
NUM_STATES = 13
STATE_SAFE_GOAL_ZONE = 0        # Piece is in goal zone
STATE_SAFE_GLOBE = 1            # Piece is on a globe
STATE_SAFE_STACKED = 2          # Piece is stacked with other pieces
STATE_SAFE_DISTANCCE = 3        # Piece is more than 6 away from another piece
STATE_CAN_KILL = 4              # Piece can kill other piece
STATE_WILL_BE_KILLED = 5        # Piece will be killed by moving
STATE_MOVE_TO_SAFETY = 6        # Piece can move out of danger zone ex more than 6 away from another piece, to a globe, to another friendly piece, to goal zone
STATE_MOVE_TO_STAR = 7          # Piece can move to a star
STATE_MOVE_TO_DANGER_ZONE = 8   # Piece will move in front of an enemy
STATE_MOVE_TO_GOAL = 9          # Piece can reach the goal
STATE_IN_DANGER_ZONE = 10       # Piece is in front of an enemy
STATE_IN_HOME = 11              # Piece is at home
STATE_ON_ENEMY_GLOBE = 12       # Piece is on an emenyy globe
#STATE_CHASE_ENEMY = 11         # Piece can chase enemy, enemy is less than 6 away


# GAME DEFINITIONS

TOTAL_NUMBER_OF_TILES = 58
DICE_MOVE_OUT_OF_HOME = 6
NO_ENEMY = -1

TILE_FREE = 0
TILE_HOME = 1
TILE_START = 2
TILE_GLOBE = 3
TILE_GOAL_AREA = 4
TILE_STAR = 5
TILE_GOAL = 6
TILE_ENEMY_1_GLOB = 7
TILE_ENEMY_2_GLOB = 8
TILE_ENEMY_3_GLOB = 9
LIST_TILE_ENEMY_GLOBS = [TILE_ENEMY_1_GLOB, TILE_ENEMY_2_GLOB, TILE_ENEMY_3_GLOB]

NULL_POS = -1
HOME_INDEX = 0
START_INDEX = 1
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]

HOME_AREA_INDEXS = [52, 53, 54, 55, 56]
GOAL_INDEX = 57
GLOB_INDEXS = [9, 22, 35, 48]
ENEMY_1_GLOB_INDX = 14
ENEMY_2_GLOB_INDX = 27
ENEMY_3_GLOB_INDX = 40
STAR_AT_GOAL_AREAL_INDX = STAR_INDEXS[-1]

BOARD_TILES = np.full(TOTAL_NUMBER_OF_TILES, TILE_FREE)
BOARD_TILES[HOME_INDEX] = TILE_HOME
BOARD_TILES[START_INDEX] = TILE_START
BOARD_TILES[STAR_INDEXS] = TILE_STAR
BOARD_TILES[GLOB_INDEXS] = TILE_GLOBE
BOARD_TILES[HOME_AREA_INDEXS] = TILE_GOAL_AREA
BOARD_TILES[GOAL_INDEX] = TILE_GOAL
BOARD_TILES[ENEMY_1_GLOB_INDX] = TILE_ENEMY_1_GLOB
BOARD_TILES[ENEMY_2_GLOB_INDX] = TILE_ENEMY_2_GLOB
BOARD_TILES[ENEMY_3_GLOB_INDX] = TILE_ENEMY_3_GLOB

ENEMY_1_INDX_AT_HOME = 40  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 1
ENEMY_2_INDX_AT_HOME = 27  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 2
ENEMY_3_INDX_AT_HOME = 14  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 3




class Agent:
    def __init__(self):
        self.chromosome = []
        self.gamesWon = 0
        self.fitness = 0

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome
    
    def get_chromosome(self):
        return self.chromosome
    
    def calc_state(self, dice, move_pieces, player_pieces, enemy_pieces):
        # Calc state of moveable pieces
        state = np.full((len(move_pieces), NUM_STATES), 0)
        
        for i in range(len(move_pieces)):
            tempState = np.full(NUM_STATES, 0)

            piecePos = player_pieces[move_pieces[i]]
            pieceNextPos = piecePos + dice
            
            # Check for overshoot
            if pieceNextPos > GOAL_INDEX:
                overshoot = pieceNextPos - GOAL_INDEX
                pieceNextPos = pieceNextPos - overshoot

            # Check if piece is in GOAL zone
            if BOARD_TILES[piecePos] == TILE_GOAL_AREA:
                tempState[STATE_SAFE_GOAL_ZONE] = 1
            
            # Check if piece is on a globe
            if BOARD_TILES[piecePos] == TILE_GLOBE:
                tempState[STATE_SAFE_GLOBE] = 1

            # Check if piece is stacked with other pieces
            count = 0
            for j in range(len(player_pieces)):
                if player_pieces[j] == piecePos:
                    count += 1
            if count > 1:
                tempState[STATE_SAFE_STACKED] = 1
            
            # Check if piece can kill other piece
            if BOARD_TILES[pieceNextPos] in enemy_pieces:
                # Check if enemy piece is on a globe
                if BOARD_TILES[pieceNextPos] == TILE_GLOBE or BOARD_TILES[pieceNextPos] in LIST_TILE_ENEMY_GLOBS:
                    if pieceNextPos in enemy_pieces:
                        tempState[STATE_WILL_BE_KILLED] = 1
                
                # Check if enemy is double staced on next position
                else:
                    count = 0
                    for j in range(len(enemy_pieces)):
                        count = 0
                        for k in range(len(enemy_pieces[j])):
                            if enemy_pieces[j][k] == pieceNextPos:
                                count += 1
                        if count > 1:
                            break
                    if count > 1:    
                        tempState[STATE_WILL_BE_KILLED] = 1
                    
                    else:
                        tempState[STATE_CAN_KILL] = 1

            # Check if piece can move to safety or will move to danger zone or is in danger zone
            for j in range(len(enemy_pieces)):
                if tempState[STATE_MOVE_TO_SAFETY] == 1 and tempState[STATE_MOVE_TO_SAFETY] == 1:
                    break
                for k in range(len(enemy_pieces[j])):
                    if enemy_pieces[j][k] - pieceNextPos > 0:
                        tempState[STATE_MOVE_TO_SAFETY] = 1
                    if pieceNextPos - enemy_pieces[j][k] <= 6 and pieceNextPos - enemy_pieces[j][k] > 0:
                        tempState[STATE_MOVE_TO_DANGER_ZONE] = 1
                    if piecePos - enemy_pieces[j][k] <= 6 and piecePos - enemy_pieces[j][k] > 0:
                        tempState[STATE_IN_DANGER_ZONE] = 1
            
            # Check if piece can move to a star
            if BOARD_TILES[pieceNextPos] == TILE_STAR:
                tempState[STATE_MOVE_TO_STAR] = 1
            
            # Check if the peice can reach goal
            if pieceNextPos == GOAL_INDEX:
                tempState[STATE_MOVE_TO_GOAL] = 1
            
            # Check if piece is in home
            if piecePos == HOME_INDEX:
                tempState[STATE_IN_HOME] = 1
            
            # Check if piece is on enenmy globe
            if BOARD_TILES[piecePos] in LIST_TILE_ENEMY_GLOBS:
                tempState[STATE_ON_ENEMY_GLOBE] = 1
            
            # add tempsate to state
            state[i] = tempState
        return state

    def calc_NN_output(self, state):
        # Calc NN output
        output = np.sum(np.dot(state, self.chromosome))
        return output
    
    def get_best_action(self, dice, move_pieces, player_pieces, enemy_pieces):
        # Calc state of moveable pieces
        state = self.calc_state(dice, move_pieces, player_pieces, enemy_pieces)

        output = np.full(len(move_pieces), 0)

        # Calc NN output
        for i in range(len(move_pieces)):
            output[i] = self.calc_NN_output(state[i])
        
        # Get best action
        best_action = np.argmax(output)
        return best_action
