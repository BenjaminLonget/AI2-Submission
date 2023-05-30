import numpy as np
from enum import Enum

# State inherits from Enum
class State(Enum):  # Each piece should have its own state. Opponents as well
    HOME = 0
    GLOBE_SAFE = 1
    GLOBE_UNSAFE = 2    # Enemy globe
    STAR = 3
    END_ZONE = 4
    GOAL = 5    # Not needed??
    FURTHEST = 6        # If furthest * 1.2 i reward eller noget
    DANGER = 7  # In front of enemy (less than 7 away or 13 if on a star)
    GLOBAL_SAFE = 8 # When not in danger? When 7/13 or more away from nearest chasing opponent
    DOUBLE = 9  # When two pieces are on the same field
    HUNTING = 10    # When an opponent is less than 7 ahead (13 if on star)

class Action(Enum):
    MoveDice = 0
    MoveFromHome = 1
    MoveToGlobe = 2
    MoveToEnemyGlobe = 3
    MoveToStar = 4
    MoveToDanger = 5
    MoveFromDanger = 6
    MoveToEndZone = 7
    MoveToGoal = 8
    MoveToDouble = 9
    MoveToKill = 10
    MoveToSuicide = 11
    MoveToHunt = 12


GLOBES = np.array([9, 22, 35, 48])  # 1 removed
ENEMY_GLOBES = np.array([14, 27, 40])
STARS = np.array([5, 12, 18, 25, 31, 38, 44, 51])
STARS_7 = ([5, 18, 31, 44])
STARS_6 = ([12, 25, 38, 51])

def getStates(player_pieces, move_pieces, enemy_pieces):  # When getting future state, just add dice to player_pieces
    old_furthest_index = -1
    old_furthest_pos = 0
    states = np.zeros((len(player_pieces), len(State)))
    for piece in move_pieces:
        piece_pos = player_pieces[piece]
        state = np.zeros(len(State))
        if piece_pos == 0:
            state[State.HOME.value] = True
        if piece_pos in GLOBES:
            state[State.GLOBE_SAFE.value] = True
        if piece_pos in ENEMY_GLOBES:
            state[State.GLOBE_UNSAFE.value] = True
        if piece_pos in STARS:
            state[State.STAR.value] = True
        if piece_pos > 51:
            state[State.END_ZONE.value] = True
        if piece_pos == 57:
            state[State.GOAL.value] = True
        if piece_pos > old_furthest_pos and piece_pos != 57:
            old_furthest_pos = piece_pos
            old_furthest_index = piece
        if check_danger(piece_pos, enemy_pieces):
            state[State.DANGER.value] = True
        if np.count_nonzero(player_pieces == piece_pos) > 1 and piece_pos != 0:
            state[State.DOUBLE.value] = True
            state[State.DANGER.value] = False
        # if not (state[State.END_ZONE.value] and state[State.DANGER.value] and state[State.GOAL.value]):
        #     state[State.GLOBAL_SAFE.value] = True
        if not (state[State.END_ZONE.value] and state[State.DANGER.value] and state[State.GOAL.value]):
            state[State.GLOBAL_SAFE.value] = True
        if check_hunt(piece_pos, enemy_pieces) and piece_pos != 0:
            state[State.HUNTING.value] = True

        states[piece] = state
    if old_furthest_pos > 0:
        states[old_furthest_index, State.FURTHEST.value] = True
    return states


def get_possible_actions(states, future_states, move_pieces, player_pieces, enemies, dice):
    actions = np.zeros((len(player_pieces), len(Action)))
    for piece in move_pieces:
        current_pos = player_pieces[piece]
        future_pos = current_pos + dice
        action_table = np.zeros(len(Action))
        if future_pos in STARS:
            if check_kill(future_pos, enemies) and future_pos < 52:
                action_table[Action.MoveToKill.value] = True
            action_table[Action.MoveToStar.value] = True
            if future_pos in STARS_6:
                future_pos += 6
            if future_pos in STARS_7:
                future_pos += 7
        if current_pos == 0 and dice == 6:              # Move from home er special case
            future_pos = 1
            action_table[Action.MoveFromHome.value] = True
        elif current_pos != 0:
            if future_pos in GLOBES:
                action_table[Action.MoveToGlobe.value] = True
            if future_pos in ENEMY_GLOBES:
                action_table[Action.MoveToEnemyGlobe.value] = True
            #if current_pos + dice in STARS:
            #    action_table[Action.MoveToStar.value] = True
            if check_danger(future_pos, enemies):  # and not states[piece, State.DANGER.value]: # Not sure what is best here, keeping it like this to hopefully have an opponent overtake
                action_table[Action.MoveToDanger.value] = True
            if states[piece, State.DANGER.value] and not check_danger(future_pos, enemies):
                action_table[Action.MoveFromDanger.value] = True
            if future_pos > 51 and current_pos < 52:
                action_table[Action.MoveToEndZone.value] = True
                action_table[Action.MoveToDanger.value] = False
                if states[piece, State.DANGER.value]:
                    action_table[Action.MoveFromDanger.value] = False
            if future_pos == 57:
                action_table[Action.MoveToGoal.value] = True
            if future_pos in player_pieces and future_pos != 0 and future_pos < 52:
                action_table[Action.MoveToDouble.value] = True
            if check_suicide(future_pos, enemies) and future_pos < 52:
                action_table[Action.MoveToSuicide.value] = True
            if check_kill(future_pos, enemies) and future_pos < 52 and not action_table[Action.MoveToSuicide.value]:  #For star kill
                action_table[Action.MoveToKill.value] = True
            # if check_kill(current_pos + dice, enemies):
            #     action_table[Action.MoveToKill.value] = True

            # if check_suicide(current_pos + dice, enemies):
            #     action_table[Action.MoveToSuicide.value] = True
            if True not in action_table:  # If nothing else is possible but the piece is still moveable, move dice
                action_table[Action.MoveDice.value] = True
        if check_hunt(future_pos, enemies) and not states[piece, State.HUNTING.value]:
            action_table[Action.MoveToHunt.value] = True
        actions[piece] = action_table
    return actions


def check_suicide(future_pos, enemies):
    if future_pos > 51:
        return False
    enemy_idx = 1
    #enemy_idx_number = 0
    #enemy_idx_list = [0, 1, 2, 3]
    #enemy_idx_list.remove(player_idx)
    for enemy in enemies:
        for enemy_piece_local in enemy:
            if enemy_piece_local != 0 and enemy_piece_local < 52:
                enemy_piece = (enemy_piece_local + (13 * enemy_idx)) % 52
                if future_pos == enemy_piece:
                    if not enemy_piece == 1 and (enemy_piece in GLOBES or np.count_nonzero(enemy == enemy_piece_local) > 1 or enemy_piece in ENEMY_GLOBES):
                        return True
                    # doub_enem = (enemy + (13 * enemy_idx)) % 52
                    # if enemy_piece in GLOBES or enemy_piece in ENEMY_GLOBES or np
        enemy_idx += 1
    return False


def check_kill(future_pos, enemies):
    enemy_idx = 1
    for enemy in enemies:
        for enemy_piece_local in enemy:
            if enemy_piece_local != 0 and enemy_piece_local < 52:
                enemy_piece = (enemy_piece_local + (13 * enemy_idx)) % 52
                if future_pos == enemy_piece:
                    if enemy_piece == 1 or (enemy_piece not in GLOBES and not np.count_nonzero(enemy == enemy_piece_local) > 1 and enemy_piece not in ENEMY_GLOBES):
                        return True
        enemy_idx += 1
    return False


def check_danger(piece_pos, enemy_positions):
    if piece_pos > 51 or piece_pos in GLOBES:
        return False
    enemy_idx = 1
    for enemy in enemy_positions:
        for enemy_piece_local in enemy:
            if enemy_piece_local != 0 and enemy_piece_local < 52:
                enemy_piece = (enemy_piece_local + (13 * enemy_idx)) % 52
                if piece_pos < 7 and enemy_piece > 47:
                    enemy_piece = enemy_piece - 52
                if piece_pos - enemy_piece > 0 and (piece_pos - enemy_piece < 7 or (piece_pos - enemy_piece < 13 and piece_pos in STARS_7) or (piece_pos - enemy_piece < 14 and piece_pos in STARS_6)):
                    return True
        enemy_idx += 1
    return False


def check_hunt(piece_pos, enemy_positions):
    if piece_pos > 51 or piece_pos == 0:
        return False
    enemy_idx = 1
    for enemy in enemy_positions:
        for enemy_piece_local in enemy:
            if enemy_piece_local != 0 and enemy_piece_local < 52:
                enemy_pos = (enemy_piece_local + (13 * enemy_idx)) % 52
                if enemy_pos - piece_pos > 0 and (enemy_pos - piece_pos < 7 or (enemy_pos - piece_pos < 13 and enemy_pos in STARS_7) or (enemy_pos - piece_pos < 14 and enemy_pos in STARS_6)):
                    return True
        enemy_idx += 1
    return False


