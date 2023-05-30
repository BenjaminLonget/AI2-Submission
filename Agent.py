import numpy as np
import random
import ludopy
import cv2
from StateSpace import getStates, get_possible_actions, State, Action
from Philip.agent import Agent


NUM_STATES = 1023   # 10 bits set to 1 indicates all possible combinations - currently missing the chasing state
NUM_ACTIONS = 13  # 12    # 12 different possible actions - currently missing the chase action

def epsilon_decay(epsilon, decay_rate, episode):
    new_eps = epsilon * np.exp(-decay_rate * episode)
    return new_eps


def binatointeger(binary):
  number = 0
  for b in binary:
    number = (2 * number) + b
  return number


def get_action(move_pieces, possible_actions, states, current_eps, q_table):
    best_action = 0
    action_idx = [0]
    highest_q_idx = move_pieces[0]
    if random.uniform(0, 1) < current_eps:  # np.random maybe?
        chosen_piece = move_pieces[np.random.randint(0, len(move_pieces))]  # The chosen piece is the index of the piece
        actions = possible_actions[chosen_piece]
        for action in range(0, len(actions)):   # this is to choose a random single acion
            if actions[action] == True:
                action_idx.append(action)
        best_action = random.choice(action_idx)
    else:
        #highest_q_idx = move_pieces[0]  # To ensure that a valid actions i chosen
        old_Q = float('-inf')   # Just to init the value
        for piece in move_pieces:   # For each piece, look at the possible action table
            # Should all valid options be summed since multiple actions occur?
            # The action table could also be extended with all combinations - Currently doing this
            # This is where we convert from the list of mundtlige actions to the actual action of which piece to move
            state = binatointeger(states[piece].astype(int))    # convert to the index of the state
            for action in range(0, len(possible_actions[piece])):
                if possible_actions[piece][action]:
                    q_value = q_table[state][action]
                    if q_value > old_Q:
                        old_Q = q_value
                        highest_q_idx = piece
                        best_action = action

        chosen_piece = highest_q_idx
    return chosen_piece, best_action


def get_reward(chosen_piece, possible_actions, best_action, states, future_states):
    reward = 0
    MOVE_DICE = 0.01
    MOVE_FROM_HOME = 1.5
    MOVE_TO_GLOBE = 0.9
    MOVE_TO_ENEMY_GLOBE = -0.05
    MOVE_TO_STAR = 0.8
    MOVE_TO_DANGER = -0.8
    MOVE_FROM_DANGER = 1.6
    MOVE_TO_END_ZONE = 0.8
    MOVE_TO_GOAL = 0.9
    MOVE_TO_DOUBLE = 0.8
    MOVE_TO_KILL = 3.5
    MOVE_TO_SUICIDE = -1.5
    MOVE_TO_HUNT = 1.0

    VERY_BAD = -0.8
    BAD = -0.3
    GOOD = 0.5
    VERY_GOOD = 1.2

    state = states[chosen_piece]
    future_state = future_states[chosen_piece]
    if state[State.HOME.value]:
        MOVE_FROM_HOME += GOOD
        MOVE_TO_KILL += VERY_GOOD
    if state[State.GLOBE_SAFE.value]:
        MOVE_DICE += BAD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += VERY_GOOD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += VERY_GOOD
        MOVE_TO_DOUBLE += GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += GOOD
    if state[State.GLOBE_UNSAFE.value]:
        MOVE_DICE += GOOD
        MOVE_TO_GLOBE += VERY_GOOD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += VERY_GOOD
        MOVE_TO_DANGER += GOOD
        MOVE_FROM_DANGER += VERY_GOOD
        MOVE_TO_END_ZONE += VERY_GOOD
        MOVE_TO_GOAL += VERY_GOOD
        MOVE_TO_DOUBLE += VERY_GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.STAR.value]:
        MOVE_DICE += GOOD
        MOVE_TO_GLOBE += GOOD
        MOVE_TO_ENEMY_GLOBE += GOOD
        MOVE_TO_STAR += VERY_GOOD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_FROM_DANGER += VERY_GOOD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += VERY_GOOD
        MOVE_TO_DOUBLE += GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.END_ZONE.value]:
        MOVE_DICE += VERY_BAD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_TO_GOAL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
    #if state[State.GOAL.value]:
    if state[State.FURTHEST.value] and not state[State.END_ZONE.value]:
        MOVE_DICE += GOOD
        MOVE_TO_GLOBE += VERY_GOOD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += VERY_GOOD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_FROM_DANGER += VERY_GOOD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += GOOD
        MOVE_TO_DOUBLE += GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.DANGER.value]:
        MOVE_DICE += VERY_GOOD
        MOVE_TO_GLOBE += VERY_GOOD
        MOVE_TO_ENEMY_GLOBE += GOOD
        MOVE_TO_STAR += VERY_GOOD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_FROM_DANGER += VERY_GOOD
        MOVE_TO_END_ZONE += VERY_GOOD
        MOVE_TO_GOAL += 0
        MOVE_TO_DOUBLE += VERY_GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.GLOBAL_SAFE.value]:
        MOVE_DICE += BAD
        MOVE_TO_GLOBE += GOOD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += GOOD
        MOVE_TO_DANGER += VERY_BAD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += GOOD
        MOVE_TO_DOUBLE += GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.DOUBLE.value]:
        MOVE_DICE += BAD
        MOVE_TO_GLOBE += GOOD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += GOOD
        MOVE_TO_DANGER += BAD
        MOVE_FROM_DANGER += GOOD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += GOOD
        MOVE_TO_DOUBLE += VERY_GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += VERY_GOOD
    if state[State.HUNTING.value]:
        MOVE_DICE += BAD
        MOVE_TO_GLOBE += VERY_GOOD
        MOVE_TO_ENEMY_GLOBE += BAD
        MOVE_TO_STAR += BAD
        MOVE_TO_DANGER += BAD
        MOVE_FROM_DANGER += GOOD
        MOVE_TO_END_ZONE += GOOD
        MOVE_TO_GOAL += GOOD
        MOVE_TO_DOUBLE += VERY_GOOD
        MOVE_TO_KILL += VERY_GOOD
        MOVE_TO_SUICIDE += VERY_BAD
        MOVE_TO_HUNT += BAD

    reward_table = [MOVE_DICE, MOVE_FROM_HOME, MOVE_TO_GLOBE, MOVE_TO_ENEMY_GLOBE,
                    MOVE_TO_STAR, MOVE_TO_DANGER, MOVE_FROM_DANGER, MOVE_TO_END_ZONE,
                    MOVE_TO_GOAL, MOVE_TO_DOUBLE, MOVE_TO_KILL, MOVE_TO_SUICIDE, MOVE_TO_HUNT]
    reward = reward_table[best_action]

    # Include future states in the rewards calculation:
    if future_state[State.DANGER.value]:
        reward += VERY_BAD
    if future_state[State.DOUBLE.value]:
        reward += GOOD
    if future_state[State.HUNTING.value]:
        reward += GOOD
    if future_state[State.GLOBE_UNSAFE.value]:
        reward += BAD
    if future_state[State.GLOBE_SAFE.value]:
        reward += GOOD

    return reward


def train_agent(episodes, opponents, epsilon, decay_rate, learning_rate, discount_factor, Q):
    #q_table = np.zeros([NUM_STATES, NUM_ACTIONS])
    #q_table = q_table - 0.5

    # Double learning
    Q_before = Q.copy()

    current_eps = epsilon
    s_cnt = 0
    a_cnt = 0
    a_used = [0]
    s_used = [0]
    ss_used = np.zeros((70, 10))

    # For plotting:
    win_avg = []
    epsilons = []
    idx = []
    wins = 0
    win_rates = []
    cummulated_rewards = []
    q_diff_list = []

    if opponents == 1:
        g = ludopy.Game(ghost_players=[1, 3])
    elif opponents == 2:
        g = ludopy.Game(ghost_players=[3])
    else:
        g = ludopy.Game(ghost_players=[])

    for episode in range(0, episodes):
        g.reset()
        there_is_a_winner = False
        cnt = 0
        cumsum = 0
        sum_diff = 0

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
            #board = g.render_environment()
            # cv2.namedWindow("Ludo board", cv2.WINDOW_NORMAL)
            # board = cv2.resize(board, (1000, 900), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("Ludo Board", board)
            # print("Enemies: ", enemy_pieces)
            # print("player: ", player_pieces)
            # cv2.waitKey(0)
            states = np.zeros((4, 10))
            future_states = np.zeros((4, 10))
            possible_actions = np.zeros((4, 12))
            reward = 0

            if len(move_pieces):
                if player_i == 0:   # idx 0 for the agent
                    states = getStates(player_pieces, move_pieces, enemy_pieces)  # Get the 4 current states
                    future_states = getStates(player_pieces + dice, move_pieces, enemy_pieces)   # Get the 4 possible future states
                    possible_actions = get_possible_actions(states, future_states, move_pieces, player_pieces,
                                                            enemy_pieces, dice)     # Get the possible actions given the movable pieces
                    chosen_piece, best_action = get_action(move_pieces, possible_actions, states, current_eps, Q) # (Qa + Qb)/2)    # future states allready calculated
                    reward = get_reward(chosen_piece, possible_actions, best_action, states, future_states)
                    cumsum += reward

                else:
                    chosen_piece = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                chosen_piece = -1

            _, _, player_pieces_next, enemy_pieces_next, player_is_a_winner, there_is_a_winner = g.answer_observation(chosen_piece)

            if player_i == 0 and chosen_piece != -1:
                # Update Q-table:
                s = binatointeger(states[chosen_piece].astype(int))
                a = best_action
                if s not in s_used:
                    s_used.append(s)

                future_states = getStates(player_pieces_next, [0, 1, 2, 3], enemy_pieces_next)
                q_next = -100.0
                for actions in range(len(player_pieces)):  # since there is 4 different states in each state, possibly change to just look at the chosen pieces' new state?
                    s_new = binatointeger(future_states[actions].astype(int))
                    q_max = np.max(Q[s_new])
                    if q_next < q_max:
                        q_next = q_max
                q_old = Q[s, a]
                # if episode > 300:
                #     print('q_next: ', q_next)
                #     print('disc*q_next', discount_factor*q_next)
                q_diff = learning_rate * (reward + discount_factor * q_next - q_old)
                Q[s, a] = q_old + q_diff

                sum_diff += q_diff

                cnt += 1

        # End of 1 episode
        current_eps = epsilon_decay(epsilon, decay_rate, episode)
        # Plotting stuff:
        epsilons.append(current_eps)
        cummulated_rewards.append(cumsum)
        #q_diff_list.append(sum_diff/cnt)
        q_diff_list.append(np.sum(abs(np.subtract(Q, Q_before))))
        Q_before = Q.copy()#(Qa + Qb)/2
        # diff = np.subtract(aiPlayer1.qLearning.QTable, lastQTable)
        # qTableChange.append(np.sum(np.abs(diff)))

        if g.first_winner_was == 0:
            win_avg.append(1)
            wins = wins + 1
        else:
            win_avg.append(0)

        idx.append(episode)

        win_rate = wins / len(win_avg)
        win_rate_percent = win_rate * 100
        win_rates.append(win_rate_percent)

        if episode % 100 == 0:
            print("Episode: ", episode)
            print(f"Win rate: {np.round(win_rate_percent, 1)}%")
            #print("Used states: ", s_cnt)
            #print("Used actions: ", a_cnt)
    #
    # for stat in s_used:
    #     print("--------------------------------------------------")
    #     print("State: ", bin(stat))
    #     for act in a_used:
    #         print("action: ", bin(act))
    #         print("QValue: ", q_table[stat, act])
    # Moving averages
    # for ss in s_used:
    #
    #     print("s: ", bin(ss))
    #     print("a: ", ((Qa+Qb)/2)[ss])
    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rates, 0, 0))
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    cumsum_vec_change = np.cumsum(np.insert(q_diff_list, 0, 0))
    q_diff_rate_ma = (cumsum_vec_change[window_size:] - cumsum_vec_change[:-window_size]) / window_size

    #cumsum_vec = np.cumsum(np.insert(max_expected_return_list, 0, 0))
    #max_expected_return_list_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    q_diff_rate_ma = moving_average_list + q_diff_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list # + max_expected_return_list_ma.tolist()

    return win_rates, win_rate_ma, epsilons, Q, q_diff_rate_ma

def train_agent_pro(episodes, opponents, epsilon, decay_rate, learning_rate, discount_factor, Q):
    # Double learning
    Qa = Q
    Qb = Q
    Q_opponent = Q
    current_eps = epsilon

    # For plotting:
    win_avg = []
    epsilons = []
    idx = []
    wins = 0
    win_rates = []

    if opponents == 1:
        g = ludopy.Game(ghost_players=[1, 3])
    elif opponents == 2:
        g = ludopy.Game(ghost_players=[3])
    else:
        g = ludopy.Game(ghost_players=[])

    for episode in range(0, episodes):
        g.reset()
        there_is_a_winner = False
        cnt = 0

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
            states = np.zeros((4, 10))
            reward = 0

            if len(move_pieces):
                if player_i == 0:   # idx 0 for the agent
                    states = getStates(player_pieces, move_pieces, enemy_pieces)  # Get the 4 current states
                    future_states = getStates(player_pieces + dice, move_pieces, enemy_pieces)   # Get the 4 possible future states
                    possible_actions = get_possible_actions(states, future_states, move_pieces, player_pieces,
                                                            enemy_pieces, dice)     # Get the possible actions given the movable pieces
                    chosen_piece, best_action = get_action(move_pieces, possible_actions, states, current_eps, (Qa + Qb)/2)    # future states allready calculated
                    reward = get_reward(chosen_piece, possible_actions, best_action, states)
                else:
                    states = getStates(player_pieces, move_pieces, enemy_pieces)  # Get the 4 current states
                    future_states = getStates(player_pieces + dice, move_pieces, enemy_pieces)   # Get the 4 possible future states
                    possible_actions = get_possible_actions(states, future_states, move_pieces, player_pieces,
                                                            enemy_pieces, dice)     # Get the possible actions given the movable pieces
                    chosen_piece = get_action(move_pieces,possible_actions, states, 0, Q_opponent)
            else:
                chosen_piece = -1

            _, _, player_pieces_next, enemy_pieces_next, player_is_a_winner, there_is_a_winner = g.answer_observation(chosen_piece)

            if player_i == 0 and chosen_piece != -1:
                #if player_is_a_winner: # If the agent came in 1st
                #    reward = reward + 100   # 100 points for a win, on top of whatever reward

                s = binatointeger(states[chosen_piece].astype(int))
                a = best_action
                future_states = getStates(player_pieces_next, [0, 1, 2, 3], enemy_pieces_next)
                q_next = -1000
                for actions in range(
                        len(player_pieces)):  # since there is 4 different states in each state, possibly change to just look at the chosen pieces' new state?
                    s_new = binatointeger(future_states[actions].astype(int))
                    if cnt % 2 == 0:
                        q_max = np.max(Qb[s_new])
                    else:
                        q_max = np.max(Qa[s_new])
                    if (q_next < q_max):
                        q_next = q_max

                if cnt % 2 == 0:
                    q_old = Qa[s, a]
                    Qa[s, a] = q_old + learning_rate * (reward + discount_factor * q_next - q_old)
                    print(q_next)
                else:
                    q_old = Qb[s, a]
                    Qb[s, a] = q_old + learning_rate * (reward + discount_factor * q_next - q_old)
                cnt += 1

        # End of 1 episode
        current_eps = epsilon_decay(epsilon, decay_rate, episode)
        # Plotting stuff:
        epsilons.append(current_eps)
        if g.first_winner_was == 0:
            win_avg.append(1)
            wins = wins + 1
        else:
            win_avg.append(0)

        idx.append(episode)
        win_rate = wins / len(win_avg)
        win_rate_percent = win_rate * 100
        win_rates.append(win_rate_percent)

        if episode % 100 == 0:
            print("Episode: ", episode)
            print(f"Win rate: {np.round(win_rate_percent, 1)}%")

    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rates, 0, 0))
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list # + max_expected_return_list_ma.tolist()

    return win_rates, win_rate_ma, epsilons, (Qa+Qb)/2

def test(episodes, opponents, Q):
    # Double learning
    Q_opponent = Q

    chromosome = [0.122144301428050, 0.415098963072078, 0.0663425351356662, 0.0590735947951773, 0.303222350642040,
                  0.0178394449174069, 0.0159276262154338, 0.0776216928169868, 0.190644848040070, 0.0500829975281394,
                  0.0158689779578756, 0.695992583082125, 0.0849567760309734]
    agent = Agent()
    agent.set_chromosome(chromosome)

    # For plotting:
    win_avg = []
    epsilons = []
    idx = []
    wins = 0
    win_rates = []

    if opponents == 1:
        g = ludopy.Game(ghost_players=[1, 3])
    elif opponents == 2:
        g = ludopy.Game(ghost_players=[3])
    else:
        g = ludopy.Game(ghost_players=[])

    for episode in range(0, episodes):
        g.reset()
        there_is_a_winner = False

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if player_i == 0:  # idx 0 for the Q-agent
                    states = getStates(player_pieces, move_pieces, enemy_pieces)  # Get the 4 current states
                    future_states = getStates(player_pieces + dice, move_pieces,
                                              enemy_pieces)  # Get the 4 possible future states
                    possible_actions = get_possible_actions(states, future_states, move_pieces, player_pieces,
                                                            enemy_pieces,
                                                            dice)  # Get the possible actions given the movable pieces
                    chosen_piece, best_action = get_action(move_pieces, possible_actions, states, 0, Q_opponent)  # future states allready calculated
                else:
                    chosen_piece = move_pieces[agent.get_best_action(dice, move_pieces, player_pieces, enemy_pieces)]
            else:
                chosen_piece = -1

            _, _, player_pieces_next, enemy_pieces_next, player_is_a_winner, there_is_a_winner = g.answer_observation(
                chosen_piece)

        # End of 1 episode
        # Plotting stuff:
        if g.first_winner_was == 0:
            win_avg.append(1)
            wins = wins + 1
        else:
            win_avg.append(0)

        idx.append(episode)
        win_rate = wins / len(win_avg)
        win_rate_percent = win_rate * 100
        win_rates.append(win_rate_percent)
        if episode % 100 == 0:
            print("Episode: ", episode)
            print(f"Win rate: {np.round(win_rate_percent, 1)}%")

    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rates, 0, 0))
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list  # + max_expected_return_list_ma.tolist()

    return win_rates, win_rate_ma, epsilons, Q_opponent

def play_against(num_games, num_agents, Q):
    if num_agents == 1:
        g = ludopy.Game(ghost_players=[1, 3])
    elif num_agents == 2:
        g = ludopy.Game(ghost_players=[3])
    else:
        g = ludopy.Game(ghost_players=[])

    g.reset()
    there_is_a_winner = False
    for episode in range(0, num_games):
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()
            board = g.render_environment()
            cv2.namedWindow("Ludo board", cv2.WINDOW_NORMAL)
            board = cv2.resize(board, (800, 800), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Ludo Board", board)
            cv2.waitKey(0)
            if len(move_pieces):
                if player_i == 0:   # Human player
                    # print("Available pieces: ", move_pieces)
                    # print("Piece Locations: ", player_pieces)
                    # _input = -1
                    # while _input == -1:
                    #     _input = int(input('Choose valid input'))
                    #     if _input not in move_pieces:
                    #         _input = -1
                    # chosen_piece = move_pieces[_input]
                    states = getStates(player_pieces, move_pieces, enemy_pieces)  # Get the 4 current states
                    future_states = getStates(player_pieces + dice, move_pieces,
                                              enemy_pieces)  # Get the 4 possible future states
                    possible_actions = get_possible_actions(states, future_states, move_pieces, player_pieces,
                                                            enemy_pieces,
                                                            dice)  # Get the possible actions given the movable pieces
                    chosen_piece, best_action = get_action(move_pieces, possible_actions, states, 0,
                                                           Q)  # future states already calculated
                    pos = []
                    for a in range(0, len(possible_actions[chosen_piece])):
                        if possible_actions[chosen_piece, a]:
                            pos.append(Action(a))

                    print("all enemies: ", enemy_pieces)
                    print("enemy positions: ", (enemy_pieces[1] + 13 * 2) % 52)
                    print("All possible actions: ", possible_actions[chosen_piece])
                    print("AI could choose: ", pos)
                    print("AI chose: ", Action(best_action))
                else:
                    chosen_piece = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                chosen_piece = -1
            _, _, _, _, _, there_is_a_winner = g.answer_observation(chosen_piece)
