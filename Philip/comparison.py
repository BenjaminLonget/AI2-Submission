import sys
sys.path.insert(0,"../") #for importing ludopy in subfolder
import ludopy
import numpy as np
from agent import Agent

# Parameters
numGames = 50
numEnemys = 3 # 1 for 1v1 game (2 player game), 3 for 4 player game
numAgents = 1 

# Setup Philips best agent
# Player trained with following parameters
# Population size: 50
# Games pr. generation: 500
# Mutation rate: 0.1
# Mutation size: 0.5 (gene = gene + random.uniform(-mutationSize, mutationSize))
# Elitism: 10%
# Number of generations: 250

chromosome = [0.122144301428050,	0.415098963072078,	0.0663425351356662,	0.0590735947951773,	0.303222350642040,	0.0178394449174069,	0.0159276262154338,	0.0776216928169868,	0.190644848040070,	0.0500829975281394,	0.0158689779578756,	0.695992583082125,	0.0849567760309734]
agent = Agent()
agent.set_chromosome(chromosome)

# Evaluate agents by playing numGames games
if numEnemys not in [1, 3]:
    raise ValueError("Number of enemys must be 1 or 3")

if numEnemys == 1:
    g= ludopy.Game([1,3])
else:
    g = ludopy.Game()

for i in range(numGames):
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            if player_i == 0:
                piece_to_move = move_pieces[agent.get_best_action(dice, move_pieces, player_pieces, enemy_pieces)]
            # ADD YOUR PLAYER HERE
            #if player_i == 1:
            #    piece_to_move = move_pieces[agent.get_best_action(dice, move_pieces, player_pieces, enemy_pieces)]
            # Random player
            else:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        there_is_a_winner = g.answer_observation(piece_to_move)
    g.reset()

print("Done")