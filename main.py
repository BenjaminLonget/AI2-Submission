import numpy as np
from Agent import get_reward, train_agent, test, play_against
import matplotlib.pyplot as plt
from Philip.agent import Agent


NUM_STATES = 2047
NUM_ACTIONS = 13
#NUM_ACTIONS = 8191
learning_rate = 0.1
discount_factor = 0.2
epsilon = 0.9
decay_rate = 0.01
episodes = 1000

Q = np.zeros([NUM_STATES, NUM_ACTIONS])

# win_rate1, ma1, epsilons1, Q1, q_diff_list1 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, 0.0, Q)
# win_rate2, ma1, epsilons1, Q1, q_diff_list2 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, 0.1, Q)
# win_rate3, ma1, epsilons1, Q1, q_diff_list3 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, 0.2, Q)
# win_rate4, ma1, epsilons1, Q1, q_diff_list4 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, 0.4, Q)
# win_rate5, ma1, epsilons1, Q1, q_diff_list5 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, 0.8, Q)

win_rate1, ma1, epsilons1, Q1, q_diff_list1 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, discount_factor, Q)
# win_rate2, ma1, epsilons2, Q2, q_diff_list2 = train_agent(episodes, 2, epsilon, decay_rate, learning_rate, discount_factor, Q)
# win_rate3, ma1, epsilons3, Q3, q_diff_list3 = train_agent(episodes, 3, epsilon, decay_rate, learning_rate, discount_factor, Q)
# win_rate4, ma1, epsilons4, Q1, q_diff_list4 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, discount_factor, Q)
# win_rate5, ma1, epsilons5, Q1, q_diff_list5 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, discount_factor, Q)

# win_rate2, ma1, epsilons1, Q2, q_diff_list1 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, discount_factor, Q1)
# win_rate1, ma1, epsilons1, Q1, q_diff_list1 = train_agent(episodes, 3, epsilon, decay_rate, learning_rate, discount_factor, Q)
# win_rate2, ma4, epsilons4, Q2, q_diff_list4 = train_agent(episodes, 1, 0, decay_rate, 0, discount_factor, Q1)
# win_rate3, ma4, epsilons4, Q2, q_diff_list4 = train_agent(episodes, 2, 0, decay_rate, 0, discount_factor, Q1)
# win_rate4, ma4, epsilons4, Q2, q_diff_list4 = train_agent(episodes, 3, 0, decay_rate, 0, discount_factor, Q1)
# win_rate5, ma5, epsilons5, Q1, q_diff_list5 = train_agent(episodes, 1, epsilon, decay_rate, learning_rate, discount_factor, Q)

#win_rate_pro, cumsum2, epsilons_pro, Q_pro, q_diff_list2 = train_agent(episodes, 1, epsilon / 2, decay_rate, learning_rate, discount_factor / 2, Q1)
# play_against(1, 1, Q1)
win_rate_comparison, win_rate_ma_final, epsilons_final, Q_test = test(1000, 1, Q1)

# Plot win rates against opponents
# fig, axs = plt.subplots(1)
# axs.set_title("Win Rate against random vs trained opponent")
# axs.set_xlabel('Episodes')
# axs.set_ylabel('Win Rate %')
# axs.plot(win_rate, color='tab:red')
# axs.plot(win_rate_pro, color='tab:blue')
# axs.legend(['1 Opponent', '2 Opponents', '3 Opponents'])
# Plot win rates against opponents
# fig, axs = plt.subplots(1)
# axs.set_title("Win Rates against different random, first vs final")
# axs.set_xlabel('Episodes')
# axs.set_ylabel('Win Rate %')
# axs.plot(win_rate, color='tab:red')
# axs.plot(win_rate_final, color='tab:blue')
# axs.legend(['1 Opponent', '2 Opponents', '3 Opponents'])

# Used for the different plots
# gamma: \u03B3
# alpha: \u03B1
fig, axs = plt.subplots(1)
axs.set_title("Comparison with Philip's GA approach")
axs.set_xlabel('Episodes')
axs.set_ylabel('Win-rate')
axs.plot(win_rate_comparison, color='tab:red')
# axs.plot(win_rate2, color='tab:blue')
# axs.plot(win_rate3, color='tab:orange')
# axs.plot(win_rate4, color='tab:green')
# axs.plot(win_rate5, color='tab:brown')
axs.legend(['1 GA opponent '+ str(round(win_rate_comparison[-1],1)) + '%'])
# axs.legend(['1 GA opponent '+ str(round(win_rate1[-1],1)) + '%', '2 opponents '+ str(round(win_rate2[-1],1)) + '%', '3 opponents '+ str(round(win_rate3[-1],1)) + '%'])
# axs.legend(['Decay rate = 0.005, wr: ' + str(round(win_rate1[-1],1)) + '%', 'Decay rate= 0.01, wr: ' + str(round(win_rate2[-1],1)) + '%',
#             'Decay rate = 0.02, wr: ' + str(round(win_rate3[-1],1)) + '%', 'Decay rate = 0.05, wr: ' + str(round(win_rate4[-1],1)) + '%',
#             'Decay rate = 0.1, wr: ' + str(round(win_rate5[-1],1)) + '%'])

# fig, axs = plt.subplots(1)
# axs.set_title("Average change in Q-table")
# axs.set_xlabel('Episodes')
# axs.set_ylabel('Change in Q-values')
# axs.plot(q_diff_list1, color='tab:red')
# axs.plot(q_diff_list2, color='tab:blue')
# axs.plot(q_diff_list3, color='tab:orange')
# # axs.plot(q_diff_list4, color='tab:green')
# # axs.plot(q_diff_list5, color='tab:brown')
# axs.legend(['1 opponent', '2 opponents', '3 opponents'])
# # axs.legend(['Decay rate = 0.005', 'Decay rate = 0.01', 'Decay rate = 0.02', 'Decay rate = 0.05', 'Decay rate = 0.1'])


# Plot epsilons
# fig, axs = plt.subplots(1)
# axs.set_title("Epsilon values")
# axs.set_xlabel('Episodes')
# axs.set_ylabel('Epsilon')
# axs.plot(epsilons1, color='tab:red')
# axs.plot(epsilons2, color='tab:blue')
# axs.plot(epsilons3, color='tab:orange')
# # axs.plot(epsilons4, color='tab:green')
# # axs.plot(epsilons5, color='tab:brown')
# axs.legend(['Decay-rate = 0.005', 'Decay-rate = 0.01', 'Decay-rate = 0.02', 'Decay-rate = 0.05', 'Decay-rate = 0.1'])

plt.show()