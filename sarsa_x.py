import sys
import numpy as np
from collections import defaultdict


def epsilon_greedy_policy(Q_s, eps, nA):
    policy = np.ones(nA)*eps/nA
    policy[np.argmax(Q_s)] = 1 - eps + eps/nA
    return policy


def sarsa_x(env, algorithm, num_episodes=50000, alpha=0.01, gamma=1.0, save_every=100):

    Q = defaultdict(lambda: np.zeros(env.nA))
    rewards = []
    

    for i_episode in range(1, num_episodes+1):        
        # Monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        # Update epsilon
        eps = 1/i_episode
        
        # Observe state_0
        state = env.reset()
        # Construct policy 0 
        policy = epsilon_greedy_policy(Q[state], eps, env.nA)

        # Choose action_0
        action = np.random.choice(range(env.nA), p=policy)
        reward_tracker = 0
        tmp_rewards =[]
        step = 1
        while True:
            # Take action & observe next state and reward
            next_state, reward, done, info = env.step(action)
            reward_tracker += reward
            
            # Update Policy (epsilon-greedy)
            policy = epsilon_greedy_policy(Q[next_state], eps, env.nA)
                        
            if algorithm == "q_learning":       
                # Update Q-table                     
                Q[state][action] += alpha*(reward + gamma**step*max(Q[next_state]) - Q[state][action]) 
                # Choose next action 
                action = np.random.choice(range(env.nA), p=policy)
                
            elif algorithm == "expected_sarsa":
                # Update Q-table
                Q[state][action] += alpha*(reward + gamma**step*np.dot(policy, Q[next_state]) - Q[state][action])
                # Choose next action
                action = np.random.choice(range(env.nA), p=policy)
                
            elif algorithm == "sarsa_zero":
                # Choose next action 
                next_action = np.random.choice(np.arange(env.nA), p=policy)
                # Update Q-table
                Q[state][action] += alpha*(reward + gamma**step*Q[next_state][next_action] - Q[state][action])
                action = next_action
                
            state = next_state
            step += 1
            if done:
                tmp_rewards.append(reward_tracker)
                break           
        
        if i_episode % save_every == 0:
            rewards.append(np.mean(np.array(tmp_rewards)))
        
    return rewards, Q