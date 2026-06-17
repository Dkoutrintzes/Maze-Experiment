import csv
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean
import json
import pickle
import random
from datetime import date
import sys
import os 
from rl_models.networks_discrete import  ReplayBuffer
# Game utility file
from game.game_utils import get_agent_only_action, get_env_action, get_distance_traveled, get_row_to_store, print_logs, test_print_logs, \
    column_names
# Offline Gradient Updates Scheduler
from game.updates_scheduler import UpdatesScheduler

# Virtual environment
# from maze3D_new.assets import *

# to track memory leaks
from pympler.tracker import SummaryTracker

def load_csv(file_path):
    """
    Load a csv file
    :param file_path: path to the csv file
    :return: the data in the csv file
    """
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

tracker = SummaryTracker()

class Experiment:
    def __init__(self,args, environment, agent=None, load_models=False, config=None, participant_name=None,second_agent=None):
        # retrieve parameters
        self.args = args
        self.config = config  # configuration file dictionary
        self.env = environment  # environment to play in
        if self.config['Experiment']['mode'] != 'human':
            self.agent = agent  # the agent to play with
            if second_agent is not None:
                self.second_agent = second_agent

            if self.config['SAC']['load_checkpoint'] == True:
                self.agent.load_models()
            if self.config['SAC']['load_second_agent'] == True:
                self.second_agent.load_models()
            if self.args.ppr:
                self.second_agent.load_models()
                self.probablistic_policy_reuse = [0.7,0.55,0.4,0.25,0.1,0.05,0.01]

            self.isAgent_discrete = config['SAC']['discrete'] if 'SAC' in config.keys() else None

        # retrieve information from the config file
        self.goal = config["game"]["goal"]
        self.mode = config['Experiment']['mode']
        self.path_to_save = 'results/' + self.mode + '/'+participant_name
        self.participant_name = participant_name
        
        self.max_blocks = config['Experiment'][self.mode]['max_blocks']
        self.action_duration = config['Experiment'][self.mode]['action_duration']
        self.max_game_duration = config['Experiment'][self.mode]['max_duration']
        self.max_blocks = config['Experiment'][self.mode]['max_blocks']
        self.log_interval = self.config['Experiment'][self.mode]['log_interval']
        
        self.second_human = config['game']['second_human'] if 'game' in config.keys() else None
        
        self.train_games_per_block = config['Experiment'][self.mode]['train_games_per_block']
        self.test_games_per_block = config['Experiment'][self.mode]['test_games_per_block']

        self.popup_window_time = config['GUI']['popup_window_time']

        self.best_game_score = 0
        self.last_score = 0
        
        self.block_number_for_rundom = 0
        self.human_actions, self.update_cycles = None, None
        self.save_models, self.flag = True, True
        self.duration_pause_total = 0
        self.current_block = 0

    def save_pickle(self, pt_name, data, baseline = False, game_mode = 'train'):
        
        name_of_file = self.participant_name + '_' + self.mode + '_' + str(date.today()) + '.pickle'
        addintional_part = 0

        if baseline:
            mode = 'baseline'
            while os.path.isfile(os.path.join('results',self.mode,game_mode,name_of_file)):
                name_of_file = self.participant_name + '_' + mode + '_' + str(date.today())+'_'+str(addintional_part)+ '.pickle'
                addintional_part += 1
                
            if not os.path.exists('results'):
                os.makedirs('results')
            if not os.path.exists(os.path.join('results',mode)):
                os.makedirs(os.path.join('results',mode))
            if not os.path.exists(os.path.join('results',mode,game_mode)):
                os.makedirs(os.path.join('results',mode,game_mode))
            
            with open(os.path.join('results',mode,game_mode,name_of_file), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            while os.path.isfile(os.path.join('results',self.mode,game_mode,name_of_file)):
                name_of_file = self.participant_name + '_' + self.mode + '_' + str(date.today())+'_'+str(addintional_part)+ '.pickle'
                addintional_part += 1

            if not os.path.exists('results'):
                os.makedirs('results')
            if not os.path.exists(os.path.join('results',self.mode)):
                os.makedirs(os.path.join('results',self.mode))
            if not os.path.exists(os.path.join('results',self.mode,game_mode)):
                os.makedirs(os.path.join('results',self.mode,game_mode))

            with open(os.path.join('results',self.mode,game_mode,name_of_file), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def mz_experiment(self,participant_name):

        train_block_metrics_dict = {}
        test_block_metrics_dict = {}
        #self.maze_game_random_agent(0,10)
        for i_block in range(self.max_blocks):
            self.current_block = i_block
            if self.config['Experiment'][self.mode]['test_block'] == True:
                print("Test Block: ", i_block)
                test_block_metrics_dict['block_'+str(i_block)] = {}
                test_block_metrics_dict = self.maze_game(test_block_metrics_dict,'block_'+str(i_block),int(self.test_games_per_block), i_block,'test')

            if self.config['Experiment'][self.mode]['training_block'] == True:
                print("Train Block: ", i_block)
                train_block_metrics_dict['block_'+str(i_block)] = {}
                train_block_metrics_dict = self.maze_game(train_block_metrics_dict,'block_'+str(i_block),int(self.train_games_per_block), i_block,'train')

        # Final Testing block
        if self.config['Experiment'][self.mode]['extra_test_block'] == True:
            print("Test Block: ", self.max_blocks)
            test_block_metrics_dict['block_'+str(self.max_blocks)] = {}
            test_block_metrics_dict = self.maze_game(test_block_metrics_dict,'block_'+str(self.max_blocks),int(self.test_games_per_block), self.max_blocks,'test')
        

        # Save to pickle file
        self.save_pickle(self.participant_name, train_block_metrics_dict, game_mode = 'train')
        self.save_pickle(self.participant_name, test_block_metrics_dict, game_mode = 'test')


        if not os.path.exists(os.path.join('results',self.mode,'buffer',self.participant_name)):
            os.makedirs(os.path.join('results',self.mode,'buffer',self.participant_name))
        
        file_name = 'buffer.npy'
        c = 1
        while os.path.isfile(os.path.join('results',self.mode,'buffer',self.participant_name,file_name)):
            file_name = 'buffer_'+str(c)+'.npy'
            c += 1
        
        self.agent.memory.save_buffer(os.path.join('results',self.mode,'buffer',self.participant_name),file_name )

    def mz_two_agents(self,participant_name):
        train_block_metrics_dict = {}
        test_block_metrics_dict = {}
        for i_block in range(self.max_blocks):
            print("Test Block: ", i_block)
            test_block_metrics_dict['block_'+str(i_block)] = {}
            test_block_metrics_dict = self.maze_two_agents(test_block_metrics_dict,'block_'+str(i_block),int(self.test_games_per_block), i_block,'test')
            print("Train Block: ", i_block)
            train_block_metrics_dict['block_'+str(i_block)] = {}
            train_block_metrics_dict = self.maze_two_agents(train_block_metrics_dict,'block_'+str(i_block),int(self.train_games_per_block), i_block,'train')

        # Final Testing block
        print("Test Block: ", self.max_blocks)
        test_block_metrics_dict['block_'+str(self.max_blocks)] = {}
        test_block_metrics_dict = self.maze_game(test_block_metrics_dict,'block_'+str(self.max_blocks),int(self.test_games_per_block), self.max_blocks,'test')
        
        # Save to pickle file
        self.save_pickle(self.participant_name, train_block_metrics_dict, game_mode = 'train')
        self.save_pickle(self.participant_name, test_block_metrics_dict, game_mode = 'test')

        if not os.path.exists(os.path.join('results',self.mode,'buffer',self.participant_name)):
            os.makedirs(os.path.join('results',self.mode,'buffer',self.participant_name))
        
        file_name = 'buffer.npy'
        c = 1
        while os.path.isfile(os.path.join('results',self.mode,'buffer',self.participant_name,file_name)):
            file_name = 'buffer_'+str(c)+'.npy'
            c += 1
        
        self.agent.memory.save_buffer(os.path.join('results',self.mode,'buffer',self.participant_name),file_name )

    def mz_only_agent(self,participant_name):
        block_metrics_dict = {}
        #self.maze_game_random_agent(0,10)
        for i_block in range(self.max_blocks):

            block_metrics_dict['block_'+str(i_block)] = {}
            block_metrics_dict = self.maze_only_agent(block_metrics_dict,'block_'+str(i_block),self.train_games_per_block, i_block)

        # Save to pickle file
        self.save_pickle(self.participant_name, block_metrics_dict)

    def mz_eval(self,participant_name):
        block_metrics_dict = {}
        block_metrics_dict['eval_block'] = {} 
        block_metrics_dict = self.maze_game(block_metrics_dict,'eval_block',self.test_games_per_block, 0,'test')

        # Save to pickle file
        self.save_pickle(self.participant_name, block_metrics_dict)

    def human_play(self,participant_name):
        self.human_replay_buffer = ReplayBuffer(self.args)

        block_metrics_dict = {}

        for i_block in range(self.max_blocks):
            block_metrics_dict['block_'+str(i_block)] = {} 
            block_metrics_dict = self.maze_only_human(block_metrics_dict,'block_'+str(i_block),self.train_games_per_block, i_block,'train')

        # Save to pickle file
        self.save_pickle(self.participant_name, block_metrics_dict)

        file_name = 'buffer.npy'
        c = 1
        while os.path.isfile(os.path.join('results',self.mode,'buffer',self.participant_name,file_name)):
            file_name = 'buffer_'+str(c)+'.npy'
            c += 1
        
        if not os.path.exists(os.path.join('results',self.mode,'buffer')):
            os.makedirs(os.path.join('results',self.mode,'buffer'))
        if not os.path.exists(os.path.join('results',self.mode,'buffer',self.participant_name)):
            os.makedirs(os.path.join('results',self.mode,'buffer',self.participant_name))
        self.human_replay_buffer.save_buffer(os.path.join('results',self.mode,'buffer',self.participant_name),file_name )

    def maze_game(self,block_metrics_dict,block_name,max_games,block_number,game_mode):
        fps_history = []
        action_history = []
        
        block_distance_travel_history = []
        agent_states = []
        step_duration_history = []
        game_duration_history = []
        step_counter_history = []

        rewards_history = []
        state_history = []
        game_score_history = []
        done_history = []
        # Starting the Block of games for each Training/Testing period
        train_game_success_counter = 0
        for i_game in range(max_games):
            game_actions = []
            game_fps = []
            is_on_pause = True
            while is_on_pause:
                print('Game Reseting')
                prev_observation, setting_up_duration, is_on_pause = self.env.reset(game_mode)  # stores the state of the environment
                norm_prev_observation = self.normalize_state(prev_observation)
            timed_out = False  # used to check if we hit the maximum train_game_number duration
            game_reward = 200  # keeps track of the rewards for each train_game_number
            distance_travel = 0  # keeps track of the ball's travelled distance

            game_reward_history = []
            game_state_history = []

            
            print('Episode: ' + str(i_game))

            redundant_end_duration = 0  # duration in the game that is not playable by the user
            step_counter = 0

            game_agent_states = []
            step_duration_game = []
            game_done = []
            # Playing the Maze game
            start_game_time = time.time()

            while True:
                step_counter += 1  # keep track of the step number for each game
                # print("train_step_counter {}".format(train_step_counter))
                
                env_agent_action, real_agent_action = self.get_agent_action(norm_prev_observation, 'agent',game_mode)
                env_agent_action = int(env_agent_action)
                #print('Agent Action:',env_agent_action,type(env_agent_action))
                # check if the game has timed out
                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                # Environment step
                md = str(i_game+1) + ' / ' + str(max_games) + ' -- ' + 'Last Score: ' + str(self.last_score) + ' -- ' + 'Best Score: ' + str(self.best_game_score)
                transition = self.env.step(env_agent_action, timed_out, self.action_duration, mode='one_agent',text = md)
                #print('Enviorment step')
                observation, reward, done, fps, duration_pause, action_pair, internet_delay = transition
                norm_observation = self.normalize_state(observation)
                redundant_end_duration += duration_pause  # keep track of the total paused time
                #print(reward,timed_out)
                # keep track of the fps
                game_fps.append(fps)
                
                # tracking player/agent actions
                #print(observation)
                game_actions.append(action_pair)
                
                # add experience to buffer
                transition_info = [block_number,i_game,step_counter]

                interaction = [norm_prev_observation, [real_agent_action], reward, norm_observation, done,transition_info ]
                if game_mode == 'train':
                    self.save_experience(interaction,block_number)

                # keep track of the rewards
                game_reward_history.append(reward)
                game_state_history.append(observation)
                game_done.append(done)

                game_reward += reward  # keep track of the total game reward

                distance_travel = get_distance_traveled(distance_travel, prev_observation, observation)

                new_row = {'prev_observation': prev_observation, 
                            'real_agent_action': real_agent_action,
                            'env_agent_action': env_agent_action, 
                            'human_action': action_pair[1], 
                            'observation': observation, 
                            'reward': reward}
                                                
                game_agent_states.append(new_row)

                # calculate game duration
                step_duration = time.time() - start_game_time - duration_pause
                step_duration_game.append(step_duration)
                
                # set the observation for the next step
                norm_prev_observation = norm_observation
                prev_observation = observation

                if done:
                    redundant_end_duration += self.popup_window_time

                    # # goal is reached
                    if not timed_out:
                        train_game_success_counter += 1
                    else:
                        game_reward = 0
                    time.sleep(self.popup_window_time)
                    break
                #print('After break')
            
            print(done,game_reward)
            self.last_score = game_reward
            if game_reward > self.best_game_score:
                self.best_game_score = game_reward
            
            done_history.append(game_done)
            step_duration_history.append(step_duration_game)
            block_distance_travel_history.append(distance_travel)
            agent_states.append(game_agent_states)
            fps_history.append(game_fps)
            action_history.append(game_actions)
            rewards_history.append(game_reward_history)
            state_history.append(game_state_history)
            game_score_history.append(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            game_duration = end_game_time - start_game_time - redundant_end_duration
            game_duration_history.append(game_duration)

            step_counter_history.append(step_counter)

            if self.agent.can_learn(block_number+1):
                if game_mode == 'train':
                    print('Start Offline Gradient Updates Session')
                    self.offline_grad_updates_session(i_game, block_number)
                    update_history = self.agent.collect_data()
                    block_metrics_dict[block_name]['update_history_' + str(i_game)] = update_history
            


        block_metrics_dict[block_name]['fps'] = fps_history
        block_metrics_dict[block_name]['actions'] = action_history
        block_metrics_dict[block_name]['distance_travel'] = block_distance_travel_history
        block_metrics_dict[block_name]['agent_states'] = agent_states
        block_metrics_dict[block_name]['step_duration'] = step_duration_history
        block_metrics_dict[block_name]['game_duration'] = game_duration_history
        block_metrics_dict[block_name]['step_counter'] = step_counter_history
        block_metrics_dict[block_name]['rewards'] = rewards_history
        block_metrics_dict[block_name]['states'] = state_history
        block_metrics_dict[block_name]['game_score'] = game_score_history
        block_metrics_dict[block_name]['win_loss'] = [train_game_success_counter,max_games-train_game_success_counter]
        block_metrics_dict[block_name]['done'] = done_history

        # if game_mode == 'train':
        #     print('Start Offline Gradient Updates Session')
        #     self.offline_grad_updates_session(i_game, block_number)
        #     update_history = self.agent.collect_data()
        #     block_metrics_dict[block_name]['update_history'] = update_history
        
        

        return block_metrics_dict

    def maze_two_agents(self,block_metrics_dict,block_name,max_games,block_number,game_mode):
        fps_history = []
        action_history = []
        
        block_distance_travel_history = []
        agent_states = []
        step_duration_history = []
        game_duration_history = []
        step_counter_history = []

        rewards_history = []
        state_history = []
        game_score_history = []
        done_history = []
        # Starting the Block of games for each Training/Testing period
        train_game_success_counter = 0
        for i_game in range(max_games):
            game_actions = []
            game_fps = []
            is_on_pause = True
            while is_on_pause:
                print('Game Reseting')
                prev_observation, setting_up_duration, is_on_pause = self.env.reset(game_mode)  # stores the state of the environment
                norm_prev_observation = self.normalize_state(prev_observation)
            timed_out = False  # used to check if we hit the maximum train_game_number duration
            game_reward = 200  # keeps track of the rewards for each train_game_number
            distance_travel = 0  # keeps track of the ball's travelled distance

            game_reward_history = []
            game_state_history = []

            
            print('Episode: ' + str(i_game))

            redundant_end_duration = 0  # duration in the game that is not playable by the user
            step_counter = 0

            game_agent_states = []
            step_duration_game = []
            game_done = []
            # Playing the Maze game
            start_game_time = time.time()

            while True:
                step_counter += 1  # keep track of the step number for each game
                # print("train_step_counter {}".format(train_step_counter))

                env_agent_action, real_agent_action = self.get_agent_action(norm_prev_observation, 'agent', game_mode)
                env_sec_agent_action, real_sec_agent_action = self.get_agent_action(norm_prev_observation, 'second_agent', game_mode)

                # check if the game has timed out
                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                # Environment step
                md = str(i_game+1) + ' / ' + str(max_games) + ' -- ' + 'Last Score: ' + str(self.last_score) + ' -- ' + 'Best Score: ' + str(self.best_game_score)
                transition = self.env.step([env_agent_action,env_sec_agent_action], timed_out, self.action_duration, mode='two_agents',text = md)
                observation, reward, done, fps, duration_pause, action_pair, internet_delay = transition
                norm_observation = self.normalize_state(observation)
                redundant_end_duration += duration_pause  # keep track of the total paused time
                #print(reward,timed_out)
                # keep track of the fps
                game_fps.append(fps)
                
                # tracking player/agent actions
                #print(observation)
                game_actions.append(action_pair)
                
                # add experience to buffer
                transition_info = [block_number,i_game,step_counter]
                interaction = [norm_prev_observation, [real_agent_action,real_sec_agent_action], reward, norm_observation, done,transition_info ]
                if game_mode == 'train':
                    self.save_experience(interaction,block_number)

                # keep track of the rewards
                game_reward_history.append(reward)
                game_state_history.append(observation)
                game_done.append(done)

                game_reward += reward  # keep track of the total game reward

                distance_travel = get_distance_traveled(distance_travel, prev_observation, observation)

                new_row = {'prev_observation': prev_observation,
                            'real_agent_action': real_agent_action,
                            'real_sec_agent_action': real_sec_agent_action,
                            'env_agent_action': env_agent_action,
                            'env_sec_agent_action': env_sec_agent_action,
                            'human_action': action_pair[1],
                            'observation': observation,
                            'reward': reward}

                                                
                game_agent_states.append(new_row)

                # calculate game duration
                step_duration = time.time() - start_game_time - duration_pause
                step_duration_game.append(step_duration)
                
                # set the observation for the next step
                norm_prev_observation = norm_observation
                prev_observation = observation

                if done:
                    redundant_end_duration += self.popup_window_time

                    # # goal is reached
                    if not timed_out:
                        train_game_success_counter += 1
                    else:
                        game_reward = 0
                    time.sleep(self.popup_window_time)
                    break
                #print('After break')
            
            print(done,game_reward)
            self.last_score = game_reward
            if game_reward > self.best_game_score:
                self.best_game_score = game_reward
            
            done_history.append(game_done)
            step_duration_history.append(step_duration_game)
            block_distance_travel_history.append(distance_travel)
            agent_states.append(game_agent_states)
            fps_history.append(game_fps)
            action_history.append(game_actions)
            rewards_history.append(game_reward_history)
            state_history.append(game_state_history)
            game_score_history.append(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            game_duration = end_game_time - start_game_time - redundant_end_duration
            game_duration_history.append(game_duration)

            step_counter_history.append(step_counter)

            if i_game >= 2 or block_number >= 1:
                block_metrics_dict[block_name]['update_history_' + str(i_game)] = {}
                if game_mode == 'train':
                    print('Start Offline Gradient Updates Session')
                    self.offline_grad_updates_session(i_game, block_number)
                    update_history = self.agent.collect_data()
                    update_history = self.second_agent.collect_data()
                    block_metrics_dict[block_name]['update_history_' + str(i_game)]['agent'] = update_history
                    block_metrics_dict[block_name]['update_history_' + str(i_game)]['second_agent'] = update_history

            
        
            # 2. Offline gradient updates session
            

            # running_reward, avg_length = print_logs(self.config["game"]["verbose"],
            #                                         self.config['game']['test_model'],
            #                                         self.train_total_steps, i_game, self.train_total_steps,
            #                                         running_reward,
            #                                         avg_length, self.log_interval, avg_game_duration)
        #print('Training Session Ended')

        block_metrics_dict[block_name]['fps'] = fps_history
        block_metrics_dict[block_name]['actions'] = action_history
        block_metrics_dict[block_name]['distance_travel'] = block_distance_travel_history
        block_metrics_dict[block_name]['agent_states'] = agent_states
        block_metrics_dict[block_name]['step_duration'] = step_duration_history
        block_metrics_dict[block_name]['game_duration'] = game_duration_history
        block_metrics_dict[block_name]['step_counter'] = step_counter_history
        block_metrics_dict[block_name]['rewards'] = rewards_history
        block_metrics_dict[block_name]['states'] = state_history
        block_metrics_dict[block_name]['game_score'] = game_score_history
        block_metrics_dict[block_name]['win_loss'] = [train_game_success_counter,max_games-train_game_success_counter]
        block_metrics_dict[block_name]['done'] = done_history

        # if game_mode == 'train':
        #     print('Start Offline Gradient Updates Session')
        #     self.offline_grad_updates_session(i_game, block_number)
        #     update_history = self.agent.collect_data()
        #     block_metrics_dict[block_name]['update_history'] = update_history
        
        return block_metrics_dict

    def maze_only_agent(self,block_metrics_dict,block_name,max_games,block_number,game_mode):
        fps_history = []
        action_history = []
        self.block_number_for_rundom = block_number
        print('Block Number:',block_number)
        block_distance_travel_history = []
        agent_states = []
        step_duration_history = []
        game_duration_history = []
        step_counter_history = []

        rewards_history = []
        state_history = []
        game_score_history = []
        done_history = []
        # Starting the Block of games for each Training/Testing period
        
        train_game_success_counter = 0
        for i_game in range(max_games):
            #vector = []
            game_actions = []
            game_fps = []
            is_on_pause = True
            while is_on_pause:
                print('Game Reseting')
                prev_observation, setting_up_duration, is_on_pause = self.env.reset()  # stores the state of the environment
                norm_prev_observation = self.normalize_state(prev_observation)
                # for i in range(5):
                #     vector.append(prev_observation)
            timed_out = False  # used to check if we hit the maximum train_game_number duration
            game_reward = 200  # keeps track of the rewards for each train_game_number
            distance_travel = 0  # keeps track of the ball's travelled distance

            game_reward_history = []
            game_state_history = []

            
            print('Episode: ' + str(i_game))

            redundant_end_duration = 0  # duration in the game that is not playable by the user
            step_counter = 0

            game_agent_states = []
            step_duration_game = []
            game_done = []
            # Playing the Maze game
            start_game_time = time.time()
            # vector = np.array(vector)
            while True:
                step_counter += 1  # keep track of the step number for each game
                # print("train_step_counter {}".format(train_step_counter))

                env_agent_action, real_agent_action = self.get_agent_action(norm_prev_observation, 'only_agent')

                #env_sec_agent_action, real_sec_agent_action = self.get_agent_action(prev_observation, 'second_agent')

                # check if the game has timed out
                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                # Environment step
                md = str(i_game+1) + ' / ' + str(max_games) + ' -- ' + 'Last Score: ' + str(self.last_score) + ' -- ' + 'Best Score: ' + str(self.best_game_score)
                transition = self.env.step(env_agent_action, timed_out, self.action_duration, mode='two_agents',text = md)
                observation, reward, done, fps, duration_pause, action_pair, internet_delay = transition
                norm_observation = self.normalize_state(observation)
                #new_vector = vector[1:] 
                #new_vector = np.append(new_vector, [observation], axis=0)
                #print(new_vector.shape)
                redundant_end_duration += duration_pause  # keep track of the total paused time
                #print(reward,timed_out)
                # keep track of the fps
                game_fps.append(fps)
                
                # tracking player/agent actions
                #print(observation)
                game_actions.append(action_pair)
                
                # add experience to buffer
                transition_info = [block_number,i_game,step_counter]

                interaction = [norm_prev_observation, [real_agent_action], reward, norm_observation, done,transition_info ]
                if game_mode == 'train':
                    self.save_experience(interaction,block_number)

                # keep track of the rewards
                game_reward_history.append(reward)
                game_state_history.append(observation)
                game_done.append(done)

                game_reward += reward  # keep track of the total game reward

                distance_travel = get_distance_traveled(distance_travel, prev_observation, observation)

                new_row = {'prev_observation': prev_observation,
                            'real_agent_action': real_agent_action,
                            'env_agent_action': env_agent_action,
                            'human_action': action_pair[1],
                            'observation': observation,
                            'reward': reward}
                                                
                game_agent_states.append(new_row)

                # calculate game duration
                step_duration = time.time() - start_game_time - duration_pause
                step_duration_game.append(step_duration)
                
                # set the observation for the next step
                norm_prev_observation = norm_observation

                if done:
                    redundant_end_duration += self.popup_window_time

                    # # goal is reached
                    if not timed_out:
                        train_game_success_counter += 1
                    else:
                        game_reward = 0
                    time.sleep(self.popup_window_time)
                    break
                #print('After break')
            
            print(done,game_reward)
            self.last_score = game_reward
            if game_reward > self.best_game_score:
                self.best_game_score = game_reward
            
            done_history.append(game_done)
            step_duration_history.append(step_duration_game)
            block_distance_travel_history.append(distance_travel)
            agent_states.append(game_agent_states)
            fps_history.append(game_fps)
            action_history.append(game_actions)
            rewards_history.append(game_reward_history)
            state_history.append(game_state_history)
            game_score_history.append(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            game_duration = end_game_time - start_game_time - redundant_end_duration
            game_duration_history.append(game_duration)

            step_counter_history.append(step_counter)
        
            # 2. Offline gradient updates session
            

            # running_reward, avg_length = print_logs(self.config["game"]["verbose"],
            #                                         self.config['game']['test_model'],
            #                                         self.train_total_steps, i_game, self.train_total_steps,
            #                                         running_reward,
            #                                         avg_length, self.log_interval, avg_game_duration)
        #print('Training Session Ended')

        block_metrics_dict[block_name]['fps'] = fps_history
        block_metrics_dict[block_name]['actions'] = action_history
        block_metrics_dict[block_name]['distance_travel'] = block_distance_travel_history
        block_metrics_dict[block_name]['agent_states'] = agent_states
        block_metrics_dict[block_name]['step_duration'] = step_duration_history
        block_metrics_dict[block_name]['game_duration'] = game_duration_history
        block_metrics_dict[block_name]['step_counter'] = step_counter_history
        block_metrics_dict[block_name]['rewards'] = rewards_history
        block_metrics_dict[block_name]['states'] = state_history
        block_metrics_dict[block_name]['game_score'] = game_score_history
        block_metrics_dict[block_name]['win_loss'] = [train_game_success_counter,max_games-train_game_success_counter]
        block_metrics_dict[block_name]['done'] = done_history

        if game_mode == 'train':
            print('Start Offline Gradient Updates Session')
            self.offline_grad_updates_session(i_game, block_number)
            update_history = self.agent.collect_data()
            block_metrics_dict[block_name]['update_history'] = update_history
        
        return block_metrics_dict
    
    def maze_only_human(self,block_metrics_dict,block_name,max_games,block_number,game_mode):
        fps_history = []
        action_history = []
        
        block_distance_travel_history = []
        agent_states = []
        step_duration_history = []
        game_duration_history = []
        step_counter_history = []

        rewards_history = []
        state_history = []
        game_score_history = []
        done_history = []
        # Starting the Block of games for each Training/Testing period
        
        train_game_success_counter = 0
        for i_game in range(max_games):
            game_actions = []
            game_fps = []
            is_on_pause = True
            while is_on_pause:
                print('Game Reseting')
                prev_observation, setting_up_duration, is_on_pause = self.env.reset(game_mode)
                norm_prev_observation = self.normalize_state(prev_observation)
            timed_out = False
            game_reward = 200
            distance_travel = 0

            game_reward_history = []
            game_state_history = []


            print('Episode: ' + str(i_game))

            redundant_end_duration = 0
            step_counter = 0

            game_agent_states = []
            step_duration_game = []
            game_done = []
            # Playing the Maze game
            start_game_time = time.time()

            while True:
                step_counter += 1

                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                md = str(i_game+1) + ' / ' + str(max_games) + ' -- ' + 'Last Score: ' + str(self.last_score) + ' -- ' + 'Best Score: ' + str(self.best_game_score)
                transition = self.env.step(None, timed_out, self.action_duration, mode='human',text = md)
                observation, reward, done, fps, duration_pause, action_pair, internet_delay = transition
                norm_observation = self.normalize_state(observation)

                redundant_end_duration += duration_pause

                game_fps.append(fps)
                
                game_actions.append(action_pair)

                # keep track of the rewards
                game_reward_history.append(reward)
                game_state_history.append(observation)
                game_done.append(done)

                game_reward += reward  # keep track of the total game reward

                distance_travel = get_distance_traveled(distance_travel, prev_observation, observation)

                new_row = {'prev_observation': prev_observation, 
                            'human_action_1': action_pair[0], 
                            'human_action_2': action_pair[1], 
                            'observation': observation, 
                            'reward': reward}
                                                
                game_agent_states.append(new_row)
                self.human_replay_buffer.add(observation, action_pair[0], reward, observation, done, 'human_'+str(block_name)+'_'+str(i_game))

                # calculate game duration
                step_duration = time.time() - start_game_time - duration_pause
                step_duration_game.append(step_duration)
                
                # set the observation for the next step
                norm_prev_observation = norm_observation
                prev_observation = observation

                if done:
                    redundant_end_duration += self.popup_window_time

                    # # goal is reached
                    if not timed_out:
                        train_game_success_counter += 1
                    else:
                        game_reward = 0
                    time.sleep(self.popup_window_time)
                    break
                #print('After break')
            
            print(done,game_reward)
            self.last_score = game_reward
            if game_reward > self.best_game_score:
                self.best_game_score = game_reward
            
            done_history.append(game_done)
            step_duration_history.append(step_duration_game)
            block_distance_travel_history.append(distance_travel)
            agent_states.append(game_agent_states)
            fps_history.append(game_fps)
            action_history.append(game_actions)
            rewards_history.append(game_reward_history)
            state_history.append(game_state_history)
            game_score_history.append(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            game_duration = end_game_time - start_game_time - redundant_end_duration
            game_duration_history.append(game_duration)

            step_counter_history.append(step_counter)

        


        block_metrics_dict[block_name]['fps'] = fps_history
        block_metrics_dict[block_name]['actions'] = action_history
        block_metrics_dict[block_name]['distance_travel'] = block_distance_travel_history
        block_metrics_dict[block_name]['agent_states'] = agent_states
        block_metrics_dict[block_name]['step_duration'] = step_duration_history
        block_metrics_dict[block_name]['game_duration'] = game_duration_history
        block_metrics_dict[block_name]['step_counter'] = step_counter_history
        block_metrics_dict[block_name]['rewards'] = rewards_history
        block_metrics_dict[block_name]['states'] = state_history
        block_metrics_dict[block_name]['game_score'] = game_score_history
        block_metrics_dict[block_name]['win_loss'] = [train_game_success_counter,max_games-train_game_success_counter]
        block_metrics_dict[block_name]['done'] = done_history

        
        

        return block_metrics_dict

    def save_experience(self, interaction,block_number):
        """
        Saves an interaction (prev_observation, agent_action, reward, observation, done) to the replay buffer of
        the agent.
        :param interaction: the interaction to be stored in the Replay Buffer
        """
        if self.config['Experiment']['agent'] == 'sac':
            observation, agent_action, reward, observation_, done, transition_info = interaction

            # we play with the RL agent
            if not self.second_human:
                self.agent.memory.add(observation, agent_action[0], reward, observation_, done, transition_info)

                if self.mode == 'no_tl_two_agents':
                    self.second_agent.memory.add(observation, agent_action[1], reward, observation_,  done, transition_info)

    def normalize_state(self,observation):
        norm_observation = [0]*len(observation)
        # x,y from -2/2 to -1/1
        norm_observation[0] = self.norm_feature(observation[0],-2,2)
        norm_observation[1] = self.norm_feature(observation[1],-2,2)
        # x,y velocity from -0/2 to 0/1
        norm_observation[2] = self.norm_feature(observation[2],-4,4)
        norm_observation[3] = self.norm_feature(observation[3],-4,4)
        # f,t from -30/30 to -1/1
        norm_observation[4] = self.norm_feature(observation[4],-30,30)
        norm_observation[5] = self.norm_feature(observation[5],-30,30)
        # # f,t velocity from -1/1 to -1/1
        norm_observation[6] = self.norm_feature(observation[6],-1.9,1.9)
        norm_observation[7] = self.norm_feature(observation[7],-1.9,1.9)

        for i in range(len(norm_observation)):
            if norm_observation[i] > 1.3:
                norm_observation[i] = 1.3
            elif norm_observation[i] < -1.3:
                norm_observation[i] = -1.3

        # print(norm_observation)
        # print("-----------------")
        # for i in range(len(observation)):
        #     if norm_observation[i] > 1.1 or norm_observation[i] < -1.1:
        #             print("ERROR")
        #             print('Feature:',i,'Value:',float(observation[i]))
        return np.array(norm_observation)

    def norm_feature(self, feature,min,max):
        norm_feature = 2*((feature - min)/(max - min)) - 1
        return norm_feature

    def get_agent_action(self, prev_observation, enemy_status=None,game_mode='train'):
        """
        Retrieves the original action from the agent and converts it into an environment-compatible one.
        Especially discrete SAC predicts actions using a categorical distribution, so we have to map these actions into
        both negative and positive numbers.
        :param prev_observation:
        :param randomness_criterion:
        :return: the environment_compatible agent's action, the original agent's action
        """
        # if playing with the agent and not with another human

        if not self.second_human and enemy_status == 'agent':
            agent_action,argmax_action = self.compute_agent_action(prev_observation,enemy_status,game_mode)
            if game_mode == 'train':
                env_agent_action = get_env_action(agent_action, self.isAgent_discrete)
            elif game_mode == 'test':
                env_agent_action = get_env_action(argmax_action, self.isAgent_discrete)
                agent_action = argmax_action

        elif not self.second_human and enemy_status == 'second_agent':
            # compute agent's action
            agent_action,argmax_action = self.compute_agent_action(prev_observation,enemy_status,game_mode)
            if game_mode == 'train':
                env_agent_action = get_env_action(agent_action, self.isAgent_discrete)
            elif game_mode == 'test':
                env_agent_action = get_env_action(argmax_action, self.isAgent_discrete)
                agent_action = argmax_action

        elif not self.second_human and enemy_status == 'random':
            #print('Here again')
            # choices available to the random agent
            choices = [0, 1, 2]
            # compute random agent's action
            agent_action = int(np.random.choice(choices))
            env_agent_action = get_env_action(agent_action, self.isAgent_discrete)

        elif not self.second_human and enemy_status == 'only_agent':
            agent_action,argmax_action = self.compute_agent_action(prev_observation,'agent',game_mode)
            if game_mode == 'train':
                multi_actions = get_agent_only_action(agent_action)
            elif game_mode == 'test':
                multi_actions = get_agent_only_action(argmax_action)
                agent_action = argmax_action
            env_agent_action = [get_env_action(multi_actions[0], self.isAgent_discrete),get_env_action(multi_actions[1], self.isAgent_discrete)]

        else:
            # get second human's action
            env_agent_action = None
            agent_action = None

        return env_agent_action, agent_action
    
    def compute_agent_action(self, observation,status,game_mode):

        if status == 'agent':
            agent_action = self.agent.actor.sample_act(observation)  
            if self.args.ppr and game_mode == 'train':
                expert_action = self.second_agent.actor.sample_act(observation)
                random_chance = random.random()
                #print('Random Chance:',random_chance)
                if random_chance < self.probablistic_policy_reuse[self.current_block]:
                    agent_action = expert_action  

            
            # qvalue = self.agent.critic.sample_qvalue(observation)
            # print(qvalue)
            return agent_action
        elif status == 'second_agent':
            agent_action = self.second_agent.actor.sample_act(observation)
            return agent_action

        return agent_action

    def offline_grad_updates_session(self, i_game, block_number):
        """
        Perform a number of offline gradient updates.
        :param i_game: current game
        :return:
        """
        start_time = time.time()
        print("off policy learning.")
        # get the number of cycles
        #if self.mode == 'train_agent':
        self.update_cycles = self.config['Experiment'][self.mode]['updates_per_ogu']
        #else:
        #    self.update_cycles = -1 #int(self.config['Experiment'][self.mode]['total_update_cycles']/(self.config['Experiment'][self.mode]['max_blocks'] - 1))

        if self.update_cycles > 0:

            grad_updates_duration = self.grad_updates(self.update_cycles, block_number+1)

            print('Saving model')
            self.agent.save_models(block_number)
            if self.mode == 'no_tl_two_agents':
                print('Saving second model')
                self.second_agent.save_models(block_number)

        time_to_learn = time.time() - start_time

        return grad_updates_duration, time_to_learn

    def grad_updates(self, update_cycles, block_number):
        """
        Performs a number of offline gradient updates on the agent.
        :param update_cycles: the number of gradient updates to perform on the agent
        :return: the duration in sec of the gradient updates performed
        """
        start_grad_updates = time.time()
        end_grad_updates = 0
        misc_duration = 0

        # we play with the RL agent
        if not self.second_human:
            print("Performing {} updates".format(update_cycles))
            # print a completion bar in the terminal
            for cycle_i in tqdm(range(update_cycles),file=sys.stdout):
                    if self.config['SAC']['freeze_agent'] == False:
                        policy_loss,q1_loss,q2_loss,alpha_temp = self.agent.learn(block_number)
                        if cycle_i % 100 == 0:
                            print('Cycle:',cycle_i,'Policy Loss:',policy_loss,'Q1 Loss:',q1_loss,'Q2 Loss:',q2_loss,'Alpha:',alpha_temp)
                    
                    if self.mode == 'no_tl_two_agents':
                        if self.config['SAC']['freeze_second_agent'] == False:
                            policy_loss,q1_loss,q2_loss,alpha_temp = self.second_agent.learn(block_number)

                    # # update the target networks
                    # self.agent.soft_update_target()
                    # if self.mode == 'no_tl_two_agents':
                    #     self.second_agent.soft_update_target()

            end_grad_updates = time.time()

        return end_grad_updates - start_grad_updates - misc_duration
    
    def test_buffer(self,grand_updates):
        self.grad_updates(grand_updates, 0)

    def pre_train_agent(self,grand_updates):
        if not self.second_human:
            for cycle_i in tqdm(range(grand_updates),file=sys.stdout):
                loss = self.agent.supervised_learn(0)

                if cycle_i % 1000 == 0:
                    print('Cycle:',cycle_i,'Loss:',loss)

            self.agent.soft_update_target()
    