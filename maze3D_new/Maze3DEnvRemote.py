import inspect
import random
import time
import traceback

import numpy as np
import requests as requests

from game.game_utils import get_config


# ---------------------------------------------------------------------------
# Reward configuration and helpers
# ---------------------------------------------------------------------------
# `distance_from_goal` is provided directly by the Unity environment in the
# step response (res['distance_from_goal']); it is not computed here.
#
# Each reward function accepts its tuning parameters as keyword arguments whose
# names match the config keys under `game -> reward_params` (which mirror the
# `experiment.*` fields of the reference QMIX implementation). compute_reward()
# forwards only the parameters a given reward function actually accepts.

# Default reward magnitudes (used when a param is not set in the config).
GOAL_REWARD = 10
TIME_STEP_PENALTY = -1

# Per-episode state shared by the progress-based reward functions.
# Call reset_reward_state() at the start of every episode (e.g. in reset()).
_prev_distance = None
_stall_counter = 0


def reset_reward_state():
    """Clear the state used by the progress-based reward functions."""
    global _prev_distance, _stall_counter
    _prev_distance = None
    _stall_counter = 0


def get_ball_speed(observation):
    """Ball speed = magnitude of its velocity. observation[2:4] = (vx, vy)."""
    vx, vy = observation[2], observation[3]
    return (vx ** 2 + vy ** 2) ** 0.5


def reward_function_timeout_penalty(goal_reached, timedout,
                                    goal_reward=GOAL_REWARD,
                                    time_step_penalty=TIME_STEP_PENALTY):
    # "Simple" reward: goal_reward on success, a constant penalty otherwise.
    if goal_reached and not timedout:
        return goal_reward
    return time_step_penalty


def reward_function_goal_distance(goal_reached, timedout, distance_from_goal,
                                  goal_reward=GOAL_REWARD,
                                  time_step_penalty=TIME_STEP_PENALTY,
                                  reward_scale=-0.01):
    # Reward reaching the goal / penalize timeout, otherwise shape the reward
    # by the (scaled) distance from the goal.
    if goal_reached and not timedout:
        return goal_reward
    if timedout:
        return time_step_penalty
    return reward_scale * abs(distance_from_goal)


def reward_function_progress_distance(goal_reached, timedout, distance_from_goal,
                                      goal_reward=GOAL_REWARD,
                                      time_step_penalty=TIME_STEP_PENALTY,
                                      reward_scale=1.0):
    # Reward the progress made towards the goal since the previous step.
    global _prev_distance
    if goal_reached and not timedout:
        reset_reward_state()
        return goal_reward
    if timedout:
        reset_reward_state()
        return time_step_penalty

    if _prev_distance is None:
        _prev_distance = distance_from_goal
    reward = reward_scale * (_prev_distance - distance_from_goal)
    _prev_distance = distance_from_goal
    return reward


def reward_function_progress_with_stalling(goal_reached, timedout, distance_from_goal,
                                           goal_reward=GOAL_REWARD,
                                           time_step_penalty=TIME_STEP_PENALTY,
                                           reward_scale=1.0, min_distance_delta=0.01,
                                           stall_penalty=-1, stall_threshold=5):
    # Like progress_distance, but also penalizes "stalling": when too many
    # consecutive steps make little progress, apply stall_penalty.
    global _prev_distance, _stall_counter
    if goal_reached and not timedout:
        reset_reward_state()
        return goal_reward
    if timedout:
        reset_reward_state()
        return time_step_penalty

    if _prev_distance is None:
        _prev_distance = distance_from_goal
    delta = _prev_distance - distance_from_goal
    _prev_distance = distance_from_goal

    if delta < min_distance_delta:
        _stall_counter += 1
    else:
        _stall_counter = 0

    if _stall_counter >= stall_threshold:
        _stall_counter = 0
        return stall_penalty

    return reward_scale * delta


def reward_function_speed_stalling(goal_reached, timedout, distance_from_goal, ball_speed,
                                   goal_reward=GOAL_REWARD,
                                   time_step_penalty=TIME_STEP_PENALTY,
                                   reward_scale=1.0, min_distance_delta=0.01,
                                   stall_penalty=-1, stall_threshold=5,
                                   goal_zone_radius=1.0, speed_scale=0.01,
                                   speed_threshold=0.0):
    # Builds on progress_with_stalling and adds a penalty for moving too fast
    # once the ball is inside the goal zone (encourages slowing near the goal).
    reward = reward_function_progress_with_stalling(
        goal_reached, timedout, distance_from_goal,
        goal_reward, time_step_penalty,
        reward_scale, min_distance_delta, stall_penalty, stall_threshold)

    if distance_from_goal <= goal_zone_radius:
        reward -= speed_scale * max(0.0, ball_speed - speed_threshold)

    return reward


# ---------------------------------------------------------------------------
# Reward dispatch
# ---------------------------------------------------------------------------
# Maps the name read from the config file (game -> reward_function) to the
# matching reward function defined above. "simple" is an alias for the basic
# timeout-penalty reward (matches the reference "simple" reward engine).
REWARD_FUNCTIONS = {
    "simple": reward_function_timeout_penalty,
    "timeout_penalty": reward_function_timeout_penalty,
    "goal_distance": reward_function_goal_distance,
    "progress_distance": reward_function_progress_distance,
    "progress_with_stalling": reward_function_progress_with_stalling,
    "speed_stalling": reward_function_speed_stalling,
}

# Reward functions that don't take a distance argument.
_DISTANCE_FREE_REWARDS = {"simple", "timeout_penalty"}


def compute_reward(name, goal_reached, timedout, distance_from_goal,
                   ball_speed=0.0, params=None):
    """Run the reward function selected by `name` (from the config file).

    distance_from_goal comes from the Unity step response; ball_speed is only
    used by the speed_stalling reward. `params` is the dict read from
    `game -> reward_params` in the config; only the keys a given reward
    function accepts are forwarded to it.
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS)}"
        )

    fn = REWARD_FUNCTIONS[name]
    params = params or {}
    # Forward only the parameters this reward function actually declares.
    accepted = inspect.signature(fn).parameters
    kwargs = {k: v for k, v in params.items() if k in accepted}

    if name in _DISTANCE_FREE_REWARDS:
        return fn(goal_reached, timedout, **kwargs)

    if name == "speed_stalling":
        return fn(goal_reached, timedout, distance_from_goal, ball_speed, **kwargs)

    return fn(goal_reached, timedout, distance_from_goal, **kwargs)


class ActionSpace:
    def __init__(self):
        self.actions = list(range(3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    def __init__(self, config=None, config_file=None):
        print("Init Maze3D")
        self.config = get_config(config_file) if config_file is not None else config
        self.network_config = get_config("game/network_config.yaml")
        self.ip_host = self.network_config["ip_distributor"]
        self.outer_host = self.network_config["maze_server"]
        self.host = self.network_config["maze_rl"]

        # self.host = "http://localhost:8080"
        self.action_space = ActionSpace()
        self.fps = 60
        self.done = False
        # Which reward function to use, read from the config file
        # (game -> reward_function). Defaults to the basic timeout penalty.
        self.reward_function_name = self.config.get('game', {}).get(
            'reward_function', 'timeout_penalty')
        # Reward tuning parameters (game -> reward_params); forwarded to the
        # selected reward function. Empty dict => use the function defaults.
        self.reward_params = self.config.get('game', {}).get('reward_params') or {}
        print("Using reward function:", self.reward_function_name)
        print("Reward params:", self.reward_params)
        self.set_host()
        self.send_config()
        self.agent_ready()
        self.observation, _,_ = self.reset('test')
        self.observation_shape = (len(self.observation),)
        self.internet_delay = []

    def send_config(self):
        config = {}

        while True:
            # try:
                #print("Sending config",self.config)
                mode = self.config['Experiment']['mode']
                config['discrete_input'] = self.config['game']['discrete_input']
                config['max_duration'] = self.config['Experiment'][mode]['max_duration']
                config['action_duration'] = self.config['Experiment'][mode]['action_duration']
                config['human_speed'] = self.config['game']['human_speed']
                config['agent_speed'] = self.config['game']['agent_speed']
                config['discrete_angle_change'] = self.config['game']['discrete_angle_change']
                config['human_assist'] = self.config['game']['human_assist']
                config['human_only'] = self.config['game']['human_only']
                config['two_humans'] = self.config['game']['two_humans']
                config['start_up_screen_display_duration'] = self.config['GUI']['start_up_screen_display_duration']
                config['popup_window_time'] = self.config['GUI']['popup_window_time']
                print(config)
                requests.post(self.host + "/config", json=config).json()
                return
            # except Exception as e:
            #     print("/agent_ready not returned", e)
            #     time.sleep(1)

    def set_host(self):
        while True:
            print('Im trying this at least')
            try:
                requests.post(self.ip_host + "/set_server_host",json={'server_host':self.outer_host}).json()
                break
            except Exception as e:
                print("ip host offline", e)
                time.sleep(0.1)
        print('I succseded here')

    def agent_ready(self):
        while True:
            try:
                res = requests.get(self.host + "/agent_ready").json()
                if 'command' in res and res['command'] == "player_ready":
                    break
            except Exception as e:
                # print("/agent_ready not returned", e)
                time.sleep(0.1)

    def send(self, namespace, method="GET", data=None):
        while True:
            try:
                if method == "GET":
                    res = requests.get(self.host + namespace).json()
                else:
                    res = requests.post(self.host + namespace, json=data).json()

                if 'command' in res and res['command'] == "player_ready":
                    continue
                return res
            except Exception as e:
                # in here when wrong request is given
                # traceback.print_exc()
                self.agent_ready()
                time.sleep(0.1)

    def reset(self,type):
        # print("reset")
        # clear state used by the progress-based reward functions
        reset_reward_state()
        start_time = time.time()
        if type == 'train':
            res = self.send("/reset")
        elif type == 'test':
            res = self.send("/testreset")
        set_up_time = time.time() - start_time
        # print("reset time:", set_up_time)
        # return np.array(res['observation']), res['setting_up_duration']
        return np.asarray(res['observation']), set_up_time, res['pause']


    def training(self, cycle, total_cycles):
        self.send("/training", "POST", {'cycle': cycle, 'total_cycles': total_cycles})

    def finished(self):
        print("finished")
        self.send("/finished", "GET")

    def step(self, action_agent, timed_out, action_duration, mode,text):
        """
        Performs the action of the agent to the environment for action_duration time.
        Simultaneously, receives input from the user via the keyboard arrows.

        :param action_agent: the action of the agent. gives -1 for down, 0 for nothing and 1 for up
        :param timed_out: used
        :param action_duration: the duration of the agent's action on the game
        :param mode: training or test
        :return: a transition [observation, reward, done, timeout, train_fps, duration_pause, action_list]
        """
        # print("step", timed_out)
        # if timed_out:
        #     print("timeout", timed_out, int(time.time()))
        if mode == 'one_agent' or mode == 'human':
            payload = {
                'action_agent': action_agent,
                'second_agent_action': -2,
                'action_duration': action_duration,
                'timed_out': timed_out,
                'mode': mode,
                'display_text': text
            }
            start_time = time.time()
            res = self.send("/step", method="POST", data=payload)
        elif mode == 'two_agents':
            #print("action_agent", type(action_agent[0]), action_agent[1])
            payload = {
                'action_agent': int(action_agent[0]),
                'second_agent_action': int(action_agent[1]),
                'action_duration': action_duration,
                'timed_out': timed_out,
                'mode': mode,
                'display_text': text
            }
            start_time = time.time()
            res = self.send("/step_two_agents", method="POST", data=payload)

            #print('But it never comes back')

       
        delay = time.time() - start_time
        self.internet_delay.append(delay)
        self.observation = np.array(res['observation'])
        self.done = res['done']  # true if goal_reached OR timeout
        fps = res['fps']
        human_action = res['human_action']
        agent_action = res['agent_action']
        duration_pause = res['duration_pause']
        internet_pause = delay - duration_pause - action_duration
        # distance to the goal is computed by the Unity environment and sent
        # back in the step response; ball speed is derived from the observation
        distance_from_goal = res.get('distance_from_goal', 0.0)
        ball_speed = get_ball_speed(self.observation)
        reward = compute_reward(self.reward_function_name, self.done, timed_out,
                                distance_from_goal, ball_speed, self.reward_params)

        return self.observation, reward, self.done, fps, duration_pause, [agent_action, human_action], internet_pause


if __name__ == '__main__':
    """Dummy execution"""
    while True:
        try:
            maze = Maze3D()
            while True:
                maze.step(random.randint(-1, 1), None, None, 200)
        except:
            traceback.print_exc()