'''
Created on 19.12.2018

@author: christian
'''

import os, time, signal
import typing
import gymnasium as gym
import gym.spaces as spa
from gym import error
from gym import utils
import json

try:
    from freecivbot.civclient import CivClient
    from freecivbot.connectivity.clinet import CivConnection
    from freecivbot.bot.base_bot import BaseBot
    
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Freeciv dependencies with 'pip install gym[freeciv].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class GymBot(BaseBot):
    def __init__(self, gym_env):
        BaseBot.__init__(self)
        self._env = gym_env
        self._last_action = None

    def calculate_next_move(self):
        if self._turn_active:
            obs, self._env.reward, self._env.done, _ = self._env.step(self._last_action)
            if self._env.done:
                pass
            action = self._env.gym_agent.act(obs, self._env.reward, self._env.done)
            if action == None:
                time.sleep(2)
                self.end_turn()
                return
            else:
                self.take_action(action)

    def reset(self):
        self._env.gym_agent.reset()
        
    def take_action(self, action):
        action_list = action[0]
        action_list.trigger_validated_action(action[1])

        self._last_action = action

    def getState(self, update=False):
        if update:
            self._acquire_state()
        return self._turn_state, self._turn_opts

    def get_reward(self):
        return self._turn_state["player"]["my_score"]

class FreecivEnv(gym.Env, utils.EzPickle):
    """ Basic Freeciv Web gym environment """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_turns=10, username="civbot", visualize=False):
        self.viewer = None
        self.status = None
        self.gym_agent = None
        self.max_turns = max_turns
        self.game_ports = [6000, 6004]
        self.current_port_id = 0
        self.visualize = visualize
        self.username = username

        aspace, ainfo = generate_action_space_info()
        self.action_space = aspace
        self.action_info = ainfo

    def __del__(self):
        pass
        """
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)
        """
    def step(self, action: gym.ActType) -> typing.tuple[gym.ObsType, gym.SupportsFloat, bool, bool, typing.dict[str, typing.Any]]:
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (terminated or truncated), it is necessary to call reset() to reset this environment's state for the next episode.

        Changed in version 0.26: The Step API was changed removing done in favor of terminated and truncated to make it clearer to users when the environment had terminated or truncated which is critical for reinforcement learning bootstrapping algorithms.

        Parameters:

            action (ActType) - an action provided by the agent to update the environment state.
        Returns:

                observation (ObsType) - An element of the environment-s observation_space as the next observation due to the agent actions. An example is a numpy array containing the positions and velocities of the pole in CartPole.

                reward (SupportsFloat) - The reward as a result of taking the action.

                terminated (bool) - Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. An example is reaching the goal state or moving into the lava from the Sutton and Barton, Gridworld. If true, the user needs to call reset().

                truncated (bool) - Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().

                info (dict) - Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. In OpenAI Gym <v26, it contains “TimeLimit.truncated” to distinguish truncation and termination, however this is deprecated in favour of returning terminated and truncated variables.

                done (bool) - (Deprecated) A boolean value for if the episode has ended, in which case further step() calls will return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes. A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully, a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        observation = self.my_bot.getState(update=True)
        reward = self.my_bot.get_reward()
        won = self.my_bot.has_won()
        turn_limit = self.my_bot.game_turn > self.max_turns
        episode_over = won | turn_limit
        if episode_over:
            self.my_bot.close_game()
        return observation, reward, won, turn_limit, {}, episode_over

    def reset(self, *, seed: int , options: typing.dict[str, typing.Any] | None = None) -> typing.tuple[gym.ObsType, typing.dict[str, typing.Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the state space and learns a generalised policy about the environment. This randomness can be controlled with the seed parameter otherwise if the environment already has a random number generator and reset() is called with seed=None, the RNG is not reset.

        Therefore, reset() should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of reset() should be super().reset(seed=seed) which implements the seeding correctly.

        Changed in version v0.25: The return_info parameter was removed and now info is expected to be returned.

        Parameters:

                seed (optional int) - The seed that is used to initialize the environment's PRNG (np_random). If the environment does not already have a PRNG and seed=None (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and seed=None is passed, the PRNG will not be reset. If you pass an integer, the PRNG will be reset even if it already exists. Usually, you want to pass an integer right after the environment has been initialized and then never again. Please refer to the minimal example above to see this paradigm in action.

                options (optional dict) - Additional information to specify how the environment is reset (optional, depending on the specific environment)

        Returns:

                observation (ObsType) - Observation of the initial state. This will be an element of observation_space (typically a numpy array) and is analogous to the observation returned by step().

                info (dictionary) - This dictionary contains auxiliary information complementing observation. It should be analogous to the info returned by step().
        """
        
        self.done = False
        self.my_bot = GymBot(self)
        self._reset_client(visualize=True)
        
        max_turns = self.max_turns
        client_port = self.game_ports[self.current_port_id]
        self.current_port_id = (self.current_port_id + 1) % len(self.game_ports)
        self.my_civ_client = CivClient(self.my_bot, self.username, client_port=client_port, visual_monitor=self.visualize)
        self.civ_conn = CivConnection(self.my_civ_client, 'http://localhost')
        
        observation = self.my_bot.getState(update=True)
        
        return observation, {}
    
    def render(self) -> gym.RenderFrame | typing.list[gym.RenderFrame] | None:
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

        return super().render()

      
    def take_snapshot(self, ob, base_dir):
        f = open(base_dir + "example_observation_turn{self.my_bot.game_turn}_state.json", "w")
        json.dump(ob[0], f, skipkeys=True, default=lambda x: x.tolist(), sort_keys=True)
        f.close()
        f = open(base_dir + "example_observation_turn{self.my_bot.game_turn}_actions.json", "w")
        json.dump(ob[1], f, skipkeys=True, default=lambda x: x.json_struct(), sort_keys=True)
        f.close()