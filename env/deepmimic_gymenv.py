import gym
import numpy as np

from .deepmimic_env import DeepMimicEnv


class DeepMimicGymEnv:
    def __init__(self, args, enable_draw=True):
        self.enable_draw = enable_draw
        self.coreEnv = DeepMimicEnv(args, enable_draw)

        self.id = 0
        self.num_envs = 1

        obs_size = self.coreEnv.get_state_size(self.id)
        obs_min = np.array([-1] * obs_size)
        obs_max = np.array([1] * obs_size)
        self.observation_space = gym.spaces.Box(obs_min, obs_max, dtype=np.float32)

        act_size = self.coreEnv.get_action_size(self.id)
        act_min = np.array([-1] * act_size)
        act_max = np.array([1] * act_size)
        self.action_space = gym.spaces.Box(act_min, act_max, dtype=np.float32)

        self.num_substeps = self.coreEnv.get_num_update_substeps()
        fps = 60
        update_timestep = 1.0 / fps
        self.timestep = update_timestep / self.num_substeps

        self.target_vel = 2.8


    def reset(self):
        self.coreEnv.reset()
        return self.coreEnv.record_state(self.id)
    
    def get_state(self):
        return self.coreEnv.record_state(self.id)
    
    def get_goal(self):
        return self.coreEnv.record_goal(self.id)
    
    def get_reward(self):
        return self.coreEnv.calc_reward(self.id)
    
    # def get_custom_reward(self):
    #     rew = self.coreEnv.calc_reward(self.id)
    #     return -abs(self.target_vel - vel_x)
    
    def need_new_action(self):
        return self.coreEnv.need_new_action(self.id)

    def set_action(self, action):
        self.coreEnv.set_action(self.id, action)
    
    def step(self, action, weight=None, subgoal=None, id=None):
        self.coreEnv.set_action(self.id, action)
        done = False
        need_new_action = False
        while (not need_new_action) and (not done):
            self.update(self.timestep)
            need_new_action = self.need_new_action()
            if self.coreEnv.check_valid_episode():
                if self.coreEnv.is_episode_end():
                    done = True
            else:
                done = True
        state = self.get_state()
        rew = self.get_reward()
        if done is not True and rew < -50:
            rew += 1000
        return state, rew, done, {0: 0}

    
    def update(self, timestep):
        self.coreEnv.update(timestep)


    def seed(self, seed):
        pass


