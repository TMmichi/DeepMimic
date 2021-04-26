import os, sys
import gym
import numpy as np
from pathlib import Path

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import path_config
from configuration import model_configuration, info, total_time_step
from util.arg_parser import ArgParser
from deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld

import stable_baselines.common.tf_util as tf_util
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_hpcsac
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.save_util import bytes_to_params


default_lr = model_configuration['learning_rate']
def _lr_scheduler(frac):
    return default_lr * frac

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, 'Failed to load args from: '+arg_file

    return arg_parser

class RL_controller:
    def __init__(self, args):
        self.coreEnv = DeepMimicEnv(args, False)
        self.arg_parser = build_arg_parser(args)

        # Dimensions of the window we are drawing into.
        self.win_width = 800
        self.win_height = int(self.win_width * 9 / 16)
        self.reshaping = False
        # anim
        fps = 60
        self.update_timestep = 1.0 / fps
        self.display_anim_time = int(1000 * self.update_timestep)
        self.animating = True
        self.playback_speed = 1
        self.playback_delta = 0.05
        # FPS counter
        self.prev_time = 0
        self.updates_per_sec = 0

        self._init_env()

        self.sess_SRL = tf_util.single_threaded_session()
        package_path = str(Path(__file__).resolve().parent.parent)
        self.model_path = package_path+"/models_baseline/"
        os.makedirs(self.model_path, exist_ok=True)

        self.trial = 54

        # Not needed
        self.id = 0
        self.num_envs = 1

        obs_size = self.coreEnv.get_state_size(self.id)
        obs_min = [-1] * obs_size
        obs_max = [1] * obs_size
        self.observation_space = gym.spaces.Box(obs_min, obs_max, dtype=np.float32)

        act_size = self.coreEnv.get_action_size(self.id)
        act_min = [-1] * act_size
        act_max = [1] * act_size
        self.action_space = gym.spaces.Box(act_min, act_max, dtype=np.float32)


    def _init_env(self):
        self._init_draw()
        self.world = RLWorld(self.coreEnv, self.arg_parser) # TODO: remove.
        self._setup_draw()
    
    def _init_draw(self):
        glutInit()  
    
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.win_width, self.win_height)
        glutCreateWindow(b'HPC benchmark: DeepMimic')
    
    def _setup_draw(self):
        glutDisplayFunc(self._draw)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutMouseFunc(self._mouse_click)
        glutMotionFunc(self._mouse_move)
        glutTimerFunc(self.display_anim_time, self._animate, 0)

        self._reshape(self.win_width, self.win_height)
        self.coreEnv.reshape(self.win_width, self.win_height)

    def _draw(self):
        self._update_intermediate_buffer()
        self.coreEnv.draw()

        glutSwapBuffers()
        self.reshaping = False
    
    def _update_intermediate_buffer(self):
        if not self.reshaping:
            if self.win_width is not self.coreEnv.get_win_widt() or self.win_height is not self.coreEnv.get_win_height():
                self.coreEnv.reshape(self.win_height, self.win_height)
    
    def _reshape(self, w, h):
        self.reshaping = True
        self.win_width = w
        self.win_height = h
    
    def _keyboard(self, key, x, y):
        key_val = int.from_bytes(key, byteorder='big')
        self.coreEnv.keyboard(key_val, x, y)
        if (key == b'\x1b'): # escape
            self._shutdown()
        elif (key == b' '):
            self._toggle_animate()
        elif (key == b'>'):
            self._step_anim(self.update_timestep)
        elif (key == b'<'):
            self._step_anim(-self.update_timestep)
        elif (key == b','):
            self._change_playback_speed(-self.playback_delta)
        elif (key == b'.'):
            self._change_playback_speed(self.playback_delta)
        elif (key == b'/'):
            self._change_playback_speed(-self.playback_speed + 1)
        elif (key == b'r'):
            self.reset()
        elif (key == b't'):
            self._toggle_training()

        glutPostRedisplay()
    
    def _shutdown(self):
        sys.exit(0)

    def _toggle_animate(self):
        self.animating = not self.animating
        if self.animating:
            glutTimerFunc(self.display_anim_time, self._animate, 0)
    
    def _animate(self):
        counter_decay = 0

        if (self.animating):
            num_steps = self._get_num_timesteps()
            curr_time = self._get_curr_time()
            time_elapsed = curr_time - self.prev_time
            prev_time = curr_time

            timestep = -self.update_timestep if (self.playback_speed < 0) else self.update_timestep
            for i in range(num_steps):
                self._update_world(timestep)
            
            # FPS counting
            update_count = num_steps / (0.001 * time_elapsed)
            if (np.isfinite(update_count)):
                updates_per_sec = counter_decay * self.updates_per_sec + (1 - counter_decay) * update_count
                self.coreEnv.set_updates_per_sec(updates_per_sec)
                
            timer_step = self._calc_display_anim_time(num_steps)
            update_dur = self._get_curr_time() - curr_time
            timer_step -= update_dur
            timer_step = np.maximum(timer_step, 0)
            
            glutTimerFunc(int(timer_step), self._animate, 0)
            glutPostRedisplay()

        if self.coreEnv.is_done():
            self._shutdown()

    def _get_num_timesteps(self):
        num_steps = int(self.playback_speed)
        num_steps = 1 if num_steps is 0 else num_steps

        num_steps = np.abs(num_steps)
        return num_steps
    
    def _get_curr_time(self):
        return glutGet(GLUT_ELAPSED_TIME)
    
    def _update_world(self, time_elapsed):
        num_substeps = self.coreEnv.get_num_update_substeps()
        timestep = time_elapsed / num_substeps
        num_substeps = 1 if (time_elapsed == 0) else num_substeps

        for i in range(num_substeps):
            self.world.update(timestep) # TODO: remove

            valid_episode = self.coreEnv.check_valid_episode()
            if valid_episode:
                end_episode = self.coreEnvenv.is_episode_end()
                if (end_episode):
                    # TODO: remove
                    self.world.end_episode()
                    self.world.reset()
                    break
            else:
                # TODO: remove
                self.world.reset()
                break
    
    def _calc_display_anim_time(self, num_timestes):
        anim_time = int(self.display_anim_time * num_timestes / self.playback_speed)
        anim_time = np.abs(anim_time)
        return anim_time

    def _step_anim(self, timestep):
        self._update_world(timestep)
        self.animating = False
        glutPostRedisplay()

    def _change_playback_speed(self, delta):
        prev_playback = self.playback_speed
        self.playback_speed += delta
        self.coreEnv.set_playback_speed(self.playback_speed)

        if (np.abs(prev_playback) < 0.0001 and np.abs(self.playback_speed) > 0.0001):
            glutTimerFunc(self.display_anim_time, self._animate, 0)

    def _mouse_click(self, button, state, x, y):
        self.coreEnv.mouse_click(button, state, x, y)
        glutPostRedisplay()
    
    def _mouse_move(self, x, y):
        self.coreEnv.mouse_move(x,y)
        glutPostRedisplay()

    def _init_time(self):
        self.prev_time = self._get_curr_time()
        self.updates_per_sec = 0


    def reset(self):
        self.coreEnv.Reset()
        return self.coreEnv.record_state(self.id)

    def step(self, action, weight=None, subgoal=None, id=None):
        timestep = 0    # TODO: set timestep
        self.coreEnv.set_action(self.id, action)
        self.coreEnv.update(timestep)
        obs = self.coreEnv.record_state(self.id)
        rew = self.coreEnv.calc_reward(self.id)
        done = self.coreEnv.is_done()
        return obs, rew, done, {0: 0}

    def seed(self, seed):
        pass

