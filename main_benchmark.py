import os, sys, time
import gym
import zipfile, json, io
import numpy as np
from pathlib import Path
from collections import OrderedDict

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from util.arg_parser import ArgParser
from env.deepmimic_env import DeepMimicEnv
from env.deepmimic_gymenv import DeepMimicGymEnv
from learning.rl_world import RLWorld

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'stable_baselines')))
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_hpcsac


def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, 'Failed to load args from: '+arg_file

    return arg_parser

class BenchMark:
    def __init__(self, args):
        self.args = args
        # Dimensions of the window we are drawing into.
        self.win_width = 800
        self.win_height = int(self.win_width * 9.0 / 16.0)
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

        # self.sess_SRL = tf_util.single_threaded_session()
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/data/policies/"
        self.model_dir = self.model_path + 'benchmark/'
        self.policy_dir = self.model_dir + 'mimic_run.zip'
        print("\033[92m"+self.model_dir+"\033[0m")

        self._init_env()


    def _init_env(self):
        self._init_draw()
        self._reload()
        self._setup_draw()
    
    def _init_draw(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.win_width, self.win_height)
        glutCreateWindow(b'HPC benchmark: DeepMimic')
    
    def _reload(self):
        self.arg_parser = build_arg_parser(self.args)
        self.env = DeepMimicGymEnv(self.args, enable_draw=True)
        self.env.coreEnv.set_playback_speed(self.playback_speed)
        # self.world = self.create_model()
        # self.world = self.create_model_HPC()
        self.world = self.load_test_model()

    def _setup_draw(self):
        glutDisplayFunc(self._draw)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutMouseFunc(self._mouse_click)
        glutMotionFunc(self._mouse_move)
        glutTimerFunc(self.display_anim_time, self._animate, 0)
        self._reshape(self.win_width, self.win_height)
        self.env.coreEnv.reshape(self.win_width, self.win_height)

    def _draw(self):
        self._update_intermediate_buffer()
        self.env.coreEnv.draw()
        glutSwapBuffers()
        self.reshaping = False
    
    def _update_intermediate_buffer(self):
        if not self.reshaping:
            if self.win_width is not self.env.coreEnv.get_win_width() or self.win_height is not self.env.coreEnv.get_win_height():
                self.env.coreEnv.reshape(self.win_width, self.win_height)
    
    def _reshape(self, w, h):
        self.reshaping = True
        self.win_width = w
        self.win_height = h
    
    def _keyboard(self, key, x, y):
        key_val = int.from_bytes(key, byteorder='big')
        self.env.coreEnv.keyboard(key_val, x, y)
        if (key == b'\x1b'):
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
            self._reload()
        elif (key == b't'):
            self._toggle_training()

        glutPostRedisplay()
    
    def _shutdown(self):
        sys.exit(0)

    def _toggle_animate(self):
        self.animating = not self.animating
        if self.animating:
            glutTimerFunc(self.display_anim_time, self._animate, 0)
    
    def _animate(self, callback_val):
        counter_decay = 0

        if (self.animating):
            num_steps = self._get_num_timesteps()
            curr_time = self._get_curr_time()
            time_elapsed = curr_time - self.prev_time
            self.prev_time = curr_time

            timestep = -self.update_timestep if (self.playback_speed < 0) else self.update_timestep
            for i in range(num_steps):
                # self._update_world(timestep)
                self._update(timestep)
            
            # FPS counting
            update_count = num_steps / (0.001 * time_elapsed)
            if (np.isfinite(update_count)):
                self.updates_per_sec = counter_decay * self.updates_per_sec + (1 - counter_decay) * update_count
                self.env.coreEnv.set_updates_per_sec(self.updates_per_sec)
                
            timer_step = self._calc_display_anim_time(num_steps)
            update_dur = self._get_curr_time() - curr_time
            timer_step -= update_dur
            timer_step = np.maximum(timer_step, 0)
            
            glutTimerFunc(int(timer_step), self._animate, 0)
            glutPostRedisplay()

        if self.env.coreEnv.is_done():
            print("10")
            self._shutdown()

    def _get_num_timesteps(self):
        num_steps = int(self.playback_speed)
        num_steps = 1 if num_steps is 0 else num_steps
        num_steps = np.abs(num_steps)
        return num_steps
    
    def _get_curr_time(self):
        return glutGet(GLUT_ELAPSED_TIME)
    
    def _update(self, time_elapsed):
        num_substeps = self.env.coreEnv.get_num_update_substeps()
        timestep = time_elapsed / num_substeps
        num_substeps = 1 if (time_elapsed == 0) else num_substeps
        done = False
        for i in range(num_substeps):
            if self.env.need_new_action():
                state = self.env.get_state()
                goal = self.env.get_goal()
                # action = self.world.predict(state)[0]
                action, _, weight = self.world.predict_subgoal(state)
                print(weight)
                self.env.set_action(action)
            rew = self.env.get_reward()
            self.env.update(timestep)
            
            valid_episode = self.env.coreEnv.check_valid_episode()
            if valid_episode:
                end_episode = self.env.coreEnv.is_episode_end()
                if (end_episode):
                    done = True
            else:
                done = True
            if done:
                self.env.reset()
                break
    
    def _calc_display_anim_time(self, num_timestes):
        anim_time = int(self.display_anim_time * num_timestes / self.playback_speed)
        anim_time = np.abs(anim_time)
        return anim_time

    def _step_anim(self, timestep):
        # self._update_world(timestep)
        self._update(timestep)
        self.animating = False
        glutPostRedisplay()

    def _change_playback_speed(self, delta):
        prev_playback = self.playback_speed
        self.playback_speed += delta
        self.env.coreEnv.set_playback_speed(self.playback_speed)

        if (np.abs(prev_playback) < 0.0001 and np.abs(self.playback_speed) > 0.0001):
            glutTimerFunc(self.display_anim_time, self._animate, 0)

    def _mouse_click(self, button, state, x, y):
        self.env.coreEnv.mouse_click(button, state, x, y)
        glutPostRedisplay()
    
    def _mouse_move(self, x, y):
        self.env.coreEnv.mouse_move(x,y)
        glutPostRedisplay()

    def _init_time(self):
        self.prev_time = self._get_curr_time()
        self.updates_per_sec = 0

    def create_model(self):
        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        # goal_size = self.env.coreEnv.get_goal_size(0)
        # print("goal size: ", goal_size)
        net_arch = {'pi': [1024,512], 'vf': [1024,512]}
        policy_kwargs = {'net_arch': [net_arch], 'obs_index':obs_index}

        param_dict = OrderedDict()
        with zipfile.ZipFile(self.policy_dir, 'r') as file_:
            namelist = file_.namelist()
            if 'parameters' in namelist:
                parameter_list_json = file_.read("parameter_list").decode()
                parameter_list = json.loads(parameter_list_json)
                serialized_params = file_.read("parameters")
                byte_file = io.BytesIO(serialized_params)
                params = np.load(byte_file)
                for param_name in parameter_list:
                    print(param_name)
                    param_dict[param_name] = params[param_name]
        
        model_dict = {'gamma': 0.99, 'tensorboard_log': self.model_dir, 'policy_kwargs': policy_kwargs, \
                        'verbose': 1, 'learning_starts':100, 'ent_coef':1e-7}
        trainer = SAC_MULTI(MlpPolicy_hpcsac, self.env, benchmark=False, **model_dict)
        trainer.load_parameters(param_dict, exact_match=False)
        return trainer
    
    def create_model_HPC(self):
        seed = 1
        policy = MlpPolicy_hpcsac
        trainer = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name='jogging')

        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        act_size = self.env.coreEnv.get_action_size(0)
        act_index = list(np.linspace(0,act_size-1,act_size,dtype=np.int32))

        prim_name = 'walking'
        policy_zip_path = self.model_dir+'walk_primitive.zip'
        trainer.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=obs_index, 
                                        act_range=None, act_index=act_index, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)
        prim_name = 'running'
        policy_zip_path = self.model_dir+'run_primitive.zip'
        trainer.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=obs_index, 
                                        act_range=None, act_index=act_index, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)
        number_of_primitives = 2
        trainer.construct_primitive_info(name='weight', freeze=False, level=1,
                                        obs_range=0, obs_index=obs_index,
                                        act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                        obs_relativity={},
                                        layer_structure={'policy':[1024, 512],'value':[1024, 512]},
                                        subgoal={})
        model_dict = {'gamma': 0.99, 'tensorboard_log': self.model_dir, 'verbose': 1, 'seed': seed, \
            'learning_starts':50000, 'ent_coef': 1e-7, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        trainer.pretrainer_load(model=trainer, policy=policy, env=self.env, **model_dict)
        return trainer
    
    def load_test_model(self):
        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        act_size = self.env.coreEnv.get_action_size(0)
        act_index = list(np.linspace(0,act_size-1,act_size,dtype=np.int32))
        
        model = SAC_MULTI(policy=MlpPolicy_hpcsac, env=None, _init_setup_model=False, composite_primitive_name='jogging')
        policy_zip_path = self.model_dir+'HPC_jogging'+str(1)+'/policy_30000.zip'
        model.construct_primitive_info(name=None, freeze=True, level=1,
                                        obs_range=None, obs_index=obs_index,
                                        act_range=None, act_index=act_index, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=True)
        SAC_MULTI.pretrainer_load(model, MlpPolicy_hpcsac, self.env)
        return model
    
    def save_model(self):
        self.world.save(self.model_dir+"/run_primitive")
    
    def main_loop(self):
        self._init_time()
        glutMainLoop()
    

if __name__ == '__main__':
    args = sys.argv[1:]
    bm = BenchMark(args)
    # bm.save_model()
    bm.main_loop()
