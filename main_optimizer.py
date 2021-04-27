import os, sys
import zipfile, json, io

import numpy as np
from pathlib import Path
from collections import OrderedDict

from env.deepmimic_env import DeepMimicEnv
from env.deepmimic_gymenv import DeepMimicGymEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'stable_baselines')))
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_hpcsac

def _lr_scheduler(frac):
    return 7e-5 * frac

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, 'Failed to load args from: '+arg_file

    return arg_parser

class Optimizer:
    def __init__(self) -> None:
        args = sys.argv[1:]
        print('args in main_test: ', args)
        self.arg_parser = build_arg_parser(args)
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/data/policies/"
        self.model_dir = self.model_path + 'benchmark'
        # self.policy_dir = self.model_dir + '/parameter_walk.zip'
        self.policy_dir = self.model_dir + '/param_walk.zip'
        print("\033[92m"+self.model_dir+"\033[0m")

        # self.sess_SRL = tf_util.single_threaded_session()
        self.env = DeepMimicGymEnv(args)
        # self.world = RLWorld(self.coreEnv, self.arg_parser) # TODO: remove.
        self.world = self.create_model()
        fps = 60
        self.update_timestep = 1.0 / fps
        
    
    def run(self):
        self.trainer.learn(10000000, save_interval=10000, save_path=self.model_dir)
        print("Train Finished")
        self.trainer.save(self.model_dir)
        # done = False
        # while not done:
        #     self.update(self.update_timestep)

    def shutdown(self):
        Logger.print('Shutting down...')
        self.world.shutdown()
    
    def main(self):
        self.run()
        self.shutdown()
    
    def create_model(self):
        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        print(obs_index)
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
                    param_dict[param_name] = params[param_name]
        for name in param_dict.keys():
            print(name)
            if name == 'agent/resource/s_norm/mean:0':
                print(param_dict[name].shape)
        
        model_dict = {'gamma': 0.99, 'tensorboard_log': self.model_dir, 'policy_kwargs': policy_kwargs, \
                        'verbose': 1, 'learning_rate':_lr_scheduler, 'learning_starts':100, 'ent_coef':1e-7}
        self.trainer = SAC_MULTI(MlpPolicy_hpcsac, self.env, benchmark=False, **model_dict)
        self.trainer.load_parameters(param_dict, exact_match=False)

         
    def update(self, time_elapsed):
        num_substeps = self.env.coreEnv.get_num_update_substeps()
        timestep = time_elapsed / num_substeps
        num_substeps = 1 if (time_elapsed == 0) else num_substeps

        for i in range(num_substeps):
            self.world.update(timestep)

            valid_episode = self.env.coreEnv.check_valid_episode()
            if valid_episode:
                end_episode = self.env.coreEnv.is_episode_end()
                if end_episode:
                    self.world.end_episode()
                    self.world.reset()
                    break
            else:
                self.world.reset()


if __name__ == '__main__':
    op = Optimizer()
    op.main()