import os, sys
import zipfile, json, io
from mpi4py import MPI
comm = MPI.COMM_WORLD

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
    return 5e-5 * frac

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

        self.trial = 2
        self.id = int(id(""))
        print("agent id: "+str(self.id))
        # self.sess_SRL = tf_util.single_threaded_session()
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/data/policies/"
        self.model_dir = self.model_path + 'benchmark/'
        self.save_dir = self.model_dir + 'HPC_jog_amp' + str(self.trial) + "/id_"+str(self.id)
        os.makedirs(self.save_dir, exist_ok=True)
        print("\033[92m"+self.save_dir+"\033[0m")

        self.env = DeepMimicGymEnv(args, enable_draw=False)
        self.world = self.create_model_HPC()
        #self.world = self.create_model_HPC_dog()
        fps = 60
        self.update_timestep = 1.0 / fps
        
    
    def run(self):
        self.world.learn(10000000, save_interval=20000, save_path=self.save_dir)
        print("Train Finished")
        self.world.save(self.save_dir)

    def shutdown(self):
        Logger.print('Shutting down...')
        self.world.shutdown()
    
    def main(self):
        self.run()
        self.shutdown()
    
    def create_model(self):
        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        net_arch = {'pi': [1024,512], 'vf': [1024,512]}
        policy_kwargs = {'net_arch': [net_arch], 'obs_index':obs_index}

        param_dict = OrderedDict()
        policy_dir = self.model_dir + 'param_walk.zip'
        with zipfile.ZipFile(policy_dir, 'r') as file_:
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
                        'verbose': 1, 'learning_rate':_lr_scheduler, 'learning_starts':100, 'ent_coef':1e-7}
        trainer = SAC_MULTI(MlpPolicy_hpcsac, self.env, benchmark=False, **model_dict)
        trainer.load_parameters(param_dict, exact_match=False)
        return trainer
    
    def create_model_HPC(self):
        seed = self.trial
        policy = MlpPolicy_hpcsac
        trainer = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name='jogging')

        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        act_size = self.env.coreEnv.get_action_size(0)
        act_index = list(np.linspace(0,act_size-1,act_size,dtype=np.int32))

        prim_name = 'walk'
        policy_zip_path = self.model_dir+prim_name+'_amp_primitive.zip'
        trainer.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=obs_index, 
                                        act_range=None, act_index=act_index, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)
        prim_name = 'run'
        policy_zip_path = self.model_dir+prim_name+'_amp_primitive.zip'
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
                                        layer_structure={'policy':[256, 256, 256],'value':[256, 256, 256]},
                                        subgoal={})
        model_dict = {'gamma': 0.99, 'tensorboard_log': self.save_dir, 'verbose': 1, 'seed': seed, \
            'learning_rate':_lr_scheduler, 'learning_starts':10000, 'ent_coef': 1e-7, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        trainer.pretrainer_load(model=trainer, policy=policy, env=self.env, **model_dict)
        return trainer
    
    def create_model_HPC_dog(self):
        seed = 1
        policy = MlpPolicy_hpcsac
        trainer = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name='jogging')

        obs_size = self.env.coreEnv.get_state_size(0)
        obs_index = list(np.linspace(0,obs_size-1,obs_size,dtype=np.int32))
        act_size = self.env.coreEnv.get_action_size(0)
        act_index = list(np.linspace(0,act_size-1,act_size,dtype=np.int32))

        prim_name = 'walking'
        policy_zip_path = self.model_dir+'dog_walk_primitive.zip'
        trainer.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=obs_index, 
                                        act_range=None, act_index=act_index, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)
        prim_name = 'running'
        policy_zip_path = self.model_dir+'dog_run_primitive.zip'
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
                                        layer_structure={'policy':[256, 256, 256],'value':[256, 256, 256]},
                                        subgoal={})
        model_dict = {'gamma': 0.99, 'tensorboard_log': self.save_dir, 'verbose': 1, 'seed': seed, \
            'learning_rate':_lr_scheduler, 'learning_starts':10000, 'ent_coef': 1e-7, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        trainer.pretrainer_load(model=trainer, policy=policy, env=self.env, **model_dict)
        return trainer


if __name__ == '__main__':
    op = Optimizer()
    op.main()