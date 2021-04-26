import numpy as np
import sys
import zipfile, json

from numpy.core import numeric

from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
# from DeepMimic import update_world


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
        self.coreEnv = DeepMimicEnv(args, False)
        self.world = RLWorld(self.coreEnv, self.arg_parser) # TODO: remove.
        # self.world = self.create_model(self.coreEnv)
        fps = 60
        self.update_timestep = 1.0 / fps
    
    def run(self):
        done = False
        while not done:
            self.update(self.update_timestep)
            # update_world(self.world, self.update_timestep)

    def shutdown(self):
        Logger.print('Shutting down...')
        self.world.shutdown()
    
    def main(self):
        self.run()
        self.shutdown()
    
    def create_model(self, env):
        model_dir = self.model_path + 'deepmimic'
        policy_dir = model_dir + '/parameter_walk.zip'
        sub_dir = '/continue1'
        print("\033[92m"+model_dir + sub_dir+"\033[0m")

        obs_size = self.coreEnv.get_state_size(self.id)
        obs_index = np.linspace(0,obs_size-1,obs_size)
        net_arch = {'pi': [1024,512], 'vf': [1024,512]}
        policy_kwargs = {'net_arch': [net_arch], 'obs_index':obs_index}

        with zipfile.ZipFile(policy_dir, 'r') as file_:
            namelist = file_.namelist()
            if 'parameters' in namelist:
                parameter_list_json = file_.read("parameter_list").decode()
                parameter_list = json.loads(parameter_list_json)
                serialized_params = file_.read("parameters")
                params = bytes_to_params(serialized_params, parameter_list)
        
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir+sub_dir, 'policy_kwargs': policy_kwargs, \
                        'verbose': 1, 'learning_rate':_lr_scheduler, 'learning_starts':100, 'ent_coef':1e-7}
        self.trainer = SAC_MULTI(MlpPolicy_hpcsac, env, **model_dict)
        self.trainer.load_parameters(params, exact_match=False)
        self.trainer.learn(total_time_step, save_interval=10000, save_path=model_dir+sub_dir)
        print("Train Finished")
        self.trainer.save(model_dir+sub_dir)
         
    def update(self, time_elapsed):
        num_substeps = self.coreEnv.get_num_update_substeps()
        timestep = time_elapsed / num_substeps
        num_substeps = 1 if (time_elapsed == 0) else num_substeps

        for i in range(num_substeps):
            self.world.update(timestep)

            valid_episode = self.coreEnv.check_valid_episode()
            if valid_episode:
                end_episode = self.coreEnv.is_episode_end()
                if end_episode:
                    self.world.end_episode()
                    self.world.reset()
                    break
            else:
                self.world.reset()


if __name__ == '__main__':
    op = Optimizer()
    op.main()