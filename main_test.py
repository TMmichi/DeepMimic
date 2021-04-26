import numpy as np
import sys
from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.logger import Logger
from DeepMimic import update_world, update_timestep, build_world
import util.mpi_util as MPIUtil

class Optimizer:
    def __init__(self) -> None:
        args = sys.argv[1:]
        self.world = build_world(args, enable_draw=False)
        self.update_timestep = update_timestep
    
    def run(self):
        done = False
        while not done:
            update_world(self.world, self.update_timestep)

    def shutdown(self):
        Logger.print('Shutting down...')
        self.world.shutdown()
    
    def main(self):
        self.run()
        self.shutdown()
         

if __name__ == '__main__':
    op = Optimizer()
    op.main()