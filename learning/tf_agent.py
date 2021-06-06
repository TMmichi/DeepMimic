import numpy as np
import tensorflow as tf
from abc import abstractmethod
import zipfile, json, io
from collections import OrderedDict

from learning.rl_agent import RLAgent
from util.logger import Logger
from learning.tf_normalizer import TFNormalizer

class TFAgent(RLAgent):
    RESOURCE_SCOPE = 'resource'
    SOLVER_SCOPE = 'solvers'

    def __init__(self, world, id, json_data):
        self.tf_scope = 'agent'
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        super().__init__(world, id, json_data)
        self._build_graph(json_data)
        self._init_normalizers()
        return

    def __del__(self):
        self.sess.close()
        return

    def save_model(self, out_path):
        with self.sess.as_default(), self.graph.as_default():
            try:
                save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
                Logger.print('Model saved to: ' + save_path)
            except:
                Logger.print("Failed to save model to: " + save_path)
        return

    def load_model(self, in_path):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            self._load_normalizers()
            Logger.print('Model loaded from: ' + in_path)
            '''
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            paramval = self.sess.run(vars)
            namelist = ['agent/main/actor/0/dense/kernel:0', 'agent/main/actor/0/dense/bias:0',\
                        'agent/main/actor/1/dense/kernel:0', 'agent/main/actor/1/dense/bias:0',\
                        'agent/main/actor/dist_gauss_diag/mean/kernel:0', 'agent/main/actor/dist_gauss_diag/mean/bias:0',\
                        'agent/main/actor/dist_gauss_diag/logstd/bias:0',
                        'agent/resource/s_norm/mean:0',
                        'agent/resource/s_norm/std:0',
                        'agent/resource/a_norm/mean:0',
                        'agent/resource/a_norm/std:0',]
            paramdict = OrderedDict()
            for param, value in zip(vars, paramval):
                if param.name == namelist[0]:
                    paramdict["model/pi/fc0/kernel:0"] = value
                elif param.name == namelist[1]:
                    paramdict["model/pi/fc0/bias:0"] = value
                elif param.name == namelist[2]:
                    paramdict["model/pi/fc1/kernel:0"] = value
                elif param.name == namelist[3]:
                    paramdict["model/pi/fc1/bias:0"] = value
                elif param.name == namelist[4]:
                    paramdict["model/pi/dense/kernel:0"] = value
                elif param.name == namelist[5]:
                    paramdict["model/pi/dense/bias:0"] = value
                elif param.name == namelist[6]:
                    paramdict["model/pi/dense_1/bias:0"] = value
                elif param.name == namelist[7]:
                    paramdict["model/pi/s_norm/mean:0"] = value
                elif param.name == namelist[8]:
                    paramdict["model/pi/s_norm/std:0"] = value
                elif param.name == namelist[9]:
                    paramdict["model/pi/a_norm/mean:0"] = value
                elif param.name == namelist[10]:
                    paramdict["model/pi/a_norm/std:0"] = value
                else:
                    paramdict[param.name] = value
            # paramdict = OrderedDict((param.name, value) for param, value in zip(vars, paramval))
            print(paramdict)
            byte_file = io.BytesIO()
            np.savez(byte_file, **paramdict)
            serialized_params = byte_file.getvalue()
            serialized_param_list = json.dumps(
                list(paramdict.keys()),
                indent=4
            )
            print(serialized_param_list)
            with zipfile.ZipFile("data/policies/benchmark/amp_backflip.zip", "w") as file_:
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)
            quit()'''
        return

    def _get_output_path(self):
        assert(self.output_dir != '')
        file_path = self.output_dir + '/agent' + str(self.id) + '_model.ckpt'
        return file_path

    def _get_int_output_path(self):
        assert(self.int_output_dir != '')
        file_path = self.int_output_dir + ('/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
        return file_path

    def _build_graph(self, json_data):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_nets(json_data)
                
                with tf.variable_scope(self.SOLVER_SCOPE):
                    self._build_losses(json_data)
                    self._build_solvers(json_data)

                self._initialize_vars()
                self._build_saver()
        return

    def _init_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            # update normalizers to sync the tensorflow tensors
            self._s_norm.update()
            self._g_norm.update()
            self._a_norm.update()
        return

    @abstractmethod
    def _build_nets(self, json_data):
        pass

    @abstractmethod
    def _build_losses(self, json_data):
        pass

    @abstractmethod
    def _build_solvers(self, json_data):
        pass

    def _tf_vars(self, scope=''):
        with self.sess.as_default(), self.graph.as_default():
            res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + scope)
            assert len(res) > 0
        return res

    def _build_normalizers(self):
        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
            with tf.variable_scope(self.RESOURCE_SCOPE):
                self._s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(), self.world.env.build_state_norm_groups(self.id))
                self._s_norm.set_mean_std(-self.world.env.build_state_offset(self.id), 
                                         1 / self.world.env.build_state_scale(self.id))
                
                self._g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
                self._g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id), 
                                         1 / self.world.env.build_goal_scale(self.id))

                self._a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
                self._a_norm.set_mean_std(-self.world.env.build_action_offset(self.id), 
                                         1 / self.world.env.build_action_scale(self.id))
        return

    def _load_normalizers(self):
        self._s_norm.load()
        self._g_norm.load()
        self._a_norm.load()
        return

    def _update_normalizers(self):
        super()._update_normalizers()
        return

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())
        return

    def _build_saver(self):
        vars = self._get_saver_vars()
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
            #vars = [v for v in vars if '/target/' not in v.name]
            assert len(vars) > 0
        return vars
    
    def _weight_decay_loss(self, scope):
        vars = self._tf_vars(scope)
        vars_no_bias = [v for v in vars if 'bias' not in v.name]
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
        return loss

    def _train(self):
        with self.sess.as_default(), self.graph.as_default():
            super()._train()
        return