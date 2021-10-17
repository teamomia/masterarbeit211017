from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.cartesian_path import CartesianPath
import numpy as np
import math

POS_MIN, POS_MAX = [-0.2, 0.3, 0.3], [0.2, 0.7, 0.7]


class WorkEnv(object):
    def __init__(self, control_mode='joint_velocity'):
        self.reward_offset = 10.0
        self.reward_range = self.reward_offset
        self.penalty_offset = 1
        self.fall_down_offset = 0.1
        self.metadata = []  # gym env argument
        self.control_mode = control_mode

        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'iiwa14.ttt')
        # SCENE_FILE = join(dirname(abspath(__file__)), 'ur10.ttt')
        self.pr.launch(SCENE_FILE)
        self.pr.set_simulation_timestep(0.05)
        self.pr.start()
        # self.agent = UR10()
        self.agent = LBRIwaa14R820()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.observation_space = np.zeros(17)
        self.action_space = np.zeros(7)
        # self.action_space = np.zeros(6)
        self.agent_rr_tip = self.agent.get_tip()
        self.target = {}
        # define 100 target points for trajectory
        for i in range(100):
            self.target[i] = Dummy('Target'+str(i))
        self.agent_state = self.agent.get_configuration_tree()
        # self.initial_joint_positions = [0, 0, 0, 0, 0, 0]
        # self.initial_joint_positions = [0, 0, 0, 0, 0, 0, 0]
        self.initial_joint_positions = [-5.2984800338745, -4.6366157531738, 5.4974708557129, -4.6424608230591, -4.7502784729004, -3.926141500473, 5.7197856903076]
        self.agent.set_joint_positions(self.initial_joint_positions)

        self.pr.step()
        self.initial_tip_positions = self.agent_rr_tip.get_position()
        self.initial_joint_position = self.agent.get_joint_positions()

    def _get_state(self):
        # 7+7+3+3
        # 6+6+3+3
        # return np.concatenate([self.agent.get_joint_positions(), self.agent.get_joint_velocities(), self.agent_rr_tip.get_position(), self.agent_rr_tip.get_orientation()])
        return np.concatenate([self.agent.get_joint_positions(), self.agent.get_joint_velocities(), self.agent_rr_tip.get_position()])

    def reinit(self):
        self.shutdown()
        self.__init__()

    def reset(self, random_target=False):
        self.pr.set_configuration_tree(self.agent_state)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.step()

        return self._get_state()  # return the current state of the environment

    def step(self, action, index):
        # initialization
        done = False  # episode finishes
        reward = 0
        isDone = False  # reach to the target point
        
        if action is None or action.shape[0] != 7:  
        # if action is None or action.shape[0] != 6:  

            print('No actions or wrong action dimensions!')
            action = list(np.random.uniform(-1, 1, 7))
            # action = list(np.random.uniform(-1, 1, 6))
        self.agent.set_joint_target_velocities(action)
        self.pr.step()

        ax, ay, az = self.agent_rr_tip.get_position() #tcp position  
        dx, dy, dz = self.target[index].get_position() #target position
        aox, aoy, aoz = self.agent_rr_tip.get_orientation() #tcp orientation
        tox, toy, toz = self.target[index].get_orientation() #target orientation

        sqrt_distance = np.sqrt((100*(ax-dx))**2+(50*(ay-dy))**2+(50*(az-dz))**2)
        distance_error_x = np.abs(ax-dx)
        sqrt_orientation = np.sqrt((50*(aox-tox)**2) + (50*(aoy-toy)**2) +(50*(aoz-toz)**2))
        
        reward -= (sqrt_distance+sqrt_orientation)

        if sqrt_distance > 1000 or sqrt_orientation>1000:
            isDone = True
            done = True
            reward -= 1000
        return self._get_state(), reward, done, {'finished': isDone}

    def get_x_error(self, action, index):
            # initialization
            done = False  # episode finishes
            reward = 0
            isDone = False  # reach to the target point
            
            if action is None or action.shape[0] != 7:  
            # if action is None or action.shape[0] != 6:  

                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-1, 1, 7))
                # action = list(np.random.uniform(-1, 1, 6))
            self.agent.set_joint_target_velocities(action)
            self.pr.step()

            ax, ay, az = self.agent_rr_tip.get_position() #tcp position  
            dx, dy, dz = self.target[index].get_position() #target position
            aox, aoy, aoz = self.agent_rr_tip.get_orientation() #tcp orientation
            tox, toy, toz = self.target[index].get_orientation() #target orientation

            sqrt_distance = np.sqrt((100*(ax-dx))**2+(50*(ay-dy))**2+(50*(az-dz))**2)
            distance_error_x = np.abs(ax-dx)
            sqrt_orientation = np.sqrt((50*(aox-tox)**2) + (50*(aoy-toy)**2) +(50*(aoz-toz)**2))
            
            reward -= (sqrt_distance+sqrt_orientation)

            if sqrt_distance > 1000 or sqrt_orientation>1000:
                isDone = True
                done = True
                reward -= 1000
            # return self._get_state(), reward, done, {'finished': isDone}
            return distance_error_x


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


if __name__ == '__main__':
    CONTROL_MODE = 'joint_velocity'  # 'end_position' or 'joint_velocity'
    # CONTROL_MODE = 'end_position'  # 'end_position' or 'joint_velocity'
    env = WorkEnv(control_mode=CONTROL_MODE)
    for eps in range(100):
        print('Starting episode %d' % eps)
        env.reset()
        x_error = env.get_x_error
        input("enter :")
        for step in range(100):
            if CONTROL_MODE == 'joint_velocity':
                action = np.random.uniform(-3.14, 3.14, 7)
                # action = np.random.uniform(-3.14, 3.14, 6)

            else:
                raise NotImplementedError
            try:
                env.step(action,step)
            except KeyboardInterrupt:
                print('Shut Down!')
    env.shutdown()
