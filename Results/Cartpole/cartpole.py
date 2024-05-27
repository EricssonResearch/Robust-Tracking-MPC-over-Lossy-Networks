import pybullet as p
import numpy as np
import os

class Cartpole:
    '''
    0 :  b'slider_to_cart'
    1 :  b'cart_to_pole'
    '''
    def __init__(self, timeStep = 1/100, initState=[0, 0, 0, 0], enableGUI=False):
        self.timeStep = timeStep
        self.client = p.connect(p.GUI) if enableGUI else p.connect(p.DIRECT)
        f_name = os.path.join(os.path.dirname(__file__), 'cartpole.urdf')
        self.cartpole = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0],
                              physicsClientId=self.client)
        # Joint indices as found by p.getJointInfo()
        self.cart_ix, self.pole_ix = 0, 1

        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.timeStep, physicsClientId=self.client)
        # Enable torque control per joint, thus disabling direct control of pole
        p.setJointMotorControl2(self.cartpole, self.cart_ix, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)
        p.setJointMotorControl2(self.cartpole, self.pole_ix, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)
        # Set initial states
        p.resetJointState(self.cartpole, self.cart_ix, initState[0], initState[1], physicsClientId=self.client)
        p.resetJointState(self.cartpole, self.pole_ix, initState[2], initState[3], physicsClientId=self.client)

    def get_ids(self):
        return self.cartpole, self.client

    def apply_action(self, action):
        # Set the force to be applied
        p.setJointMotorControl2(self.cartpole, self.cart_ix, p.TORQUE_CONTROL, force=action, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)

    def get_observation(self):
        # Get x, x_dot, theta, theta_dot in the simulation
        state = np.asarray(p.getJointState(self.cartpole, self.cart_ix, physicsClientId=self.client)[0:2] \
                    + p.getJointState(self.cartpole, self.pole_ix, physicsClientId=self.client)[0:2])
        return state

    def reset_state(self,x_reset):
        x_reset.shape = (4,1)
        p.resetJointState(self.cartpole, self.cart_ix, x_reset[0], x_reset[1], physicsClientId=self.client)
        p.resetJointState(self.cartpole, self.pole_ix, x_reset[2], x_reset[3], physicsClientId=self.client)

    def make_gif(self, trajectory, filename="test"):
        from PIL import Image
        RENDER_WIDTH, RENDER_HEIGHT = int(320*4), int(200*4)
        RENDER_FRAME_DURATION = 1/50
        step = int(RENDER_FRAME_DURATION / self.timeStep)
        VIEW_MATRIX = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.2, 0, 0.5],
                                                              distance=1.2,
                                                              yaw=0,
                                                              pitch=0,
                                                              roll=0,
                                                              upAxisIndex=2,
                                                              physicsClientId=self.client)
        PROJ_MATRIX = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                    nearVal=0.1,
                                                    farVal=100.0,
                                                    physicsClientId=self.client)
        frames = []

        for state in trajectory[:,::step].T:
            self.reset_state(state)
            (w, h, rgb, _, _) = p.getCameraImage(width=RENDER_WIDTH,
                                                    height=RENDER_HEIGHT,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                    viewMatrix=VIEW_MATRIX,
                                                    projectionMatrix=PROJ_MATRIX,
                                                    physicsClientId=self.client)
            frames.append(np.asarray(np.reshape(rgb, (h, w, 4)), dtype=np.uint8))

        frames_stacked = np.stack(frames)
        ims = [Image.fromarray(a_frame) for a_frame in frames_stacked]
        ims[0].save(filename+".gif", save_all=True, append_images=ims[1:], loop=0, duration=RENDER_FRAME_DURATION*1000) # duration is the duration of each frame in ms
    
if __name__ == '__main__':
    from time import sleep
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(0.02)
    cartpole = p.loadURDF(os.path.join(os.path.dirname(__file__), 'cartpole.urdf'), basePosition=[0, 0, 0])
    if False: # Set to True to check joint numbers, False for an interactive PyBullet gui
        number_of_joints = p.getNumJoints(cartpole)
        for joint_number in range(number_of_joints):
            info = p.getJointInfo(cartpole, joint_number)
            print(info[0], ": ", info[1])
    else:
        cart_ix, pole_ix = 0, 1
        #Enable torque control per joint, thus disabling direct control of pole
        p.setJointMotorControl2(cartpole, cart_ix, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(cartpole, pole_ix, p.VELOCITY_CONTROL, force=0)
        # Easy manual input
        linear = p.addUserDebugParameter('Linear', -10, 10, 0)
        sleep(3)
        while True:
            user_lin = p.readUserDebugParameter(linear)
            p.setJointMotorControl2(cartpole, cart_ix, p.TORQUE_CONTROL, force=user_lin)
            p.stepSimulation()
            state = p.getJointState(cartpole, cart_ix)[0:2] + p.getJointState(cartpole, pole_ix)[0:2]
            x, x_dot, theta, theta_dot = state
            print("x: ", x)
            print("x_dot: ", x_dot)
            print("theta: ", theta)
            print("theta_dot: ", theta_dot)
            