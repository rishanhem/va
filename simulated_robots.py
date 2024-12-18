#////////////////////////////////////////////////////////////////////////////////////////////////

# Full code to host server and control simulated robot with the app
# Make sure the app has your ip address (defined in contentview)
# *** To run, from terminal do: mjpython simulated_robots.py ***

#////////////////////////////////////////////////////////////////////////////////////////////////

import sys
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
import threading
import socket
from pinocchio_ik_solver import IK_Solver, Robot_Workspace
from transformations import euler_from_quaternion
import os
os.environ['MUJOCO_LOG_FILE'] = '/dev/null' # doesn't seem to work, not sure how to disable mujoco log

class Simulation():

    def __init__(self, port, xml_path, urdf_path, workspace, end_effector_joint_index, dummy_joint = True, dt = 0.02, grav_comp = True, site_name = "attachment_site", home_key_name = "home"):
        self.port = port
        self.urdf_path = urdf_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.home_rob_pos = np.zeros(3)
        self.home_rob_quat = np.zeros(4)
        self.init_phone_pos = np.zeros(3)
        self.init_phone_quat = np.zeros(4)
        self.conj_init_phone_quat = np.zeros(4)
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.zeros(4)
        self.rot = False
        self.end_effector_joint_index = end_effector_joint_index
        self.dummy_joint = dummy_joint

        self.dt = dt
        self.model.opt.timestep = dt

        self.site_name = site_name
        self.site_id = self.model.site(site_name).id

        self.model.body_gravcomp[:] = float(grav_comp)

        self.home_key_id = self.model.key(home_key_name).id

        self.workspace = workspace

        self.buttons = [0,0,0]
        self.reset_phone_origin = True

        self.lock = threading.Lock()
    
    def simulate(self):

        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False) as viewer:

            # Reset the simulation to the initial keyframe.
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
            mujoco.mj_step(self.model, self.data)
            # Initialize the camera view to that of the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

            # Toggle site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            self.home_rob_pos = np.array(self.data.site(self.site_id).xpos.copy())
            mujoco.mju_mat2Quat(self.home_rob_quat, self.data.site(self.site_id).xmat)
            
            self.init_rob_pos = self.home_rob_pos.copy()
            self.init_rob_quat = self.home_rob_quat.copy()

            self.goal_pos = self.init_rob_pos.copy()
            self.goal_quat = self.init_rob_quat.copy()

            ik = IK_Solver(self.urdf_path, self.workspace, self.end_effector_joint_index, self.dummy_joint)

            server_thread = threading.Thread(target=self.run_server, daemon=True)
            server_thread.start()

            current_q = np.array(self.data.qpos.copy())
            self.init_q = current_q.copy()
            while viewer.is_running():

                step_start = time.time()
                mat_goal = np.zeros(9)

                mujoco.mju_quat2Mat(mat_goal, self.goal_quat)
                mat_goal = mat_goal.reshape((3,3))
                #mat_id = np.identity(3)
                #q = ik.solve(self.goal_pos, mat_id, current_q)
                current_q = np.array(self.data.qpos.copy())
                q = ik.solve(self.goal_pos, mat_goal, current_q) #init_q)
                
                for idx, val in enumerate(q):
                    if val - current_q[idx] > .5:
                        q[idx] = current_q[idx] + 0.0
                        print("clipping delta q")
                    elif val - current_q[idx] < -0.5:
                        q[idx] = current_q[idx] - 0.0
                        print("clipping delta q")

                #current_q = np.array(q.copy())

                self.data.qpos = q.copy()
                mujoco.mj_step(self.model, self.data)

                viewer.sync()
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    
    def string_to_tform(self, string):

        number_strings = [s for s in string.split() if s]
        numbers = [float(num) for num in number_strings]

        transl = numbers[:3]
        transl = [-transl[2], -transl[0], transl[1]]
        eulers = numbers[3:6]
        eulers = [-eulers[2], eulers[0], eulers[1]]
        
        new_buttons = numbers[6:9]
        if new_buttons[0] != self.buttons[0]:
            print("0")
            """self.reset_phone_origin = True
            self.reset_rob_origin()"""
            
        if new_buttons[1] != self.buttons[1]:
            print("1")
            self.reset_phone_origin = True
            self.reset_rob()

        if new_buttons[2] != self.buttons[2]:
            print("2")
        self.buttons = new_buttons

        slider = numbers[9:12]
        self.slider = slider[0]*100 + slider[1]*10 + slider[2]

        return transl, eulers
    
    def reset_rob(self):
        self.data.qpos = self.init_q.copy()

        self.init_rob_pos = self.home_rob_pos.copy()
        self.init_rob_quat = self.home_rob_quat.copy()
        self.goal_pos = self.init_rob_pos.copy()
        self.goal_quat = self.init_rob_quat.copy()
    
    """def reset_rob_origin(self):
        self.init_rob_pos = np.array(self.data.site(self.site_id).xpos.copy())
        mujoco.mju_mat2Quat(self.init_rob_quat, self.data.site(self.site_id).xmat)
        self.goal_pos = self.init_rob_pos.copy()
        self.goal_quat = self.init_rob_quat.copy()"""

    def run_server(self, ip_address='192.168.1.8'):
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind the socket to the IP address and port
        server_socket.bind((ip_address, self.port))
        # Start listening for incoming connections
        server_socket.listen(1)
        print(f"Listening on {ip_address}:{self.port}")

        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        with self.lock:
            try:
                while True:
                    if self.reset_phone_origin:
                        self.reset_phone_origin = False
                        data = client_socket.recv(1024)  # Buffer size of 1024 bytes
                        if not data:
                            break
                        self.init_phone_pos, init_phone_eulers = self.string_to_tform(data.decode())
                        mujoco.mju_euler2Quat(self.init_phone_quat, init_phone_eulers, 'XYZ')
                        mujoco.mju_negQuat(self.conj_init_phone_quat, self.init_phone_quat)
                        
                        
                    # Receive data from the client
                    data = client_socket.recv(1024)  # Buffer size of 1024 bytes
                    if not data:
                        break
                    
                    curr_phone_pos, curr_phone_eulers = self.string_to_tform(data.decode())
                    
                    if not self.reset_phone_origin:
                        self.goal_pos = self.init_rob_pos - self.init_phone_pos + curr_phone_pos

                        #print(curr_phone_eulers)
                        #curr_phone_quat = self.init_phone_quat.copy()
                        curr_phone_quat = np.zeros(4)
                        mujoco.mju_euler2Quat(curr_phone_quat, curr_phone_eulers, 'XYZ')
                        
                        delta_phone_quat = np.zeros(4)
                        mujoco.mju_mulQuat(delta_phone_quat, curr_phone_quat, self.conj_init_phone_quat)
                        mujoco.mju_mulQuat(self.goal_quat, self.init_rob_quat, delta_phone_quat)
                        
                        #eulers = euler_from_quaternion(self.goal_quat)
                        #print(eulers)    
            finally:
                client_socket.close()
                server_socket.close()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    # Default robot if no argument is provided
    robot = "lite6"

    # Check if an argument is passed and set the robot accordingly
    if len(sys.argv) > 1:
        robot = sys.argv[1].lower()

    port = 42923

    if robot == "panda":
        xml_path = str(script_dir / "xmls" / "franka_emika_panda" / "scene.xml")
        urdf_path = str(script_dir / "urdfs" / "panda.urdf")
        workspace = Robot_Workspace("sphere", center=[0,0,0], radius=0.855)
        end_effector_joint_index = 8
        dummy_joint = True

    elif robot == "lite6":
        xml_path = str(script_dir / "xmls" / "ufactory_lite6" / "scene.xml")
        urdf_path = str(script_dir / "urdfs" / "lite6.urdf")
        workspace = Robot_Workspace("sphere", center = [0,0,0.2], radius = .4)
        end_effector_joint_index = 6
        dummy_joint = False

    else:
        print(f"Unknown robot: {robot}")
        sys.exit(1)

    sim = Simulation(port, xml_path, urdf_path, workspace, end_effector_joint_index, dummy_joint)
    sim.simulate()
