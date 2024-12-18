#////////////////////////////////////////////////////////////////////////////////////////////////

# IK class and functions
# Class for defining robot workspace, functionality is currently pretty limited

#////////////////////////////////////////////////////////////////////////////////////////////////

from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
import pinocchio

class IK_Solver():

    def __init__(self, urdf_path, workspace, end_effector_joint_index, dummy_joint = True, eps = 1e-4, max_steps = 1000, dt = 1e-1, damp = 1e-12):
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        """for name, oMi in zip(self.model.names, self.data.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
        exit()"""

        self.workspace = workspace
        self.eps = eps
        self.max_steps = max_steps
        self.dt = dt
        self.damp = damp
        self.joint_id = end_effector_joint_index
        self.dummy_joint = dummy_joint
    
    def solve(self, goal_pos, goal_mat, q):
        if self.dummy_joint is True:
            q=np.hstack((q,[0]))
        goal_pos = self.workspace.clip_to_workspace(goal_pos)
        target_tform = pinocchio.SE3(goal_mat, goal_pos)

        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.joint_id].actInv(target_tform)

            err = pinocchio.log(iMd).vector  # in joint frame
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.max_steps:
                success = False
                break
            i += 1
            
            J = pinocchio.computeJointJacobian(self.model, self.data, q, self.joint_id)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.dt)

        #print("Convergence achieved!" if success else "failed to converge")

        if self.dummy_joint is True:    
            return q.flatten().tolist()[:-1]
        else:
            return q.flatten().tolist()
    
class Robot_Workspace:
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.center = kwargs.get("center")
        if self.shape == "sphere":
            self.radius = kwargs.get("radius")
        elif self.shape == "box":
            self.width = kwargs.get("width")
            self.length = kwargs.get("length")
            self.height = kwargs.get("height")
    def clip_to_workspace(self, point):
        if self.shape == "sphere":
            if np.linalg.norm(point - self.center) <= self.radius:
                return point
            else:
                print("bounding box clipping")
                return self.center + (point - self.center) / np.linalg.norm(point - self.center) * self.radius
            

















