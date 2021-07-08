#!/usr/bin/python

# Copyright (c) 2013-2014, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import PyKDL

import rospy
import rospkg

from ur_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from ur_control import transformations

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

# Set constants for links
BASE_LINK = 'base_link'
SHOULDER_LINK = 'shoulder_link'
UPPER_ARM_LINK = 'upper_arm_link'
FOREARM_LINK = 'forearm_link'
WRIST_1_LINK = 'wrist_1_link'
WRIST_2_LINK = 'wrist_2_link'
WRIST_3_LINK = 'wrist_3_link'
EE_LINK = 'ur3_robotiq_85_gripper'
EE_LINK = 'ee_link'

# Only edit these when editing the robot joints and links.
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK]


def frame_to_list(frame):
    pos = frame.p
    rot = PyKDL.Rotation(frame.M)
    rot = rot.GetQuaternion()
    return np.array([pos[0], pos[1], pos[2],
                     rot[0], rot[1], rot[2], rot[3]])


class ur_kinematics(object):
    """
    UR Kinematics with PyKDL
    """

    def __init__(self, base_link=None, ee_link=None, robot=None, prefix=None, rospackage=None):
        if robot:
            rospack = rospkg.RosPack()
            rospackage_ = rospackage if rospackage is not None else 'ur_pykdl'
            pykdl_dir = rospack.get_path(rospackage_)
            TREE_PATH = pykdl_dir + '/urdf/' + robot + '.urdf'
            self._ur = URDF.from_xml_file(TREE_PATH)
        else:
            self._ur = URDF.from_parameter_server()

        self._kdl_tree = kdl_tree_from_urdf_model(self._ur)
        self._base_link = BASE_LINK if base_link is None else base_link

        self._tip_link = EE_LINK if ee_link is None else ee_link
        self._tip_frame = PyKDL.Frame()
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)

        # UR Interface Limb Instances
        self._joint_names = JOINT_ORDER if prefix is None else [prefix + joint for joint in JOINT_ORDER]
        self._num_jnts = len(self._joint_names)

        # KDL Solvers
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain,
                                                   self._fk_p_kdl,
                                                   self._ik_v_kdl)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain,
                                            PyKDL.Vector.Zero())

    def print_robot_description(self):
        nf_joints = 0
        for j in self._ur.joints:
            if j.type != 'fixed':
                nf_joints += 1
        print(("URDF non-fixed joints: %d;" % nf_joints))
        print(("URDF total joints: %d" % len(self._ur.joints)))
        print(("URDF links: %d" % len(self._ur.links)))
        print(("KDL joints: %d" % self._kdl_tree.getNrOfJoints()))
        print(("KDL segments: %d" % self._kdl_tree.getNrOfSegments()))

    def print_kdl_chain(self):
        for idx in range(self._arm_chain.getNrOfSegments()):
            print(('* ' + self._arm_chain.getSegment(idx).getName()))

    def joints_to_kdl(self, type, values):
        kdl_array = PyKDL.JntArray(self._num_jnts)

        cur_type_values = values

        for idx in range(self._num_jnts):
            kdl_array[idx] = cur_type_values[idx]
        if type == 'velocities':
            kdl_array = PyKDL.JntArrayVel(kdl_array)
        return kdl_array

    def kdl_to_mat(self, data):
        mat = np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i, j] = data[i, j]
        return mat

    def end_effector_transform(self, joint_values, tip_link=None):
        pose = self.forward(joint_values, tip_link)
        translation = np.array([pose[:3]])
        transform = transformations.quaternion_matrix(pose[3:])
        transform[:3, 3] = translation
        return transform

    def forward(self, joint_values, tip_link=None):
        if not tip_link or tip_link == self._tip_link:
            return self.forward_position_kinematics(joint_values)

        arm_chain = self._kdl_tree.getChain(self._base_link,
                                            tip_link)
        fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(arm_chain)
        end_frame = PyKDL.Frame()
        fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                           end_frame)
        return frame_to_list(end_frame)

    def forward_position_kinematics(self, joint_values):
        end_frame = PyKDL.Frame()
        self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                                 end_frame)
        return frame_to_list(end_frame)

    def forward_velocity_kinematics(self, joint_velocities):
        end_frame = PyKDL.FrameVel()
        self._fk_v_kdl.JntToCart(self.joints_to_kdl('velocities', joint_velocities),
                                 end_frame)
        return end_frame.GetTwist()

    def inverse_kinematics(self, position, orientation=None, seed=None):
        ik = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        pos = PyKDL.Vector(position[0], position[1], position[2])
        if isinstance(orientation, (np.ndarray, np.generic, list)):
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(orientation[0], orientation[1],
                                 orientation[2], orientation[3])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if isinstance(seed, (np.ndarray, np.generic, list)):
            seed_array.resize(len(seed))
            for idx, jnt in enumerate(seed):
                seed_array[idx] = jnt
        else:
            seed_array = self.joints_to_kdl('positions', None)  # TODO: Fixme

        # Make IK Call
        if orientation.size != 0:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = np.array(list(result_angles))
            return result
        else:
            return None

    def jacobian(self, joint_values=None):
        jacobian = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', joint_values), jacobian)
        return self.kdl_to_mat(jacobian)

    def jacobian_transpose(self, joint_values=None):
        return self.jacobian(joint_values).T

    def jacobian_pseudo_inverse(self, joint_values=None):
        return np.linalg.pinv(self.jacobian(joint_values))

    def inertia(self, joint_values=None):
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        self._dyn_kdl.JntToMass(self.joints_to_kdl('positions', joint_values), inertia)
        return self.kdl_to_mat(inertia)

    def cart_inertia(self, joint_values=None):
        js_inertia = self.inertia(joint_values)
        jacobian = self.jacobian(joint_values)
        return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)
