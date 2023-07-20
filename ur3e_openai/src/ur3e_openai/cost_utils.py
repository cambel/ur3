import numpy as np
from ur_control.constants import DONE, FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND, SPEED_LIMIT_EXCEEDED
from ur_control import spalg


def sparse(self, done):
    if ground_true(self):
        return self.cost_goal
    return 0


def slicing_with_vel(self, obs, done):
    num_dims = 6 if self.target_dims is None else len(self.target_dims)

    distance = np.abs(obs[1]) # np.linalg.norm(obs[:num_dims])  # ,6
    velocity = np.abs(obs[num_dims+1]) # np.linalg.norm(obs[:num_dims])  # ,6
    jerkiness = np.linalg.norm(obs[num_dims*2:num_dims*3])  # 12,18

    cut_completion = np.linalg.norm(obs[num_dims*3:num_dims*3+1])

    max_force_torque = np.array([self.controller.max_force_torque[i] for i in self.target_dims])

    wrench_size = self.wrench_hist_size*num_dims
    force = np.reshape(obs[-wrench_size:], (-1, num_dims)) * max_force_torque
    force = np.average(force, axis=0)
    norm_force_torque = np.linalg.norm(force[:3])

    r_distance = - np.tanh(5.0 * distance)
    r_velocity = np.clip(velocity, 0.0, 1.0) - 1.0 
    r_force = -1/(1 + np.exp(-norm_force_torque/2+3))  # s-shaped penalization, no penalization for lower values, max penalization for high values
    r_jerkiness = -1 * jerkiness  # penalize jerkiness
    r_jerkiness = np.clip(r_jerkiness, -1., 0.)

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    # reward discounted by the percentage of steps left for the episode (encourage faster termination)
    # r_done = self.cost_done + (1-self.step_count/self.steps_per_episode) if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0
    r_done = self.cost_done if self.goal_reached and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    # encourage faster termination
    r_step = self.cost_step

    weights = [self.w_dist, self.w_force, self.w_jerkiness, self.w_velocity]
    w_norm1 = np.linalg.norm(weights, ord=1)

    # reward components
    reward = w_norm1[0]*r_distance + w_norm1[1]*r_force + w_norm1[2]*r_jerkiness + w_norm1[3]*r_velocity
    
    # episode termination reward/penalization
    reward += r_collision + r_done + r_step 
    
    print('r', round(reward, 2), round(r_distance, 2), round(r_force, 2), round(r_jerkiness, 2), round(r_velocity, 2))

    self.cumulated_dist += distance
    self.cumulated_vel += velocity
    self.cumulated_force += norm_force_torque
    self.cumulated_jerk += jerkiness

    return reward, [r_distance, r_force, r_jerkiness, r_velocity, r_step, r_collision, r_done]

def slicing(self, obs, done):
    num_dims = 6 if self.target_dims is None else len(self.target_dims)

    distance = np.abs(obs[1]) # np.linalg.norm(obs[:num_dims])  # ,6
    jerkiness = np.linalg.norm(obs[num_dims*2:num_dims*3])  # 12,18

    cut_completion = np.linalg.norm(obs[num_dims*3:num_dims*3+1])

    max_force_torque = np.array([self.controller.max_force_torque[i] for i in self.target_dims])

    wrench_size = self.wrench_hist_size*num_dims
    force = np.reshape(obs[-wrench_size:], (-1, num_dims)) * max_force_torque
    force = np.average(force, axis=0)
    norm_force_torque = np.linalg.norm(force[:3])

    r_distance = - np.tanh(5.0 * distance)
    r_force = -1/(1 + np.exp(-norm_force_torque/2+3))  # s-shaped penalization, no penalization for lower values, max penalization for high values
    r_jerkiness = -1 * jerkiness  # penalize jerkiness
    r_jerkiness = np.clip(r_jerkiness, -1., 0.)

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    # reward discounted by the percentage of steps left for the episode (encourage faster termination)
    # r_done = self.cost_done + (1-self.step_count/self.steps_per_episode) if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0
    r_done = self.cost_done if self.goal_reached and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    # encourage faster termination
    r_step = self.cost_step

    # encourage complete cut of the material
    r_cut_completion = -1/(1+np.exp((-10)*(cut_completion-0.5))) * self.cost_cut_completion  # use of a sigmoid, low reward for low cut, high reward for complete cut

    weights = [self.w_dist, self.w_force, self.w_jerkiness]
    w_norm1 = np.linalg.norm(weights, ord=1)

    # reward components
    reward = w_norm1[0]*r_distance + w_norm1[1]*r_force + w_norm1[2]*r_jerkiness
    
    # episode termination reward/penalization
    reward += r_collision + r_done + r_step 

    self.cumulated_dist += distance
    self.cumulated_force += norm_force_torque
    self.cumulated_jerk += jerkiness

    # print('r', round(reward, 2), round(r_distance, 2), round(r_force, 2), round(r_jerkiness, 2))
    return reward, [r_distance, r_force, r_jerkiness, 0.0, r_step, r_collision, r_done]


def peg_in_hole(self, obs, done):
    num_dims = 6 if self.target_dims is None else len(self.target_dims)

    distance = np.linalg.norm(obs[:num_dims])  # ,6
    jerkiness = np.linalg.norm(obs[num_dims*2:num_dims*3])  # 12,18
    compliance = np.linalg.norm(np.interp(obs[num_dims*4:num_dims*5], [-1., 1.], [0., 1.])) / np.linalg.norm(np.ones(6))  # last action[24:30]

    max_force_torque = np.array([self.controller.max_force_torque[i] for i in self.target_dims])

    wrench_size = self.wrench_hist_size*num_dims
    force = np.reshape(obs[-wrench_size:], (-1, num_dims)) * max_force_torque
    force = np.average(force, axis=0)
    norm_force_torque = np.linalg.norm(force[:3])

    r_distance = -l1l2(self, distance)
    r_force = -1/(1 + np.exp(-norm_force_torque/2+3))  # s-shaped penalization, no penalization for lower values, max penalization for high values
    r_jerkiness = -1 * jerkiness  # penalize jerkiness
    r_compliance = compliance  # encourage high compliance

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    # reward discounted by the percentage of steps left for the episode (encourage faster termination)
    # r_done = self.cost_done + (1-self.step_count/self.steps_per_episode) if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0
    position_reached = np.all(obs[1:3]*self.max_distance[1:3] < self.position_threshold_cl)
    r_done = self.cost_done if done and position_reached and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    # encourage faster termination
    r_step = self.cost_step

    reward = self.w_dist*r_distance + self.w_force*r_force + self.w_jerkiness*r_jerkiness + self.w_compliance*r_compliance + r_collision + r_done + r_step
    # print('r', round(reward, 4), round(r_distance, 4), round(r_force, 4), round(r_jerkiness, 4), r_done, jerkiness)
    return reward, [r_distance, r_force, r_jerkiness, r_compliance, r_collision, r_done, r_step]


def dense_distance(self, obs, done):
    reward = 0

    r_distance = 1 - np.tanh(10.0 * np.linalg.norm(obs[:6]))

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    position_reached = np.all(obs[:3] < self.position_threshold)
    r_done = self.steps_per_episode - self.step_count if done and position_reached else 0.0
    r_step = self.cost_step

    reward = self.w_dist*r_distance + r_collision + r_done + r_step

    return reward, [r_distance, r_collision, r_done]


def dense_pft(self, obs, done):
    reward = 0

    r_distance = 1 - np.tanh(10.0 * np.linalg.norm(obs[:3]))
    r_orientation = 1 - np.tanh(10.0 * np.linalg.norm(obs[3:6]))
    # If multiple reading, compute average force before applying l1l2 norm
    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (6, -1))
    force = np.average(force, axis=1)
    r_force = 1 - np.tanh(10.0 * np.linalg.norm(force))

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    r_done = self.cost_done + self.steps_per_episode - self.step_count if done else 0.0

    reward = 0.25*r_distance + 0.25*r_orientation + 0.5*r_force + r_collision + r_done
    return reward, [0.25*r_distance, 0.25*r_orientation, 0.5*r_force, r_collision, r_done]


def dense_pdft(self, obs, done):
    reward = 0

    r_position = 1 - np.tanh(10.0 * np.linalg.norm(obs[:3]))
    r_orientation = 1 - np.tanh(10.0 * np.linalg.norm(obs[3:6]))
    r_distance = 0.5*r_position + 0.5*r_orientation
    # If multiple reading, compute average force before applying l1l2 norm
    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (6, -1))
    force = np.average(force, axis=1)
    norm_force = np.linalg.norm(force) if np.linalg.norm(force) > 2 else 0.0
    r_force = -np.tanh(10.0 * norm_force)

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    r_done = self.steps_per_episode - self.step_count if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    reward = self.w_dist*r_distance + self.w_force*r_force + r_collision + r_done + + r_step
    # print("r", self.w_dist*r_distance, self.w_force*r_force, r_collision, r_done)
    return reward, [r_distance, r_force, r_collision, r_done]


def dense_distance_force(self, obs, done):
    reward = 0

    distance = np.linalg.norm(obs[:6])

    r_distance = -distance

    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (-1, 6)) * self.controller.max_force_torque
    force = np.average(force, axis=0)
    norm_force_torque = np.linalg.norm(force[:3])

    r_force = -1/(1 + np.exp(-norm_force_torque/2+3))  # s-shaped penalization, no penalization for lower values, max penalization for high values

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    # reward discounted by the percentage of steps left for the episode (encourage faster termination)
    position_reached = np.all(obs[:3]*self.max_distance[:3] < self.position_threshold_cl)
    r_done = self.cost_done if done and position_reached and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    # encourage faster termination
    r_step = self.cost_step

    reward = self.w_dist*r_distance + self.w_force*r_force + r_collision + r_done + r_step
    # print('r', round(reward, 4), round(self.w_dist*r_distance, 4), round(r_force, 4), r_done)
    return reward, [r_distance, r_force, r_collision, r_done, r_step]


def dense_distance_velocity_force(self, obs, done):
    reward = 0

    distance = np.linalg.norm(obs[:6])
    velocity = np.linalg.norm(obs[6:12])

    # r_distance_velocity = (1-np.tanh(distance*5.)) * (1-velocity) + (velocity*0.5)**2
    r_distance_velocity = (1-distance) * (1-velocity) + (velocity*0.25)**2 - 1

    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (-1, 6)) * self.controller.max_force_torque
    force = np.average(force, axis=0)
    norm_force_torque = np.linalg.norm(force)
    # r_force = - np.tanh(5*norm_force_torque) # monotonic penalization
    r_force = -1/(1 + np.exp(-norm_force_torque/2+3))  # s-shaped penalization, no penalization for lower values, max penalization for high values
    # print(round(r_distance_velocity, 4), round(r_force, 4))

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    # reward discounted by the percentage of steps left for the episode (encourage faster termination)
    # r_done = self.cost_done + (1-self.step_count/self.steps_per_episode) if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0
    position_reached = np.all(obs[:3]*self.max_distance[:3] < self.position_threshold_cl)
    r_done = self.cost_done if done and position_reached and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    # encourage faster termination
    r_step = self.cost_step

    reward = self.w_dist*r_distance_velocity + self.w_force*r_force + r_collision + r_done + r_step
    # print('r', round(reward, 4), round(self.w_dist*r_distance_velocity, 4), round(r_force, 4), r_done)
    return reward, [r_distance_velocity, r_force, r_collision, r_done, r_step]


def dense_factors(self, obs, done):
    # Reward proximity to the goal but discount the contact force perceived
    reward = 0

    r_distance = 1 - np.tanh(10.0 * np.linalg.norm(obs[:6]))

    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (6, -1))
    force = np.average(force, axis=1)
    norm_force_torque = np.linalg.norm(force)
    r_force = 1 - np.tanh(10.0 * norm_force_torque)

    r_collision = self.cost_collision if self.action_result == FORCE_TORQUE_EXCEEDED else 0.0
    r_done = self.steps_per_episode if done and self.action_result != FORCE_TORQUE_EXCEEDED else 0.0

    reward = r_distance * r_force + r_collision + r_done
    return reward, [r_distance * r_force, r_collision, r_done]


def distance(self, obs, norm='standard'):
    pose = obs[:6] / self.max_distance
    distance_norm = None

    if norm == 'standard':
        distance_norm = np.linalg.norm(pose, axis=-1)
    elif norm == 'l1l2':
        distance_norm = l1l2(self, pose)

    if self.cost_positive:
        return max_range(distance_norm, 30, -70)

    return np.interp(distance_norm, [30, -70], [0, 1])


def l1l2(self, dist):
    l1 = self.cost_l1
    l2 = self.cost_l2
    dist = dist
    norm = (0.5 * (dist ** 2) * l2 +
            np.log(self.cost_alpha + (dist ** 2)) * l1)
    return norm.sum()


def weighted_norm(vector, weights=None):
    if weights is None:
        return np.linalg.norm(vector)
    else:
        return np.linalg.norm(vector*weights)


def contact_force(self, obs):
    # If multiple reading, compute average force before applying l1l2 norm
    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (6, -1))
    force = np.average(force, axis=1)
    net_force = l1l2(self, force, weights=[0.35, 0.35, 0.35, 0.1, 0.1, 0.1])  # TODO fix hardcoded param
    return max_range(net_force, 10, -15) if self.cost_positive else np.interp(net_force, [10, -15], [0, 1])


def actions(self):
    last_actions = self.last_actions[:6]
    reward = np.sum(np.sqrt(last_actions**2))
    return max_range(reward, 6) if self.cost_positive else np.interp(reward, [0, self.max_action], [0, 1])


def distance_force_action_step_goal(self, obs, done, norm='l1l2'):
    cdistance = distance(self, obs, norm)
    cforce = contact_force(self, obs)
    cactions = actions(self)

    cost_ws = self.cost_ws / self.cost_ws.sum() if self.cost_ws.sum() != 0 else self.cost_ws

    reward = (np.dot(np.array([cdistance, cforce, cactions]), cost_ws))
    reward = reward * 200. / float(self.steps_per_episode)

    speed_cost = 0
    ik_cost = 0
    collision_cost = 0

    done_reward = self.cost_goal + 100*(1-self.step_count/float(self.steps_per_episode)) if ground_true(self) else 0

    if self.action_result == SPEED_LIMIT_EXCEEDED:
        speed_cost += self.cost_speed_violation

    elif self.action_result == IK_NOT_FOUND:
        ik_cost += self.cost_ik_violation

    elif self.action_result == FORCE_TORQUE_EXCEEDED:
        collision_cost += self.cost_collision

    reward += ik_cost + speed_cost + collision_cost + done_reward

    ### logging ###
    # distance, action, force, step cost, goal, position, contact force
    self.reward_per_step.append([cdistance, cforce, cactions, done_reward, speed_cost, ik_cost, collision_cost])
    return reward


def ground_true(self):
    error = spalg.translation_rotation_error(self.true_target_pose, self.ur3e_arm.end_effector())*1000.0
    return np.linalg.norm(error[:3], axis=-1) < self.distance_threshold


def max_range(value, max_value, min_value=0):
    return np.interp(value, [min_value, max_value], [1, 0])
