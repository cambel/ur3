#ifndef KINEMATICS


#include <stdio.h>
#include <stdlib.h>
#include <vector>
// #include <ros/ros.h>
// #include <geometry_msgs/Pose.h>

// #include <tf2/LinearMath/Transform.h>
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2/LinearMath/Vector3.h>

namespace kinematics
{
  float SIGN(float x);
  float NORM(float a, float b, float c, float d);

  void forward(const double* joint_values, double* pose, char res_type = 'm');
  int inverse(const double* pose, double* q_sols);

  // bool isIKSuccess(const std::vector<double> &pose, std::vector<double> &joints, int& numOfSolns);

  // const std::string getRobotName();

  // bool isIkSuccesswithTransformedBase(const geometry_msgs::Pose& base_pose, const geometry_msgs::Pose& grasp_pose, std::vector<double>& joint_soln,
  //                                     int& numOfSolns);
}

#endif  // KINEMATICS