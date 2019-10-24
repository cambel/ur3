#!/usr/bin/env python
import os
import rosparam
import rospkg


def load_ros_params(rospackage_name, rel_path_from_package_to_file,
                           yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)
