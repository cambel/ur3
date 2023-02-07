# Basic models for dynamic spawn/despawn
#
# Author: Cristian C Beltran-Hernandez

import numpy as np

SPHERE = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="{}">
    <static>true</static>
    <link name="link">
      <pose>0 0 0 0 0 0</pose>
      <visual name="visual">
        <transparency> 0.5 </transparency>
        <geometry>
          <sphere>
            <radius>{}</radius>
          </sphere>
        </geometry>
        <material>
          <script>
            <uri>model://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/{}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

SPHERE_COLLISION = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="{}">
    <static>true</static>
    <link name="link">
      <pose>0 0 0 0 0 0</pose>
      <visual name="visual">
        <transparency> 0.5 </transparency>
        <geometry>
          <sphere>
            <radius>{}</radius>
          </sphere>
        </geometry>
        <material>
          <script>
            <uri>model://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/{}</name>
          </script>
        </material>
      </visual>
      <collision name="collision">
        <pose frame=''>0 0 0 0 0 0</pose>
        <laser_retro>0</laser_retro>
        <max_contacts>10</max_contacts>
        <geometry>
          <sphere>
            <radius>{}</radius>
          </sphere>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1</mu>
              <mu2>1</mu2>
              <fdir1>0 0 0</fdir1>
              <slip1>0.1</slip1>
              <slip2>0.1</slip2>
            </ode>
            <torsional>
              <coefficient>1</coefficient>
              <patch_radius>0</patch_radius>
              <surface_radius>0</surface_radius>
              <use_patch_radius>1</use_patch_radius>
              <ode>
                <slip>0</slip>
              </ode>
            </torsional>
          </friction>
          <bounce>
            <restitution_coefficient>0</restitution_coefficient>
            <threshold>1e+06</threshold>
          </bounce>
          <contact>
            <collide_without_contact>0</collide_without_contact>
            <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
            <collide_bitmask>1</collide_bitmask>
            <ode>
              <kp>{}</kp>
              <kd>1</kd>
              <max_vel>0.01</max_vel>
              <min_depth>0</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>
    </link>
  </model>
</sdf>"""


def get_box_model(
        name, size, mass=1, color=[1, 0, 0, 0],
        mu=1, mu2=1, slip1=0.001, slip2=0.001, kp=1e10, kd=1, max_vel=100.0):
    inertia = mass*(size**2)*np.array([[2./3., -1./4., -1./4.], [-1./4., 2./3., -1./4.], [-1./4., -1./4., 2./3.]])
    return BOX.format(
        model_name=name, x=size / 2., y=size / 2., z=size / 2., size=size, mass=mass, mu=mu, mu2=mu2, slip1=slip1,
        slip2=slip2, kp=kp, kd=kd, max_vel=max_vel, ixx=inertia[0, 0],
        ixy=inertia[0, 1],
        ixz=inertia[0, 2],
        iyy=inertia[1, 1],
        iyz=inertia[1, 2],
        izz=inertia[2, 2],
        r=color[0],
        g=color[1],
        b=color[2],
        transparency=color[3])


BOX = """<?xml version="1.0" ?> 
<sdf version="1.5"> 
  <model name="{model_name}"> 
    <link name="link"> 
      <pose>{x} {y} {z} 0 0 0</pose> 
      <inertial> 
        <mass>{mass}</mass> 
        <inertia> 
          <ixx>{ixx}</ixx> 
          <ixy>{ixy}</ixy> 
          <ixz>{ixz}</ixz> 
          <iyy>{iyy}</iyy> 
          <iyz>{iyz}</iyz> 
          <izz>{izz}</izz> 
        </inertia> 
      </inertial> 
      <visual name="visual"> 
        <geometry> 
          <box> 
            <size>{size} {size} {size}</size> 
          </box> 
        </geometry> 
        <transparency> {transparency} </transparency>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
          <specular>0 0 0 0</specular>
          <emissive>{r} {g} {b} 1</emissive>
        </material>
      </visual> 
      <collision name="boxcollision"> 
        <pose frame=''>0 0 0 0 0 0</pose> 
        <laser_retro>0</laser_retro> 
        <max_contacts>10</max_contacts> 
        <geometry> 
          <box> 
            <size>{size} {size} {size}</size> 
          </box> 
        </geometry> 
        <surface> 
          <friction> 
            <ode> 
              <mu>{mu}</mu> 
              <mu2>{mu2}</mu2> 
              <fdir1>0 0 0</fdir1> 
              <slip1>{slip1}</slip1> 
              <slip2>{slip2}</slip2> 
            </ode> 
            <torsional> 
              <coefficient>1</coefficient> 
              <patch_radius>0</patch_radius> 
              <surface_radius>0.1</surface_radius> 
              <use_patch_radius>1</use_patch_radius> 
              <ode> 
                <slip>0</slip> 
              </ode> 
            </torsional> 
          </friction> 
          <bounce> 
            <restitution_coefficient>0</restitution_coefficient> 
            <threshold>1e+06</threshold> 
          </bounce> 
          <contact>
            <poissons_ratio>0.347</poissons_ratio>
            <elastic_modulus>8.8e+09</elastic_modulus>
            <collide_without_contact>0</collide_without_contact> 
            <collide_without_contact_bitmask>1</collide_without_contact_bitmask> 
            <collide_bitmask>1</collide_bitmask> 
            <ode> 
              <kp>{kp}</kp> 
              <kd>{kd}</kd> 
              <max_vel>{max_vel}</max_vel> 
              <min_depth>0.001</min_depth> 
            </ode> 
          </contact> 
        </surface> 
      </collision> 
    </link> 
  </model> 
</sdf>"""


def get_peg_board_model(kp=1e5, mu=1, mu2=1, color=[0, 0, 1, 0], scale=1.0, peg_shape='cuboid'):
    return PEG_BOARD.format(
        mu=mu, mu2=mu2, kp=kp, r=color[0],
        g=color[1],
        b=color[2],
        transparency=color[3],
        scale=scale,
        peg_shape=peg_shape)


PEG_BOARD = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='simple_peg_board'>
    <static>true</static>
    <link name='board'>
      <pose frame=''>0.0 0.0 0.0 1.5707 0 0</pose>
      <self_collide>0</self_collide>
      <kinematic>0</kinematic>
      <gravity>0</gravity>
      <inertial>
        <mass>1</mass>
        <pose frame=''>0 0 0 0 0 0</pose>
        <inertia>
          <ixx>0.999223</ixx>
          <ixy>0.039421</ixy>
          <ixz>0.000141</ixz>
          <iyy>0.999222</iyy>
          <iyz>-0.001474</iyz>
          <izz>0.999999</izz>
        </inertia>
      </inertial>
      <visual name='visual'>
        <pose frame=''>0 0 0 1.5707 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://simple_peg_board/meshes/board-{peg_shape}.stl</uri>
              <scale>{scale} {scale} {scale}</scale>
          </mesh>
        </geometry>
        <transparency> {transparency} </transparency>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <specular>0.2 0.2 0.2 64</specular>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>0.1 0 0.1 1</emissive>
        </material>
      </visual>
      <collision name='collision'>
        <pose frame=''>0 0 0 1.5707 0 0</pose>
        <laser_retro>0</laser_retro>
        <max_contacts>10</max_contacts>
        <geometry>
          <mesh>
            <uri>model://simple_peg_board/meshes/board-{peg_shape}.stl</uri>
            <scale>{scale} {scale} {scale}</scale>
          </mesh>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{mu}</mu>
              <mu2>{mu2}</mu2>
              <fdir1>0 0 0</fdir1>
              <slip1>0.1</slip1>
              <slip2>0.1</slip2>
            </ode>
            <torsional>
              <coefficient>1</coefficient>
              <patch_radius>0</patch_radius>
              <surface_radius>0</surface_radius>
              <use_patch_radius>1</use_patch_radius>
              <ode>
                <slip>0</slip>
              </ode>
            </torsional>
          </friction>
          <bounce>
            <restitution_coefficient>0</restitution_coefficient>
            <threshold>1e+06</threshold>
          </bounce>
          <contact>
            <collide_without_contact>0</collide_without_contact>
            <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
            <collide_bitmask>1</collide_bitmask>
            <ode>
              <kp>{kp}</kp>
              <kd>1</kd>
              <max_vel>0.01</max_vel>
              <min_depth>0</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>
    </link>
    <static>1</static>
    <allow_auto_disable>1</allow_auto_disable>
  </model>
</sdf>"""


def get_button_model(spring_stiffness=-500., damping=0., friction=0., base_mass=1., button_mass=0.1, color=[1, 0, 0, 0],
                     kp=1e5, kd=1, max_vel=100.0):
    """ Create a String SDF model of a simple button
    Spring_stiffness must be negative.
    """
    base_inertia = base_mass*(0.06**2)*np.array([[2./3., -1./4., -1./4.],
                                                 [-1./4., 2./3., -1./4.], [-1./4., -1./4., 2./3.]])
    button_inertia = button_mass*(0.05**2)*np.array([[2./3., -1./4., -1./4.],
                                                     [-1./4., 2./3., -1./4.], [-1./4., -1./4., 2./3.]])
    return BUTTON.format(base_mass=base_mass, button_mass=button_mass,
                         kp=kp, kd=kd, max_vel=max_vel,
                         spring_stiffness=abs(spring_stiffness)*-1,  # always negative values
                         damping=damping, friction=friction,
                         base_ixx=base_inertia[0, 0], base_ixy=base_inertia[0, 1], base_ixz=base_inertia[0,
                                                                                                         2], base_iyy=base_inertia[1, 1], base_iyz=base_inertia[1, 2], base_izz=base_inertia[2, 2],
                         button_ixx=button_inertia[0, 0], button_ixy=button_inertia[0, 1], button_ixz=button_inertia[0,
                                                                                                                     2], button_iyy=button_inertia[1, 1], button_iyz=button_inertia[1, 2], button_izz=button_inertia[2, 2],
                         r=color[0], g=color[1], b=color[2], transparency=color[3])


BUTTON = """
<?xml version="1.0"?>
<sdf version="1.5">
  <model name="button">
    <pose>0 0 0.03 0 0 0</pose>
    <link name="base_link">
      <inertial>
        <inertia>
          <ixx>{base_ixx}</ixx>
          <ixy>{base_ixy}</ixy>
          <ixz>{base_ixz}</ixz>
          <iyy>{base_iyy}</iyy>
          <iyz>{base_iyz}</iyz>
          <izz>{base_izz}</izz>
        </inertia>
        <mass>{base_mass}</mass>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.07 0.07 0.06</size>
          </box>
        </geometry>
        <material>
          <script><uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/DarkGrey</name></script>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.07 0.07 0.06</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>10</mu>
              <mu2>10</mu2>
              <fdir1>0 0 0</fdir1>
              <slip1>0</slip1>
              <slip2>0</slip2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
    <joint name="joint" type="prismatic">
      <parent>base_link</parent>
      <child>top_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <limit>
          <lower>-0.020</lower>
          <upper>0.025</upper>
        </limit>
        <xyz>0.0 0.0 1.0</xyz>
        <dynamics>
          <spring_stiffness>{spring_stiffness}</spring_stiffness>
          <spring_reference>-0.025</spring_reference>
          <damping>{damping}</damping>
          <friction>{friction}</friction>
        </dynamics>
      </axis>
    </joint>
    <link name="top_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <inertia>
          <ixx>{button_ixx}</ixx>
          <ixy>{button_ixy}</ixy>
          <ixz>{button_ixz}</ixz>
          <iyy>{button_iyy}</iyy>
          <iyz>{button_iyz}</iyz>
          <izz>{button_izz}</izz>
        </inertia>
        <mass>{button_mass}</mass>
      </inertial>
      <visual name="visual">
        <pose>0 0 0.025 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 0.05 0.05</size>
          </box>
        </geometry>
        <transparency> {transparency} </transparency>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
          <specular>0 0 0 0</specular>
          <emissive>{r} {g} {b} 1</emissive>
        </material>
      </visual>
      <collision name="collision">
        <pose>0 0 0.025 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 0.05 0.05</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <ode> 
              <kp>{kp}</kp> 
              <kd>{kd}</kd> 
              <max_vel>{max_vel}</max_vel> 
              <min_depth>0.001</min_depth> 
            </ode> 
          </contact> 
        </surface> 
      </collision>
    </link>
  </model>
</sdf>
"""
