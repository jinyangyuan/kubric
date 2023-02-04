# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import bpy
import copy
import json
import kubric as kb
import logging
import numpy as np
import os
from imageio import imwrite
from kubric.randomness import random_rotation
from kubric.renderer import Blender
from kubric.simulator import PyBullet


parser = kb.ArgumentParser()
parser.add_argument('--seed_offset', type=int, default=0)
parser.add_argument('--min_objects', type=int, default=3)
parser.add_argument('--max_objects', type=int, default=6)
parser.add_argument('--scene_idx', type=int, default=0)
parser.add_argument('--min_pixels', type=int, default=128)
parser.add_argument('--max_trials', type=int, default=100)
parser.add_argument('--min_scale', type=float, default=1.25)
parser.add_argument('--max_scale', type=float, default=2.5)
parser.add_argument('--min_theta', type=float, default=0.)
parser.add_argument('--max_theta', type=float, default=1.)
parser.add_argument('--min_phi', type=float, default=0.35)
parser.add_argument('--max_phi', type=float, default=0.7)
parser.add_argument('--min_rho', type=float, default=7.)
parser.add_argument('--max_rho', type=float, default=9.)
parser.add_argument('--min_x', type=float, default=-3.)
parser.add_argument('--max_x', type=float, default=3.)
parser.add_argument('--min_y', type=float, default=-3.)
parser.add_argument('--max_y', type=float, default=3.)
parser.add_argument('--kubasic_assets', type=str, default='../downloads/assets/KuBasic/KuBasic.json')
parser.add_argument('--hdri_assets', type=str, default='../downloads/assets/HDRI_haven/HDRI_haven.json')
parser.add_argument('--gso_assets', type=str, default='../downloads/assets/GSO/GSO.json')
parser.add_argument('--bck_indices', type=str, default='./bck_indices.json')
parser.add_argument('--obj_indices', type=str, default='./obj_indices.json')
parser.set_defaults(frame_end=1, resolution=128)
FLAGS = parser.parse_args()
FLAGS.seed = FLAGS.seed_offset + FLAGS.scene_idx + 1
FLAGS.job_dir = '{}_{}'.format(FLAGS.min_objects, FLAGS.max_objects)

if not os.path.exists(FLAGS.job_dir):
  os.mkdir(FLAGS.job_dir)
folder_out = os.path.join(FLAGS.job_dir, 'scene_{}'.format(FLAGS.scene_idx))
try:
  os.mkdir(folder_out)
except:
  exit()

scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=64)
if bpy.app.version < (2, 78, 0):
  bpy.context.user_preferences.system.compute_device_type = 'CUDA'
  bpy.context.user_preferences.system.compute_device = 'CUDA_0'
else:
  bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
hdri = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
with open(FLAGS.bck_indices, 'r') as f:
  bck_indices = json.load(f)
with open(FLAGS.obj_indices, 'r') as f:
  obj_indices = json.load(f)
bck_assets_all = [*sorted(hdri._assets.keys())]
obj_assets_all = [*sorted(gso._assets.keys())]
bck_assets_sel = [bck_assets_all[idx] for idx in bck_indices]
obj_assets_sel = [obj_assets_all[idx] for idx in obj_indices]

logging.info('Setting up the camera')
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
theta = 2 * np.pi * rng.uniform(FLAGS.min_theta, FLAGS.max_theta)
phi = 0.5 * np.pi * rng.uniform(FLAGS.min_phi, FLAGS.max_phi)
rho = rng.uniform(FLAGS.min_rho**3, FLAGS.max_rho**3) ** (1/3.)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
scene.camera.position = (rho * cos_phi * cos_theta, rho * cos_phi * sin_theta, rho * sin_phi)
scene.camera.look_at((0, 0, 0))

logging.info('Adding background')
bck_id = rng.choice(bck_assets_sel)
bck = hdri.create(asset_id=bck_id)
scene.metadata['background'] = bck_id
renderer._set_ambient_light_hdri(bck.filename)
dome = kubasic.create(asset_id='dome', name='dome', static=True, background=True)
scene.add(dome)
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes['Image Texture']
texture_node.image = bpy.data.images.load(bck.filename)

num_objects = rng.randint(FLAGS.min_objects, FLAGS.max_objects + 1)
scene.metadata['num_instances'] = num_objects
obj_list = []
num_retry_scene = 0
logging.info('Adding objects (%s retries)', num_retry_scene)
while len(obj_list) < num_objects:
  obj_id = rng.choice(obj_assets_sel)
  obj = gso.create(asset_id=obj_id)
  scale = rng.uniform(FLAGS.min_scale, FLAGS.max_scale)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata['scale'] = scale
  scene.add(obj)
  obj_list.append(obj)
  spawn_region = np.array([[FLAGS.min_x, FLAGS.min_y, 1.], [FLAGS.max_x, FLAGS.max_y, 1.]], dtype=float) - obj.aabbox
  spawn_region[1][2] = spawn_region[0][2]
  for num_trials in range(FLAGS.max_trials):
    logging.info('    Adding %s / %s objects (%s retries)', len(obj_list), num_objects, num_trials)
    obj.quaternion = random_rotation(axis='z', rng=rng)
    obj.position = rng.uniform(*spawn_region)
    if simulator.check_overlap(obj):
      continue
    data_stack = renderer.render(return_layers=['segmentation'])
    kb.compute_visibility(data_stack['segmentation'], scene.assets)
    num_pixels_list = [np.sum(asset.metadata['visibility']) for asset in scene.foreground_assets]
    if np.min(num_pixels_list) < FLAGS.min_pixels:
      continue
    logging.info('    Added %s / %s objects', len(obj_list), num_objects)
    break
  else:
    for obj in obj_list:
      scene.remove(obj)
      del obj
    obj_list.clear()
    num_retry_scene += 1
    logging.info('Adding objects (%s retries)', num_retry_scene)
for obj in obj_list:
  obj.position = np.array([*obj.position[:2], obj.position[2] - 1], dtype=float)

logging.info('Rendering the scene')
data_stack = renderer.render(return_layers=['rgba', 'segmentation'])
data_stack['segmentation'] = kb.adjust_segmentation_idxs(data_stack['segmentation'], scene.assets, obj_list)
image = data_stack['rgba'].squeeze(0)[..., :3]
segmentation = data_stack['segmentation'].squeeze(0).squeeze(-1).astype(np.uint8)
imwrite(os.path.join(folder_out, 'image.png'), image)
imwrite(os.path.join(folder_out, 'segmentation.png'), segmentation)
for obj in obj_list:
  scene.remove(obj)
for idx, obj in enumerate(obj_list):
  obj_new = copy.copy(obj)
  scene.add(obj_new)
  data_stack = renderer.render(return_layers=['segmentation'])
  data_stack['segmentation'] = kb.adjust_segmentation_idxs(data_stack['segmentation'], scene.assets, [obj_new])
  mask = data_stack['segmentation'].squeeze(0).squeeze(-1).astype(np.uint8)
  imwrite(os.path.join(folder_out, 'mask_{}.png'.format(idx)), mask)
  scene.remove(obj_new)
kb.write_json({'flags': vars(FLAGS), 'num_objects': num_objects}, os.path.join(folder_out, 'metadata.json'))

kb.done()
