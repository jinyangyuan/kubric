# Copyright 2020 Google LLC
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
""" Helloworld for 3D visualization in Tensorflow Graphics.

Inspired by https://threejs.org/docs/#manual/en/introduction/Creating-a-scene.

WARNING: this needs to be executed in either of two ways
  1) Via the python REPL inside blender as `blender --background --python helloworld.py`
  2) Via `pip install bpy` via https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule
"""

# TODO: tensorflow_graphics is not installed in Blender's REPL (also this is from source)
import os, sys
sys.path.append("/Users/atagliasacchi/dev/graphics")

# --- note: this selects blender as the standard backend
import tensorflow_graphics.viewer.blender as THREE

# --- renderer & scene
renderer = THREE.Renderer()
renderer.set_size(640,480)
scene = THREE.Scene()

# --- camera
camera = THREE.OrthographicCamera()
camera.position = (2, 2, 2)
camera.look_at(0, 0, .25)

# --- ambient light
amb_light = THREE.AmbientLight(color=0x0000FF, intensity=1)
scene.add(amb_light)

# --- sunlight
dir_light = THREE.DirectionalLight(color=0xFFFFFF, intensity=2)
dir_light.position = (.5, -.5, 2)
dir_light.look_at(0,0,0)
scene.add(dir_light)

# --- instantiate object
geometry = THREE.BoxGeometry()
geometry.scale = (.1, .1, .1)
geometry.position = (0, .2, .05)
material = THREE.MeshFlatMaterial()
cube = THREE.Mesh(geometry, material)
scene.add(cube)

# --- raw mesh object
import trimesh
import numpy as np
url = "https://storage.googleapis.com/tensorflow-graphics/public/spot.ply"
mesh = trimesh.load_remote(url)
faces = np.array(mesh.faces)
vertices = np.array(mesh.vertices)
# --- mesh from vertices/faces
import mathutils
geometry = THREE.BufferGeometry()
geometry.set_index(faces)
geometry.set_attribute("position", THREE.Float32BufferAttribute(vertices,3))
material = THREE.MeshPhongMaterial()
spot = THREE.Mesh(geometry, material)
spot.position = (-0.14, 0.22, 0)
spot.scale = (.5, .5, .5)
spot.quaternion = mathutils.Quaternion((1,0,0), np.pi/2) 
scene.add(spot)

# --- Invisible ground
geometry = THREE.PlaneGeometry()
geometry.scale = (2, 2, 2)
material = THREE.ShadowMaterial()
floor = THREE.Mesh(geometry, material)
scene.add(floor)

# --- render to PNG or save .blend file (according to extension)
renderer.render(scene, camera, path="helloworld.png")