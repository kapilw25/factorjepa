import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import json, re, ast, math
import numpy as np
import trimesh
import pyrender
from pygltflib import GLTF2
from PIL import Image

# ---------------------------------------------------------
# RENDER
# ---------------------------------------------------------
def load_glb_scene(glb_path):
    scene = trimesh.load(glb_path + '.glb', force='scene')
    bounds = scene.bounds
    extents = scene.extents

    print("\nGLB bounds:", bounds)
    print("GLB size:", extents)

    return scene, bounds, extents

def render_topdown(glb_path):

    mesh, bounds, extents = load_glb_scene(glb_path)
    
    scene = pyrender.Scene()

    if isinstance(mesh, trimesh.Scene):
        # If the GLB contains multiple meshes
        pyrender_mesh = pyrender.Scene.from_trimesh_scene(mesh)
    else:
        # Single mesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # --- Create the scene ---
    scene = pyrender.Scene()

    if isinstance(pyrender_mesh, pyrender.Scene):
        # If converted from a trimesh.Scene
        for node in pyrender_mesh.get_nodes():
            scene.add_node(node)
    else:
        scene.add(pyrender_mesh)


    # -----------------------------
    # Camera + Light
    # -----------------------------
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3)

    # top down view

    scene.add(camera, pose=np.array([
        [1,0,0,0],
        [0,0,1,1.5],
        [0,-1,0,0],
        [0,0,0,1]
    ]))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=np.eye(4))

    # -----------------------------
    # View Scene
    # -----------------------------
    pyrender.Viewer(scene, use_raymond_lighting=True)

def render_frontal(glb_path):

    mesh, bounds, extents = load_glb_scene(glb_path)
    
    scene = pyrender.Scene()

    if isinstance(mesh, trimesh.Scene):
        # If the GLB contains multiple meshes
        pyrender_mesh = pyrender.Scene.from_trimesh_scene(mesh)
    else:
        # Single mesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # --- Create the scene ---
    scene = pyrender.Scene()

    if isinstance(pyrender_mesh, pyrender.Scene):
        # If converted from a trimesh.Scene
        for node in pyrender_mesh.get_nodes():
            scene.add_node(node)
    else:
        scene.add(pyrender_mesh)

      # scene.add(sphere, pose=T)

    # -----------------------------
    # Camera + Light
    # -----------------------------
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3)

    # angular view
    scene.add(camera, pose=np.array([
        [1,0,0,0],
        [0,1,1.5,1],
        [0,0,1,1.5],
        [0,0,0,1]
    ]))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=np.eye(4))

    # -----------------------------
    # View Scene
    # -----------------------------
    pyrender.Viewer(scene, use_raymond_lighting=True)

glb_path = '4.2.sbg.dining_room'

scene = trimesh.load(glb_path + '.glb')

# Load your GLB scene and render the path inside it
render_frontal(glb_path)
plt.show()

render_topdown(glb_path)
plt.show()