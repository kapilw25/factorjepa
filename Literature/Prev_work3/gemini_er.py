from google import genai
from google.genai import types
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



# Initialize the GenAI client and specify the model
MODEL_ID = "gemini-robotics-er-1.5-preview"
text_input = '"Go and sit on the nearest chair."'
PROMPT = """You are given a top-down image and a frontal image of a scene. Your task is to 
          predict a navigable path following the instructions: """ + text_input + """
          Determine the ranges of all objects like furniture, 
          automobile etc. Plan the shortest path that avoids obstacles 
          in the first, top-down image. Carpets and roads can be walked on. Treat the first, 
          top-down image as a 25x25x25 grid for path calculation and the bottom left behind 
          corner of the floor is (0, 0, 0). XZ plane is the floor. +X axis goes right, 
          +Y axis goes up, and +Z axis comes front, outside the screen. Output the path 
          in the form of coordinates list: Provide an ordered sequence of (x, y, z) coordinates 
          representing the path across the image for an agent to walk, while avoiding all 
          obstacles. No points should lie on any object or obstacles but near it. The output format 
          should be in json as follows: [{"point": [x0, y0, z0], "label":}. Y should be the height of the floor.]
          """

print("prompt:", PROMPT)
client = genai.Client(api_key='AIzaSyBBVox9Qx03FoNV6ZVx-0Fm9HrxeEChTCQ')

glb_path = 'Others/34_Pedestrian mall with.4'

# Load your image
with open("images/" + glb_path + "_2.png", 'rb') as f:
    image_bytes4 = f.read()

with open("images/" + glb_path + "_1.png", 'rb') as f:
    image_bytes5 = f.read()

image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=image_bytes4,
            mime_type='image/png',
        ),
        types.Part.from_bytes(
            data=image_bytes5,
            mime_type='image/png',
        ),
        PROMPT
    ],
    config = types.GenerateContentConfig(
        temperature=1.0,
        thinking_config=types.ThinkingConfig(thinking_budget=2000)
    )
)

print(image_response.text)

json_text = image_response.text.split("```json")[1].split("```")[0].strip()
data = json.loads(json_text)

# Extract points as tuples
path = [tuple(item["point"]) for item in data]

print(path)

# ---------------------------------------------------------
# LOAD GLB
# ---------------------------------------------------------
def load_glb_scene(glb_path):
    scene = trimesh.load("glb/" + glb_path + '.glb', force='scene')
    bounds = scene.bounds
    extents = scene.extents

    print("\nGLB bounds:", bounds)
    print("GLB size:", extents)

    return scene, bounds, extents



def make_sphere(radius=0.02, color=[1.0, 0.0, 0.0]):
    """Return a pyrender.Mesh sphere with given color."""
    sphere = trimesh.primitives.Sphere(radius=radius)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color + [1.0]      # RGBA
    )
    return pyrender.Mesh.from_trimesh(sphere, material=material)

def map_path_0_24_to_scene(path_points, glb_mesh):
    """
    Map a path with coordinates inside [0, 24]
    into the bounding box of the GLB mesh.
    """
    path_points = np.asarray(path_points)

    # scene bounding box
    if isinstance(glb_mesh, trimesh.Scene):
        mesh = glb_mesh.dump(concatenate=True)
    else:
        mesh = glb_mesh

    scene_min, scene_max = mesh.bounds
    scene_size = scene_max - scene_min

    # scale factor from [0,24] cube to scene bounding box
    scale = scene_size / 24.0

    # transformed path
    transformed = path_points * scale + scene_min

    return transformed

# ---------------------------------------------------------
# RENDER
# ---------------------------------------------------------
def render_path_in_glb(glb_path, path):

    mesh, bounds, extents = load_glb_scene(glb_path)
    # unit_scale = compute_unit_scale(extents)

    # scaled_path = scale_and_place_path(path, unit_scale, bounds)
    # # tube_mesh = create_tube_mesh(scaled_path)
    # ticks_mesh = create_xz_ticks(unit_scale, bounds)

    # pr_scene = pyrender.Scene()

    # # add GLB geometry
    # for geom in scene_trimesh.geometry.values():
    #     pr_scene.add(pyrender.Mesh.from_trimesh(geom))

    # # add path & ticks
    # # pr_scene.add(pyrender.Mesh.from_trimesh(tube_mesh))
    # pr_scene.add(pyrender.Mesh.from_trimesh(ticks_mesh))

    # debug_axes = create_debug_axes(length=1.0)
    # pr_scene.add(pyrender.Mesh.from_trimesh(debug_axes))


    # -----------------------------------------------------------------
    # FRONT CAMERA (Y-UP)
    # ----------------------------------------------------------------

    # --- Create the scene ---
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

    for i, p in enumerate(path):
      if (i == 0):
        sphere = make_sphere(radius=0.02, color=[1,0,0])  # red spheres
      elif (i > 0):
        sphere = make_sphere(radius=0.02, color=[0,0,1])

      T = np.eye(4)
      T[:3, 3] = p         # place sphere at trajectory point
      scene.add(sphere, pose=T)

    # -----------------------------
    # Camera + Light
    # -----------------------------
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3)
    # frontal view
    # scene.add(camera, pose=np.array([
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,1,1.5],
    #     [0,0,0,1]
    # ]))

    # angular view
    # scene.add(camera, pose=np.array([
    #     [1,0,0,0],
    #     [0,1,1.5,1],
    #     [0,0,1,1.5],
    #     [0,0,0,1]
    # ]))

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
    # renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    # color, depth = renderer.render(scene, use_raymond_lighting=True)

    # # renderer.delete()
    # Image.fromarray(color).save("images/"+glb_path+"_2.png")
    print('saved')

scene = trimesh.load("glb/" + glb_path + '.glb')

# path = ([0, 2, 0], [0, 2, 24], [24, 2, 0], [24, 2, 24])
# path = [(9, 0, 22), (11, 0, 21), (12, 0, 21), (14, 0, 21)]
mapped_path = map_path_0_24_to_scene(path, scene)
print('mapped_path', mapped_path)
# Load your GLB scene and render the path inside it
render_path_in_glb(glb_path, mapped_path)
plt.show()