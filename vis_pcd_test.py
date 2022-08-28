# Author: Julien Fischer
#
# Remark: Most of the pyrender code is taken from SMPLify-X. However, since the original SMPLify-X visualization code resulted in
#         errors on my system, I decided to write a new visualization method that is employed _after_ the fitting pipeline. 
import pyrender
import os
import numpy as np
import trimesh
import argparse
from PIL import Image as pil_img
import matplotlib.pyplot as plt
from pyrender import Node, DirectionalLight
from typing import Tuple, Union, List
from tqdm import tqdm
import shutil
try:
    import cPickle as pickle
except ImportError:
    import pickle


def _create_raymond_lights():
    """Taken from pyrender source code"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(Node(
            light=DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def read_pickle_file(file_location) -> dict:
    """Reads the pickle file and returns it as a dictionary"""
    with open(file_location, "rb") as file:
        pickle_dict = pickle.load(file, encoding="latin1")
    return pickle_dict


def load_mesh(mesh_location: str) -> pyrender.Mesh:
    """Loads a mesh with applied material from the specified file."""

    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )
    body_mesh = trimesh.load(mesh_location)
    mesh = pyrender.Mesh.from_trimesh(
        body_mesh,
        material=material)

    return mesh


def get_scene_render(body_mesh: pyrender.Mesh,
                     image_width: int,
                     image_height: int,
                     camera_center: np.ndarray, 
                     camera_translation: np.ndarray,
                     camera_focal_length: float = 5000,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Renders the scene and returns the color and depth output"""

    import open3d as o3d
    pcd = o3d.io.read_point_cloud("frame0.ply")
    body_mesh = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors))

    fuze_trimesh = trimesh.load('frame0.ply')
    body_mesh = pyrender.Mesh.from_points(points=fuze_trimesh.vertices,colors=fuze_trimesh.colors)




    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(body_mesh, 'mesh')
    light_nodes = _create_raymond_lights()
    for node in light_nodes:
       scene.add_node(node)
    pyrender.Viewer(scene)
    camera_translation[0] *= -1.0
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation

    camera = pyrender.camera.IntrinsicsCamera(
       fx=camera_focal_length, fy=camera_focal_length, 
       cx=camera_center[0], cy=camera_center[1]
    )
    scene.add(camera, pose=camera_pose)
    pyrender.Viewer(scene)
    light_nodes = _create_raymond_lights()
    for node in light_nodes:
       scene.add_node(node)


    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    # scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(
        viewport_width=image_width,
        viewport_height=image_height,
        point_size=1.0
    )
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    plt.imshow(color)
    plt.show()
    return color, depth


def combine_scene_image(scene_rgba: Union[np.ndarray, List[np.ndarray]],
                        original_image) -> pil_img:
    """Combines the rendered scene with the original image and returns it as PIL.Image
    If a list of rendered scenes is given, they are all added to the original image.
    TODO: Strategy when bodies overlap (e.g. consider depth maps to infer which body should be shown)
    """
    # rewrite such that only lists are accepted and processed
    original_normalized = (np.asarray(original_image) / 255.0).astype(np.float32)
    if isinstance(scene_rgba, np.ndarray):
        scene_normalized = scene_rgba.astype(np.float32) / 255.0
        valid_mask = (scene_normalized[:, :, -1] > 0)[:, :, np.newaxis]

        output_image = (scene_normalized[:, :, :-1] * valid_mask +
                        (1 - valid_mask) * original_normalized)
        img = pil_img.fromarray((output_image * 255).astype(np.uint8))
        return img
    elif isinstance(scene_rgba, list):
        bodies = []
        valid_masks = []
        scenes_normalized = np.asarray(scene_rgba).astype(np.float32) / 255.0
        for scene in scenes_normalized:
            valid_mask = (scene[:, :, -1] > 0)[:, :, np.newaxis]
            bodies.append(scene[:, :, :-1] * valid_mask)
            valid_masks.append(valid_mask)
        body_mask = (np.sum(valid_masks, axis=0)[:, :, 0] > 0)[:, :, np.newaxis]
        # for now: prevent overflow when bodies overlap by clipping. Later: only show the one in front using depth map
        output_image = np.clip(np.sum(bodies, axis=0), 0, 1) + (1 - body_mask) * original_normalized
        img = pil_img.fromarray((output_image * 255).astype(np.uint8))
        return img        


def load_image(path) -> pil_img:
    """Loads the image from the given path and returns it"""
    img = pil_img.open(path)
    return img


def main(args):
    
    mesh_folder = os.path.join(args.data, 'meshes')
    result_pickle_folder = os.path.join(args.data, 'results')

    images = [file for file in os.listdir(args.images) if os.path.splitext(file)[1] in ['.png', '.jpg']]

    os.makedirs(args.output, exist_ok=True)

    # visualize each image separately
    for image in tqdm(images, desc="Image Processing"):
        # try:
        image_name = os.path.splitext(image)[0]

        if not os.path.exists(os.path.join(mesh_folder, image_name)):
            if args.verbosity > 0:
                print(f"No mesh generated for image {image}")
            if args.copy_empty:
                shutil.copy(os.path.join(args.images, image), os.path.join(args.output, image))
            continue

        persons_meshes = os.listdir(os.path.join(mesh_folder, image_name))
        persons_results = os.listdir(os.path.join(result_pickle_folder, image_name))
        img = load_image(os.path.join(args.images, image))

        assert len(persons_meshes) == len(persons_results), f"Not the same amount of persons in meshes and results folder for image {image}"

        # at the moment, only a single person is supported
        #person = persons_meshes[0]
        renders_to_combine = []
        for person in persons_meshes:

            person_id = os.path.splitext(person)[0]
            result = read_pickle_file(os.path.join(result_pickle_folder, image_name, person_id+'.pkl'))
            print(result)
            body_mesh = load_mesh(os.path.join(mesh_folder, image_name, person))

            scene_rgba, _ = get_scene_render(
                body_mesh,
                img.size[0],
                img.size[1],
                # result['camera_center'],
                # [img.size[1]/2,img.size[0]/2],#1920,1080
                [img.size[0]/2,img.size[1]/2],
                result['camera_translation'].squeeze()
            )
            renders_to_combine.append(scene_rgba)
            overlayed = combine_scene_image(scene_rgba, img)

            if args.show_results:
                #overlayed.show()
                plt.imshow(overlayed)
                plt.gcf().canvas.manager.set_window_title(f"{image_name} - {person_id}")
                plt.axis('off')
                plt.show()

            if not args.no_save:
                overlayed.save(os.path.join(args.output, f"{image_name}_{person_id}.png"))

        if len(renders_to_combine) > 1:
            overall_overlayed = combine_scene_image(renders_to_combine, img)
            if args.show_results:
                    #overlayed.show()
                    plt.imshow(overall_overlayed)
                    plt.gcf().canvas.manager.set_window_title(f"{image_name} - all")
                    plt.axis('off')
                    plt.show()

            if not args.no_save:
                overall_overlayed.save(os.path.join(args.output, f"{image_name}_all.png"))
        
        # except Exception as e:
        #     tqdm.write(f'failed at {image_name}')
        #     pass
        # exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizes the fitted SMPL meshes on the original images.")
    parser.add_argument("-d", "--data", type=str, default="colorflip-output", help="Path to the SMPLify-X output folder that contains the meshes and pickle files.")
    parser.add_argument("-i", "--images", type=str, default="colorflip/images/",help="Path to the folder that contains the input images.")
    parser.add_argument("-o", "--output", type=str, default="align-output",help="Location where the resulting images should be saved at.")
    parser.add_argument("--focal_length", type=float, default=5000, help="Focal length of the camera.")
    parser.add_argument("--copy_empty", action="store_true", help="Copies input images without persons to the output folder.")
    parser.add_argument('--show_results', action="store_true", help="Show the resulting overlayed images.")
    parser.add_argument('--no_save', action="store_true", help="Do not save the resulting overlayed images.")
    parser.add_argument('-v', '--verbosity', type=int, default=0, help="Verbosity level.")
    args = parser.parse_args()

    main(args)