import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
from tqdm import tqdm

file_i = f'frame0.ply'
pcd = o3d.io.read_point_cloud(file_i)
pcd_center = np.mean(np.array(pcd.points), axis=0)
if __name__ == '__main__':
    # for i in tqdm(range(1,237)):
    # fuze_trimesh=trimesh.load(f'meshes/out{i:0>4d}/000.obj')
    fuze_trimesh = trimesh.load(f'colorflip-output/meshes/out0001/000.obj')
    R = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
    T = np.array([[0, 0, 0]]).T
    mesh_pose=np.hstack([R,T])
    mesh_pose=np.vstack([mesh_pose,[0,0,0,1]])
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',  # 'BLEND',
        baseColorFactor=(66 / 255.0, 55 / 255.0, 62 / 255.0, 1.0),  # (0.67, 0.67, 0.67, 0.5),
        wireframe=True)


    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh,poses=mesh_pose,material=material)
    # , smooth = False
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(mesh)
    camera_rotate = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
    camera_translate = np.array([[0, 0.5, 1]]).T


    camera_pose=np.hstack([camera_rotate,camera_translate])
    camera_pose=np.vstack([camera_pose,[0,0,0,1]])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=None)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)


    pyrender.Viewer(scene)
    # r = pyrender.OffscreenRenderer(800, 800)
    W, H = 1920,1080
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=1.0)
    color, depth = r.render(scene,flags = pyrender.RenderFlags.RGBA)
    # , flags = pyrender.RenderFlags.RGBA
    # pyrender.Viewer(scene)
    plt.imshow(color)
    plt.show()

    # im = Image.fromarray(color)
    # im.save(f"test.png")

    # break
    # im = Image.fromarray(color)
    # im.save(f"mesh-front/frame{i:0>4d}.png")
