import numpy as np

import pyrender
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
from tqdm import tqdm

file_i = f'out0001-pcd.ply'
pcd = o3d.io.read_point_cloud(file_i)
pcd_center = np.mean(np.array(pcd.points), axis=0)
if __name__ == '__main__':
    # for i in tqdm(range(1,237)):
    file_i=f'out0001-pcd.ply'
    pcd = o3d.io.read_point_cloud(file_i)
    R = pcd.get_rotation_matrix_from_xyz((np.pi*1.0, 0, 0))
    R = pcd.get_rotation_matrix_from_xyz((np.pi*1.0, 0, 0))
    # pcd_center=np.mean(np.array(pcd.points),axis=0)
    pcd.rotate(R, center=pcd_center)
    mesh = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors))
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(mesh)
    camera_rotate=pcd.get_rotation_matrix_from_xyz((0,0, 0))
    camera_translate=np.array([[0,0,9300]]).T
    camera_pose=np.hstack([camera_rotate,camera_translate])
    camera_pose=np.vstack([camera_pose,[0,0,0,1]])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=None)
    scene.add(camera, pose=camera_pose)

    pyrender.Viewer(scene)

    pyrender.Viewer(scene)
    r = pyrender.OffscreenRenderer(800, 800)
    color, depth = r.render(scene)

    plt.imshow(color)
    plt.show()

    # im = Image.fromarray(color)
    # im.save(f"front-view/frame{i:0>4d}.png")
