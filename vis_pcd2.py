import numpy as np

import pyrender
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
from tqdm import tqdm




file_i = f'out0001-pcd.ply'
pcd = o3d.io.read_point_cloud(file_i)
pcd_center = np.mean(np.array(pcd.points), axis=0)

pcd_rotate = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
pcd_translate = np.array([[0, 0, 0]]).T
# camera_translate=np.array([[0,0,1000]]).T
pcd_pose = np.hstack([pcd_rotate, pcd_translate])
pcd_pose = np.vstack([pcd_pose, [0, 0, 0, 1]])
if __name__ == '__main__':
    file_i=f'out0001-pcd.ply'
    pcd = o3d.io.read_point_cloud(file_i)
    # R = pcd.get_rotation_matrix_from_xyz((np.pi*1.0, 0, 0))
    # # pcd_center=np.mean(np.array(pcd.points),axis=0)
    # pcd.rotate(R, center=pcd_center)
    print(np.array(pcd.points).shape)
    print(np.array(pcd.colors).shape)
    mesh = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors), poses=pcd_pose)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 100.0])
    scene.add(mesh)
    pyrender.Viewer(scene)
    camera_rotate=pcd.get_rotation_matrix_from_xyz((0,0, 0))
    camera_translate=np.array([[0,0,9300]]).T
    # camera_translate=np.array([[0,0,1000]]).T
    camera_pose=np.hstack([camera_rotate,camera_translate])
    camera_pose=np.vstack([camera_pose,[0,0,0,1]])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=None,zfar=10000000)
    print(camera.get_projection_matrix(width=1920,height=1080))
    camera = pyrender.camera.IntrinsicsCamera(
       fx=1920/2, fy =1920/2,
       cx=1920/2, cy=1080/2,zfar=1000000000000000000000000000000000
    )
    camera_pose = np.eye(4)
    # camera_pose = RT
    camera_pose[1, :] = - camera_pose[1, :]
    camera_pose[2, :] = - camera_pose[2, :]
    # camera_pose[2,3]=9300
    camera = pyrender.camera.IntrinsicsCamera(
       fx=1.08137000e+03, fy=1.08137000e+03,
       cx=9.59500000e+02, cy=5.39500000e+02,zfar=10e20
    )

    scene.add(camera, pose=camera_pose)

    pyrender.Viewer(scene)
    # 480,
    r = pyrender.OffscreenRenderer(1920, 1080)
    color, depth = r.render(scene)


    im = Image.fromarray(color)
    plt.imshow(im)
    plt.show()
    im.save(f"test-PCD.png")
    # test

