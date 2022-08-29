import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import open3d as o3d




pcd = o3d.io.read_point_cloud("frame0.ply")

R = pcd.get_rotation_matrix_from_xyz((np.pi * 1.0, np.pi * 0.3, 0))
pcd_center = np.mean(np.array(pcd.points), axis=0)
pcd.rotate(R, center=pcd_center)



camera_rotate = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
camera_translate = np.array([[4500, 500, 13000]]).T
camera_pose = np.hstack([camera_rotate, camera_translate])
camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=None)



# o3d.visualization.draw_geometries([pcd])

mesh = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors))
scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
scene.add(mesh)
# 1080 1920
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=None)
scene.add(camera, pose=camera_pose)
light = pyrender.PointLight(intensity = 500)
light = pyrender.SpotLight(color=np.ones(3), intensity=100.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
pyrender.Viewer(scene)
r = pyrender.OffscreenRenderer(800, 800)
color, depth = r.render(scene)
plt.imshow(color)
plt.show()
from PIL import Image
# im = Image.fromarray(color)
# im.save("result-right.jpeg")
