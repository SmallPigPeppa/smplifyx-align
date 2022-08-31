import os
meshes_dir="meshes_cam"
f=os.listdir(meshes_dir)
for file in f:
    # print(file)
    idx=file.split('.')[0]
    suffix=file.split('.')[-1]
    new_name=f"{int(idx):06d}.{suffix}"
    os.rename(f"{meshes_dir}/{file}",f"{meshes_dir}/{new_name}")
    # print(idx,suffix,new_name)
    # break
