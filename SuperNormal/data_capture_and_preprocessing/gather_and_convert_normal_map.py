# import os
# import cv2
# import pyexr
# from glob import glob
# import numpy as np
# import shutil
# from bs4 import BeautifulSoup  # $ pip install beautifulsoup4 lxml
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--sdm_unips_result_dir", type=str, default="../data/diligent_mv_normals/buddha/normal_camera_space_sdmunips")
# parser.add_argument("--data_dir", type=str, default="../data/diligent_mv_normals/buddha")
# args = parser.parse_args()

# xml_path = os.path.join(args.data_dir, "cameras.xml")
# obj_name = os.path.basename(args.data_dir)
# print(obj_name)
# print(xml_path)
# num_views = 20

# normal_map_camera_dir = os.path.join(args.data_dir, "normal_camera_space_sdmunips")
# normal_map_world_dir = os.path.join(args.data_dir, "normal_world_space_sdmunips")

# # create directories
# os.makedirs(normal_map_camera_dir, exist_ok=True)
# os.makedirs(normal_map_world_dir, exist_ok=True)

# with open(xml_path, "r") as f:
#     xml_data = f.read()
# bs_data = BeautifulSoup(xml_data, "xml")
# print(bs_data)
# b_unique = bs_data.find_all('camera')

# for tag in b_unique:
#     img_name = tag.get("label")
#     print(img_name)
#     view_idx = img_name
#     # camera to world transform
#     C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))

#     print(C2W)


# normal_map_all = []
# normal_map_path_all = []

# for i in range(num_views):
#     # view_dir = os.path.join(args.sdm_unips_result_dir, f"view_{i:05d}.data")
#     for tag in b_unique:
#         name = int(tag.get("label"))    
#         print(name)
#         name = f'{name:05d}'
#         if name == f"{i:05d}":
#             C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))
#             R = C2W[:3, :3]
#             break

#         # print(f'R: {R}')
#     # print(f'view_idx: {view_dir}')
#     # if os.path.exists(view_dir):
#         # copy normal map
        
#     normal_map_file = os.path.join(normal_map_camera_dir, f"{i:05d}.exr")
#     new_normal_map_file = os.path.join(normal_map_world_dir, f"{i:05d}.exr")
#     print(new_normal_map_file)
#     shutil.copy(normal_map_file, new_normal_map_file)

#     # convert normal map to world space
#     normal_map_camera = pyexr.read(new_normal_map_file)
#     # normal_map_camera[..., [1, 2]] *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
    
#     # normal_map_camera *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
#     normal_map_camera[..., [1, 2]] *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
#     # normal_map_camera *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front


#     H, W = normal_map_camera.shape[:2]
#     normal_world = (R @ normal_map_camera.reshape(-1, 3).T).T.reshape([H, W, 3])

    
#     # normal_world[..., [1, 2]] *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
#     # normal_world[..., [1, 2]] = normal_world[..., [2, 1]] 
#     normal_world[..., [0, 2]] = normal_world[..., [2, 0]] 
#     # normal_world[..., [1, 0]] = normal_world[..., [0, 1]] 

#     # normal_world[..., 2] *= -1  # Invierte el eje Y
#     # normal_world[..., [1, 2]] = normal_world[..., [2, 1]] # y z 


#     pyexr.write(os.path.join(normal_map_world_dir, f"{i:05d}.exr"), normal_world)








import os
import cv2
import pyexr
from glob import glob
import numpy as np
import shutil
from bs4 import BeautifulSoup  # $ pip install beautifulsoup4 lxml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sdm_unips_result_dir", type=str, default="../data/tomato/tom_cherry/normal_camera_space_sdmunips")
parser.add_argument("--data_dir", type=str, default="../data/tomato/tom_cherry")
args = parser.parse_args()

xml_path = os.path.join(args.data_dir, "cameras.xml")
obj_name = os.path.basename(args.data_dir)
num_views =  40

normal_map_camera_dir = os.path.join(args.data_dir, "normal_camera_space_sdmunips")
normal_map_world_dir = os.path.join(args.data_dir, "normal_world_space_sdmunips")

# create directories
os.makedirs(normal_map_camera_dir, exist_ok=True)
os.makedirs(normal_map_world_dir, exist_ok=True)

with open(xml_path, "r") as f:
    xml_data = f.read()
bs_data = BeautifulSoup(xml_data, "xml")
b_unique = bs_data.find_all('camera')

for tag in b_unique:
    img_name = tag.get("label")
    view_idx = int(img_name.split("_")[-1])
    # camera to world transform
    C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))


normal_map_all = []
normal_map_path_all = []

for i in range(num_views): 
    # view_dir = os.path.join(args.sdm_unips_result_dir, f"view_{i:05d}.data")
    for tag in b_unique:
        name = int(tag.get("label")) 
        print(name)
        name = f'{name:05d}'
        if name == f"{i:05d}":
            C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))
            R = C2W[:3, :3]
            break
    # copy normal map
    print('i', i)
    normal_map_file = os.path.join(normal_map_camera_dir, f"{i:05d}.exr")
    new_normal_map_file = os.path.join(normal_map_world_dir, f"{i:05d}.exr")
    shutil.copy(normal_map_file, new_normal_map_file)
    # convert normal map to world space
    normal_map_camera = pyexr.read(new_normal_map_file)
    normal_map_camera[..., [1, 2]] *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
    # normal_map_camera[..., [1, 2]] =  normal_map_camera[..., [2, 1]]# revert y and z axis to match opencv conversion, X right, Y down, Z front

    H, W = normal_map_camera.shape[:2]
    normal_world = (R @ normal_map_camera.reshape(-1, 3).T).T.reshape([H, W, 3])
    pyexr.write(os.path.join(normal_map_world_dir, f"{i:05d}.exr"), normal_world)


