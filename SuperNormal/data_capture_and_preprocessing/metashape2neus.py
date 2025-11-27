import os.path
import xml
from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
import numpy as np
import matplotlib.pyplot as plt
# details of camera normalization can be found in Sec. C.3 in https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cao_Multi-View_Azimuth_Stereo_CVPR_2023_supplemental.pdf

def make4x4(P):
    assert P.shape[-1] == 4 or P.shape[-1] == 3
    assert len(P.shape) == 2
    assert P.shape[0] == 3 or P.shape[0] == 4
    ret = np.eye(4)
    ret[:P.shape[0], :P.shape[1]] = P
    return ret


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_sphere(ax, center, radius=1.0, color='orange', alpha=0.3):
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

def plot_camera(ax, pose, scale=0.05):
    center = pose[:3, 3]
    R = pose[:3, :3]
    x_axis = center + R[:, 0] * scale
    y_axis = center + R[:, 1] * scale
    z_axis = center + R[:, 2] * scale

    ax.plot([center[0], x_axis[0]], [center[1], x_axis[1]], [center[2], x_axis[2]], c='r')
    ax.plot([center[0], y_axis[0]], [center[1], y_axis[1]], [center[2], y_axis[2]], c='g')
    ax.plot([center[0], z_axis[0]], [center[1], z_axis[1]], [center[2], z_axis[2]], c='b')
    ax.scatter(*center, c='k', s=10)




def normalize_camera(R_list, t_list, camera2object_ratio):
    A_camera_normalize = 0
    b_camera_normalize = 0
    camera_center_list = []
    for view_idx in range(len(R_list)):

        R = R_list[view_idx]
        t = t_list[view_idx]
        print(f"Cámara {view_idx}: Posición antes de normalizar: {t}")
    
        camera_center = - R.T @ t  # in world coordinate
        camera_center_list.append(camera_center)
        vi = R[2][:, None]  # the camera's principal axis in the world coordinates
        Vi = vi @ vi.T
        A_camera_normalize += np.eye(3) - Vi
        b_camera_normalize += camera_center.T @ (np.eye(3) - Vi)
        print(f"Cámara {view_idx}: Posición tras normalizar: {t}")
    offset = np.linalg.lstsq(A_camera_normalize, np.squeeze(b_camera_normalize), rcond=None)[0]
    camera_center_dist_list = [np.sqrt(np.sum((np.squeeze(c) - offset) ** 2)) for c in camera_center_list]
    scale = np.max(camera_center_dist_list) / camera2object_ratio
    return offset, scale



class MetashapePoseLoader:
    def __init__(self, xml_path, camera2object_ratio):
        with open(xml_path, "r") as f:
            xml_data = f.read()
        bs_data = BeautifulSoup(xml_data, "xml")

        c_unique = bs_data.find_all('resolution')
        img_width = int(c_unique[0].get("width"))
        img_height = int(c_unique[0].get("height"))

        c_intrinsics = bs_data.find_all('calibration')
        f = float(c_intrinsics[0].find("f").text)
        cx_offset = float(c_intrinsics[0].find("cx").text)
        cy_offset = float(c_intrinsics[0].find("cy").text)

        K = np.array([
            [f, 0, (img_width - 1) / 2 + cx_offset],
            [0, f, (img_height - 1) / 2 + cy_offset],
            [0, 0, 1]
        ])

        b_unique = bs_data.find_all('camera')
        R_list = []
        t_list = []
        C2W_list = []
        camera_sphere = dict()

        for view_idx, tag in enumerate(b_unique):

            C2W = np.array([float(i) for i in tag.find("transform").text.split()]).reshape((4, 4))
            C2W_list.append(C2W)

            W2C = np.linalg.inv(C2W)
            R_list.append(W2C[:3, :3])
            t_list.append(W2C[:3, 3])

            camera_sphere[f"world_mat_{view_idx}"] = make4x4(K) @ W2C

        offset, scale = normalize_camera(R_list, t_list, camera2object_ratio=camera2object_ratio)
        print("offset", offset, "scale", scale)
        num_views = len(C2W_list)

        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= scale
        scale_mat[:3, 3] = offset
        # scale_mat = np.eye(4)
        # scale_mat[:3, :3] *= scale
        # scale_mat[:3, 3] = -scale * offset

        for im_idx in range(num_views):
            camera_sphere[f"scale_mat_{im_idx}"] = scale_mat



        data_dir = os.path.dirname(xml_path)
        np.savez(os.path.join(data_dir, 'cameras_sphere.npz'), **camera_sphere)


        # Visualización de cámaras
        data_path = f'./data/myobjects/cherries/cherry_{cherry_idx}/cameras_sphere.npz'
        data = np.load(data_path)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        view_indices = [int(k.split('_')[-1]) for k in data.keys() if k.startswith("world_mat_")]
        view_indices = sorted(set(view_indices))

        for i in view_indices:
            world_mat = data[f"world_mat_{i}"]
            scale_mat = data[f"scale_mat_{i}"]
            full_mat = np.linalg.inv(scale_mat) @ world_mat
            
            pose = np.linalg.inv(full_mat)  # W2C → C2W
            plot_camera(ax, pose)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Cámaras normalizadas")
        ax.set_box_aspect([1,1,1])
        plot_sphere(ax, center=offset, radius=2.0, color='orange', alpha=0.2)
        # Al final de la visualización:
        set_axes_equal(ax)
        # plt.show()
import argparse
    
for cherry_idx in range(62,64):
    # ratio = float(input("ratio parameter : "))
    ratio = 1.5
    # ratio = 2.0


    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=ratio)

    args = parser.parse_args()


    # cherry_idx = int(input("cherry_idx: "))



    MetashapePoseLoader(
        f'./data/myobjects/cherries/cherry_{cherry_idx}/cameras.xml',
        camera2object_ratio=args.ratio,
    )


    import os
    import cv2
    import pyexr
    from glob import glob
    import numpy as np
    import shutil
    from bs4 import BeautifulSoup  # $ pip install beautifulsoup4 lxml
    import argparse

    # tipe = input("type of normal map (sdmunips or ps): ")
    tipe = "ps"

    parser.add_argument("--sdm_unips_result_dir", type=str, default=f"./data/myobjects/cherries/cherry_{cherry_idx}/normal_camera_space_{tipe}")
    parser.add_argument("--data_dir", type=str, default=f"./data/myobjects/cherries/cherry_{cherry_idx}")
    # parser.add_argument("--data_dir", type=str, default="./data/myobjects/cherries/cherry_2")
    args = parser.parse_args()

    xml_path = os.path.join(args.data_dir, "cameras.xml")
    obj_name = os.path.basename(args.data_dir)
    num_views = 20

    normal_map_camera_dir = os.path.join(args.data_dir, f"normal_camera_space_{tipe}")
    normal_map_world_dir = os.path.join(args.data_dir, f"normal_world_space_{tipe}")

    os.makedirs(normal_map_world_dir, exist_ok=True)

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
    
    for i in range(0,num_views): 
        # view_dir = os.path.join(args.sdm_unips_result_dir, f"view_{i:06d}.data")
        for tag in b_unique:
            name = int(tag.get("label"))
            # print(name)
            name = f'{name:06d}'
            if name == f"{i:06d}":
                C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))
                R = C2W[:3, :3]
                break
        # copy normal map
        normal_map_file = os.path.join(normal_map_camera_dir, f"{(i):06d}.exr")
        new_normal_map_file = os.path.join(normal_map_world_dir, f"{(i):06d}.exr")
        shutil.copy(normal_map_file, new_normal_map_file)
        # convert normal map to world space
        normal_map_camera = pyexr.read(new_normal_map_file)
        
        normal_map_camera[..., [1, 2]] *= -1  # revert y and z axis to match opencv conversion, X right, Y down, Z front
        # normal_map_camera[..., [1, 2]] =  normal_map_camera[..., [2, 1]]# revert y and z axis to match opencv conversion, X right, Y down, Z front

        H, W = normal_map_camera.shape[:2]
        normal_world = (R @ normal_map_camera.reshape(-1, 3).T).T.reshape([H, W, 3])
        pyexr.write(os.path.join(normal_map_world_dir, f"{i:06d}.exr"), normal_world)






