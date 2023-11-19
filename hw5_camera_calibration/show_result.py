import numpy as np
import matplotlib.pyplot as plt
import click_utils as click
import cv2 
import os

def get_transformation_matrix():
    image_path = './calibration/camera/1518069838279552217.jpg'
    lidar_path = './calibration/lidar/1518069838220356412.npy'

    image = cv2.imread(image_path)
    lidar = np.load(lidar_path)

    height, width, _ = image.shape

    focal_length = height / 2
    intrinsic = np.array([[focal_length, 0, height / (2*698.939)],
                          [0, focal_length, width / height],
                          [0, 0, 1]])
    
    dist = np.zeros(5)

    uv_coordinates = click.click_points_2D(image)
    world_coordinates = click.click_points_3D(lidar)

    _, rvec, tvec, _ = cv2.solvePnPRansac(world_coordinates, uv_coordinates, intrinsic, dist)

    rvec, tvec = cv2.solvePnPRefineLM(world_coordinates, uv_coordinates, intrinsic, dist, rvec, tvec)

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = tvec.flatten()

    print("Transformation Matrix:")
    print(transformation_matrix)

    print(transformation_matrix.shape)
    print(intrinsic.shape)

    return intrinsic, transformation_matrix


def project2qrcode(intrinsic, transformation_matrix):
    lidar_folder = './qrcode/lidar/'
    camera_folder = './qrcode/camera/'
    lidar_files = os.listdir(lidar_folder)
    camera_files = os.listdir(camera_folder)
    
    for i in range(len(lidar_files)):
        img = cv2.imread(camera_folder + camera_files[i])
        pointcloud = np.load(lidar_folder + lidar_files[i])

        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_subplot()

        uv = []

        for j in range(len(pointcloud)):
            pw = np.concatenate([pointcloud[j, :3], [0]])
            temp = intrinsic @ transformation_matrix[:3, :] @ pw
            uv.append(temp)

        uv = np.array(uv).T

        ax.imshow(img)
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        scatter = ax.scatter(uv[0, :], uv[1, :], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        ax.set_axis_off()
        
        plt.draw()
        plt.pause(0.1)

        scatter.remove()

def main():
    intrinsic, transformation_matrix = get_transformation_matrix()
    
    project2qrcode(intrinsic, transformation_matrix)

if __name__ == '__main__':
    main()
