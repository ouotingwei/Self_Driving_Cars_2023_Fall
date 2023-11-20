import numpy as np
import matplotlib.pyplot as plt
import click_utils as click
import cv2 
import os

def get_transformation_matrix():
    image_path = './calibration/camera/1518069838279552217.jpg'
    lidar_path = './calibration/lidar/1518069842434335859.npy'

    image = cv2.imread(image_path)
    lidar = np.load(lidar_path)

    height, width, _ = image.shape
    print(height, width)

    focal_length = 698.939
    intrinsic = np.array([[focal_length, 0, width/2],
                          [0, focal_length, height/2],
                          [0, 0, 1]])
    
    dist = np.zeros(5)

    uv_coordinates = click.click_points_2D(image)
    
    world_coordinates = click.click_points_3D(lidar)

    _, rvec, tvec, _ = cv2.solvePnPRansac(world_coordinates, uv_coordinates, intrinsic, dist)

    rvec, tvec = cv2.solvePnPRefineLM(world_coordinates, uv_coordinates, intrinsic, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)

    transformation_matrix = np.column_stack((R, tvec))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    print("Transformation Matrix:")
    print(transformation_matrix)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot()

    uv = []

    projection_model = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0]])

    for j in range(len(lidar)):
        pw = np.concatenate([lidar[j, :3], [1]])
        temp = ((intrinsic @ projection_model) @ transformation_matrix) @ pw.T
        scales = 1/temp[2]
        temp = temp * scales 
        uv.append(temp)

    uv = np.array(uv).T

    ax.imshow(image)
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.scatter(uv[0, :], uv[1, :], c=lidar[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
    ax.set_axis_off()

    plt.show()  # This line will display the plot

    return intrinsic, transformation_matrix


def project2qrcode(intrinsic, transformation_matrix):
    lidar_folder = './qrcode/lidar/'
    camera_folder = './qrcode/camera/'
    lidar_files = os.listdir(lidar_folder)
    camera_files = os.listdir(camera_folder)
    lidar_len = len(lidar_files)
    cam_len = len(camera_files)

    delta = cam_len / lidar_len

    print(delta)

    k = 0
    for i in range(cam_len):
        if i % 3 == 0:
            i += 1
        
        else:
            img = cv2.imread(camera_folder + camera_files[k])
            pointcloud = np.load(lidar_folder + lidar_files[i])

            fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
            ax = fig.add_subplot()

            uv = []

            for j in range(len(pointcloud)):
                pw = np.concatenate([pointcloud[j, :3], [1]])
                temp = intrinsic @ transformation_matrix[:3, :] @ pw
                uv.append(temp)

            uv = np.array(uv).T

            ax.imshow(img)
            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)
            ax.scatter(uv[0, :], uv[1, :], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
            ax.set_axis_off() 

            k += 1
            plt.pause(0.1)  # Pause to allow the plot to update


def main():
    intrinsic, transformation_matrix = get_transformation_matrix()
    transformation_matrix = np.array([[0.85942994, 0.51122949, -0.00495944, 0.12869505],
                                      [0.03469834, -0.0680042, -0.99708147, -0.32801881],
                                      [-0.51007471, 0.85674958, -0.07618366, -0.13943649],
                                      [0, 0, 0, 1]])
    
    focal_length = 698.939
    intrinsic = np.array([[focal_length, 0, 1280/2],
                          [0, focal_length, 720/2],
                          [0, 0, 1]])
    
    #project2qrcode(intrinsic, transformation_matrix)

if __name__ == '__main__':
    main()
