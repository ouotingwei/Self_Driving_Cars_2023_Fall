import numpy as np
import matplotlib.pyplot as plt
import click_utils as click
import cv2 
import os
import matplotlib.animation as animation
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import click  # Assuming this is a custom module for clicking points in the image

def get_transformation_matrix():
    # Paths to the image and lidar data
    image_path = './calibration/camera/1518069838279552217.jpg'
    lidar_path = './calibration/lidar/1518069842434335859.npy'

    # Read the image and lidar data
    image = cv2.imread(image_path)
    lidar = np.load(lidar_path)

    # Get image dimensions
    height, width, _ = image.shape
    print("Image Dimensions:", height, width)

    # Camera intrinsic parameters
    focal_length = 698.939
    intrinsic = np.array([[focal_length, 0, width/2],
                          [0, focal_length, height/2],
                          [0, 0, 1]])

    # Distortion coefficients
    dist = np.zeros(5)

    # Get 2D and 3D coordinates by clicking points on the image and lidar data
    uv_coordinates = click.click_points_2D(image)
    world_coordinates = click.click_points_3D(lidar)

    # Solve PnP using RANSAC algorithm
    _, rvec, tvec, _ = cv2.solvePnPRansac(world_coordinates, uv_coordinates, intrinsic, dist)

    # Refine the solution using Levenberg-Marquardt optimization
    rvec, tvec = cv2.solvePnPRefineLM(world_coordinates, uv_coordinates, intrinsic, dist, rvec, tvec)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Create the transformation matrix
    transformation_matrix = np.column_stack((R, tvec))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    print("Transformation Matrix:")
    print(transformation_matrix)

    # Visualization of the 3D lidar points projected onto the image
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot()

    uv = []

    # Projection model matrix
    projection_model = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0]])

    # Project lidar points onto the image
    for j in range(len(lidar)):
        pw = np.concatenate([lidar[j, :3], [1]])
        temp = ((intrinsic @ projection_model) @ transformation_matrix) @ pw.T
        scales = 1/temp[2]
        temp = temp * scales 
        uv.append(temp)

    uv = np.array(uv).T

    # Plot the image with lidar points
    ax.imshow(image)
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.scatter(uv[0, :], uv[1, :], c=lidar[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
    ax.set_axis_off()

    plt.show()

    return intrinsic, transformation_matrix


def project2moving(intrinsic, transformation_matrix):
    print("[!] (moving_video.mp4) is processing !")

    # Output folder for saving processed frames
    output_folder = './moving_output_frames'
    lidar_folder = './moving/lidar/'
    camera_folder = './moving/camera/'

    # Get sorted list of camera and lidar files
    camera_files = sorted(os.listdir(camera_folder), key=lambda x: int(x.split('.')[0]))
    lidar_files = sorted(os.listdir(lidar_folder), key=lambda x: int(x.split('.')[0]))
    loop_len = len(lidar_files)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each frame in the video
    for frame in range(loop_len):
        # Read the image and lidar data for the current frame
        img = cv2.imread(camera_folder + camera_files[frame])
        pointcloud = np.load(lidar_folder + lidar_files[frame])

        uv = []

        # Projection model matrix
        projection_model = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]])

        # Project lidar points onto the image
        for j in range(len(pointcloud)):
            pw = np.concatenate([pointcloud[j, :3], [1]])
            temp = intrinsic @ projection_model @ transformation_matrix @ pw
            scales = 1 / temp[2]
            temp = temp * scales
            uv.append(temp)

        uv = np.array(uv).T

        # Save the result image for the current frame
        result_file = os.path.join(output_folder, f'{frame}.png')

        plt.figure(figsize=(12.8, 7.2), dpi=100)
        plt.imshow(img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.scatter(uv[0, :], uv[1, :], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        plt.axis('off')

        plt.savefig(result_file)
        plt.close()


def project2nctu(intrinsic, transformation_matrix):
    print("[!] (NCTU_video.mp4) is processing !")
    output_folder = './NCTU_output_frames'
    lidar_folder = './NCTU/lidar/'
    camera_folder = './NCTU/camera/'
    camera_files = sorted(os.listdir(camera_folder), key=lambda x: int(x.split('.')[0]))
    lidar_files = sorted(os.listdir(lidar_folder), key=lambda x: int(x.split('.')[0]))
    loop_len = len(lidar_files)

    os.makedirs(output_folder, exist_ok=True)

    for frame in range(loop_len):
        img = cv2.imread(camera_folder + camera_files[frame])
        pointcloud = np.load(lidar_folder + lidar_files[frame])

        uv = []

        projection_model = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0]])

        for j in range(len(pointcloud)):
            pw = np.concatenate([pointcloud[j, :3], [1]])
            temp = intrinsic @ projection_model @ transformation_matrix @ pw
            scales = 1 / temp[2]
            temp = temp * scales
            uv.append(temp)

        uv = np.array(uv).T

        result_file = os.path.join(output_folder, f'{frame}.png')

        plt.figure(figsize=(12.8, 7.2), dpi=100)
        plt.imshow(img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.scatter(uv[0, :], uv[1, :], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        plt.axis('off')

        plt.savefig(result_file)
        plt.close()


def images_to_video(input_folder, output_video_path, fps=15):
    image_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))

    # Assuming all images have the same resolution as the first image
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()


def main():
    intrinsic, transformation_matrix = get_transformation_matrix()
    #transformation_matrix = np.array([[0.85942994, 0.51122949, -0.00495944, 0.12869505],
    #                                  [0.03469834, -0.0680042, -0.99708147, -0.32801881],
    #                                  [-0.51007471, 0.85674958, -0.07618366, -0.13943649],
    #                                  [0, 0, 0, 1]])
    
    #focal_length = 698.939
    #intrinsic = np.array([[focal_length, 0, 1280/2],
    #                      [0, focal_length, 720/2],
    #                      [0, 0, 1]])
    
    project2moving(intrinsic, transformation_matrix)
    images_folder = './moving_output_frames'
    output_video_path = './moving_video.mp4'
    images_to_video(images_folder, output_video_path)

    project2nctu(intrinsic, transformation_matrix)
    images_folder = './NCTU_output_frames'
    output_video_path = './NCTU_video.mp4'
    images_to_video(images_folder, output_video_path)

if __name__ == '__main__':
    main()
