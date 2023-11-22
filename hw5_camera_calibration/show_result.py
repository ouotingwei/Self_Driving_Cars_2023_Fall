import numpy as np
import matplotlib.pyplot as plt
import click_utils as click
import cv2 
import os
import matplotlib.animation as animation
import time

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load lidar point cloud
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
    # Print processing message
    print("[!] (NCTU_video.mp4) is processing !")

    # Set output folder and input folders for lidar and camera data
    output_folder = './NCTU_output_frames'
    lidar_folder = './NCTU/lidar/'
    camera_folder = './NCTU/camera/'

    # Get sorted list of camera and lidar files
    camera_files = sorted(os.listdir(camera_folder), key=lambda x: int(x.split('.')[0]))
    lidar_files = sorted(os.listdir(lidar_folder), key=lambda x: int(x.split('.')[0]))

    # Get the number of frames in the dataset
    loop_len = len(lidar_files)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each frame
    for frame in range(loop_len):
        # Read the camera image
        img = cv2.imread(camera_folder + camera_files[frame])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load lidar point cloud
        pointcloud = np.load(lidar_folder + lidar_files[frame])

        # Initialize empty list to store UV coordinates
        uv = []

        # Define a projection model matrix
        projection_model = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]])

        # Loop through each point in the point cloud
        for j in range(len(pointcloud)):
            # Append UV coordinates to the list
            pw = np.concatenate([pointcloud[j, :3], [1]])
            temp = intrinsic @ projection_model @ transformation_matrix @ pw
            scales = 1 / temp[2]
            temp = temp * scales
            uv.append(temp)

        # Convert UV coordinates to array and transpose
        uv = np.array(uv).T

        # Define result file path
        result_file = os.path.join(output_folder, f'{frame}.png')

        # Plot the camera image with lidar points overlaid
        plt.figure(figsize=(12.8, 7.2), dpi=100)
        plt.imshow(img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.scatter(uv[0, :], uv[1, :], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        plt.axis('off')

        # Save the result image
        plt.savefig(result_file)

        # Close the plot
        plt.close()


def images_to_video(input_folder, output_video_path, fps=15):
    # Get sorted list of image files in the input folder
    image_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop through each image file and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        video.write(img)

    # Release the VideoWriter and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()


def main():
    intrinsic, transformation_matrix = get_transformation_matrix()
    
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
