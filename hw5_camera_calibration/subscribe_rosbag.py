import os
import cv2
import rospy
import numpy as np

from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge

output_root = './NCTU/'
output_root_lidar = output_root + 'lidar/'
output_root_camera = output_root + 'camera/'
global file_name
file_name = 0

if not os.path.exists(output_root_lidar):
    os.makedirs(output_root_lidar)

if not os.path.exists(output_root_camera):
    os.makedirs(output_root_camera)


def read_point_from_msg(msg):
    points_list = []
    for point in pcl2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        points_list.append(point)
    return np.asarray(points_list, dtype=np.float32)


class MySubscriber():
    def __init__(self):
        rospy.Subscriber("/points", PointCloud2, self.lidar_callback, queue_size=None)
        rospy.Subscriber("/left/image/compressed", CompressedImage, self.camera_callback, queue_size=None)

    def lidar_callback(self, msg):
        global file_name
        # Check the header
        # print(msg.header, '\n')

        timestamp = str(msg.header.stamp.secs) + "{:09d}".format(msg.header.stamp.nsecs)
        pointcloud = read_point_from_msg(msg)
        np.save(output_root_lidar + str(file_name) + '.npy', pointcloud)

        # Call camera callback after processing lidar data
        self.camera_callback(msg)

    def camera_callback(self, msg):
        global file_name
        # Check the header
        # print(msg.header, '\n')

        timestamp = str(msg.header.stamp.secs) + "{:09d}".format(msg.header.stamp.nsecs)
        img = CvBridge().compressed_imgmsg_to_cv2(msg)

        # Check if the image is not empty and has a valid size
        if img is not None and img.size > 0:
            # Check the type of the image
            print("Image type:", img.dtype)

            cv2.imwrite(output_root_camera + str(file_name) + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            file_name += 1
        else:
            print("Warning: Empty or invalid image received.")


if __name__ == '__main__':
    rospy.init_node('subscriber_node', anonymous=True)
    MySubscriber()
    rospy.spin()
