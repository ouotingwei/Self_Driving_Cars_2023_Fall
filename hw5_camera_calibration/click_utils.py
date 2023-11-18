import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def click_points_2D(image):
  
  # Setup matplotlib GUI
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Select 2D Image Points')
  ax.set_axis_off()
  ax.imshow(image)

  # Pick points
  picked, corners = [], []
  def onclick(event):

    # Only clicks inside this axis are valid
    if fig.canvas.cursor().shape() != 0: 
      return

    x = event.xdata
    y = event.ydata
    if (x is None) or (y is None):
      return

    # Display the picked point
    picked.append((x, y))
    corners.append((x, y))
    print(str(picked[-1]))

    if len(picked) > 1:

      # Draw the line
      temp = np.array(picked)
      ax.plot(temp[:, 0], temp[:, 1])
      ax.figure.canvas.draw_idle()

      # Reset list for future pick events
      del picked[0]

  # Display GUI
  fig.canvas.mpl_connect('button_press_event', onclick)
  plt.show()
  
  if len(corners) > 1: del corners[-1]
  print()
  return np.array(corners)


def click_points_3D(points):

  # Select points within a specific range
  x_min, x_max = -2.0, 0
  y_min, y_max =  0.0, 2
  z_min, z_max = -1.0, 1
  inrange = np.where((points[:, 0] > x_min) & (points[:, 0] < x_max) &
                     (points[:, 1] > y_min) & (points[:, 1] < y_max) &
                     (points[:, 2] > z_min) & (points[:, 2] < z_max))
  points = points[inrange[0]]

  # Color map for the points
  cmap = matplotlib.colormaps['hsv']
  colors = cmap(points[:, -1] / np.max(points[:, -1]))

  # Setup matplotlib GUI
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title('Select 3D LiDAR Points')
  ax.set_axis_off()
  ax.set_facecolor((0, 0, 0))
  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20, picker=5)

  # Equalize display aspect ratio for all axes
  max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
  mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
  mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
  mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  # Pick points
  picked, corners = [], []
  def onpick(event):
    ind = event.ind[0]
    x, y, z = event.artist._offsets3d

    # Ignore if same point selected again
    if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
      return
    
    # Only clicks inside this axis are valid
    if fig.canvas.cursor().shape() != 0: 
      return
    
    # Display picked point
    picked.append((x[ind], y[ind], z[ind]))
    corners.append((x[ind], y[ind], z[ind]))
    print(str(picked[-1]))
    
    if len(picked) > 1:

      # Draw the line
      temp = np.array(picked)
      ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
      ax.figure.canvas.draw_idle()

      # Reset list for future pick events
      del picked[0]

  # Display GUI
  fig.canvas.mpl_connect('pick_event', onpick)
  plt.show()

  if len(corners) > 1: del corners[-1]
  print()
  return np.array(corners)

