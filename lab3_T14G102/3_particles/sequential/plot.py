from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define cone parameters
height = 2.5e-1
radius = np.tan(np.pi / 6) * height
num_points = 30  # Resolution of the cone

# mesh grid in polar coordinates
theta = np.linspace(0, 2 * np.pi, num_points)
z_cone = np.linspace(0, -height, num_points)
theta, z_cone = np.meshgrid(theta, z_cone)

# Convert to Cartesian coordinates
r = radius * z_cone / height  # Linear increase of radius
x_cone = r * np.cos(theta)
y_cone = r * np.sin(theta)

# Create a VideoWriter object
video_dir = Path("./video")
video_dir.mkdir(exist_ok=True)

video_filename = video_dir.joinpath("video.mp4")
frame_size = (640, 480)
fps = 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(video_filename), fourcc, fps, frame_size)

for i, file in enumerate(sorted(Path(".").glob("out/*.csv"))):
    data = np.loadtxt(file, float, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    fig.patch.set_alpha(0)

    ax.plot_surface(x_cone, y_cone, z_cone, alpha=0.3)

    # Axis
    ax.plot(
        [0, radius * 1.1], [0, 0], [0, 0], "b", linewidth=2, label="X-axis"
    )  # X-axis
    ax.plot(
        [0, 0], [0, radius * 1.1], [0, 0], "g", linewidth=2, label="Y-axis"
    )  # Y-axis
    ax.plot(
        [0, 0], [0, 0], [0, -height / 2], "r", linewidth=2, label="Z-axis"
    )  # Z-axis

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    ax.scatter(x, y, z, c="b", marker="o")
    ax.set_xlim([-radius*1.2, radius*1.2])
    ax.set_ylim([-radius*1.2, radius*1.2])
    ax.set_zlim([-height*1.2, 0])

    image_path = video_dir.joinpath(f"frame_{i}.png")
    plt.savefig(image_path)
    plt.close()

    frame = cv2.imread(image_path)  # Read the frame/image
    frame_resized = cv2.resize(
        frame, frame_size
    )  # Resize to match the video resolution
    out.write(frame_resized)  # Write the frame to the video

out.release()
#cv2.destroyAllWindows()