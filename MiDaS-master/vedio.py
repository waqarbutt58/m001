import cv2
import numpy as np
import time
import os
# Load input image and depth (grayscale or color)
input_image = cv2.imread("frame_01151.jpg")  # Original RGB image
input_image_plain = cv2.imread("frame_24973.jpg")  # Original RGB image
depth_map = cv2.imread("output_depth_gray.jpg", cv2.IMREAD_GRAYSCALE)  # Grayscale depth
depth_map_plain = cv2.imread("plain_output_depth_gray.jpg", cv2.IMREAD_GRAYSCALE)  # Grayscale depth


# Example position and fake pothole depth
#x, y, w, h = 100, 150, 80, 80  # (x, y) is top-left, w and h are width and height
x, y, w, h = 35, 1020, 550, 105  # (x, y) is top-left, w and h are width and height

pothole_location = (500, 750)
pothole_depth_cm = 12.5  # Example depth in cm

# Annotate the input image
input_annotated = input_image.copy()
input_annotated_plain = input_image_plain.copy()
#cv2.circle(input_annotated, pothole_location, 10, (0, 0, 255), -1)
cv2.rectangle(input_annotated, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
cv2.putText(input_annotated, "Pothole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

roi_depth = depth_map[y:y+h, x:x+w]
average_distance = np.mean(roi_depth)/1000

# Round for display
distance_text = f"{average_distance:.2f} m"

if len(depth_map.shape) == 2:
    depth_annotated = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
else:
    depth_annotated = depth_map.copy()


if len(depth_map_plain.shape) == 2:
    depth_plain_annotated = cv2.cvtColor(depth_map_plain, cv2.COLOR_GRAY2BGR)
else:
    depth_plain_annotated = depth_map_plain.copy()

#cv2.circle(depth_annotated, pothole_location, 10, (0, 0, 255), -1)
#cv2.putText(depth_annotated, f"Estimated Depth", (50, 50),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.rectangle(depth_annotated, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
cv2.putText(depth_annotated, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Resize both to same size if needed
height = 384
width = 672
input_resized = cv2.resize(input_annotated, (width, height))
depth_resized = cv2.resize(depth_annotated, (width, height))


input_resized_plain = cv2.resize(input_image_plain, (width, height))
depth_resized_plain = cv2.resize(depth_plain_annotated, (width, height))

print(input_resized_plain.shape, depth_resized_plain.shape)
print(input_resized.shape, depth_resized.shape)
combined_frame = np.hstack((input_resized, depth_resized))  # Side-by-side
combined_frame_plain = np.hstack((input_resized_plain, depth_resized_plain))  # Side-by-side



# Parameters
output_filename = 'pothole_depth_video.avi'
fps = 10
duration = 15  # total video duration in seconds
frame_count = fps * duration
half_frame_count = frame_count // 2


# Define video codec and writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (combined_frame.shape[1], combined_frame.shape[0]))

# Write first sample (e.g., pothole)
for _ in range(half_frame_count):
    video_writer.write(combined_frame)

# Write second sample (e.g., no pothole)
for _ in range(frame_count - half_frame_count):
    video_writer.write(combined_frame_plain)


video_writer.release()
print(f"Video saved: {output_filename}")
