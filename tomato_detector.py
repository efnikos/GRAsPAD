import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque  # optional for fixed-size history
import threading
import time


class TomatoDetector:
    def __init__(self, model_path='models/keypoints_new.pt', show_frame=True, history_size=50):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.last_frame= None 

        # Load YOLOv8 Pose model
        self.model = YOLO(model_path)
        self.show_frame = show_frame

        # Define colors for keypoints
        self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # Variable to store the last keypoint set (each keypoint is a dict with 3D coordinates and confidence)
        self.last_keypoint_set = None

        # Optional: store a history of keypoint sets with a maximum length to avoid unbounded memory usage.
        self.keypoints_history = deque(maxlen=history_size)
        # Start the detection loop in a new thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def deproject_to_3d(self, x, y, depth_frame, intrinsics):
        frame_width = depth_frame.width
        frame_height = depth_frame.height
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            return None
        depth = depth_frame.get_distance(x, y)
        if depth > 0:
            return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return None

    def get_average_depth(self, x1, x2, y1, y2, depth_frame):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        region_size = 5
        depths = []
        for dx in range(-region_size // 2, region_size // 2 + 1):
            for dy in range(-region_size // 2, region_size // 2 + 1):
                depth = depth_frame.get_distance(center_x + dx, center_y + dy)
                if depth > 0:
                    depths.append(depth)
        if depths:
            return sum(depths) / len(depths)
        return 0

    def calculate_object_size(self, x1, x2, y1, y2, depth_frame, intrinsics):
        depth = self.get_average_depth(x1, x2, y1, y2, depth_frame)
        if depth <= 0:
            return None
        fx = intrinsics.fx
        pixel_length = abs(x2 - x1)
        object_size = (pixel_length * depth) / fx
        return object_size

    def calculate_reference_frame(self, p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        z_axis = p1 - p2
        z_axis /= np.linalg.norm(z_axis)
        zz_t = np.outer(z_axis, z_axis)
        I = np.eye(3)
        nul = I - zz_t
        x_axis = nul @ (p1 - p3)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return {
            "origin": p1.tolist(),
            "x_axis": x_axis.tolist(),
            "y_axis": y_axis.tolist(),
            "z_axis": z_axis.tolist()
        }

    def draw_reference_frame(self, image, origin, x_axis, y_axis, z_axis, intrinsics):
        def project_to_2d(point):
            return rs.rs2_project_point_to_pixel(intrinsics, point)

        origin_2d = project_to_2d(origin)
        x_2d = project_to_2d((np.array(origin) + np.array(x_axis) * 0.1).tolist())
        y_2d = project_to_2d((np.array(origin) + np.array(y_axis) * 0.1).tolist())
        z_2d = project_to_2d((np.array(origin) + np.array(z_axis) * 0.1).tolist())
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, x_2d)), (0, 0, 255), 2)
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, y_2d)), (0, 255, 0), 2)
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, z_2d)), (255, 0, 0), 2)

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        results = self.model.predict(source=color_image, conf=0.5, save=False, save_txt=False, show=False,
                                     verbose=False)
        detections = results[0]
        # Dictionary to store keypoints with their original YOLO index
        current_keypoints = {idx: {"idx": idx, "point": [0, 0, 0], "confidence": 0.000000000001} for idx in
                          range(3)}  # Default missing keypoints
        # Temporary list to hold keypoints for the current frame
        #current_keypoints = []

        if detections.keypoints is not None:
            for det, box in zip(detections.keypoints.data, detections.boxes.xyxy):
                keypoints = det.cpu().numpy()
                points_3d = []
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(color_image, "tomato", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                object_size = self.calculate_object_size(x1, x2, y1, y2, depth_frame, depth_intrinsics)
                if object_size is not None:
                    object_radius = object_size / 2
                    # print(f"Bounding box length: {object_radius:.2f} m")
                    cv2.putText(color_image, f"L:{object_radius:.2f}", (x1, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                for idx, kp in enumerate(keypoints):
                    kp_x, kp_y, kp_conf = int(kp[0]), int(kp[1]), kp[2]
                    color = self.keypoint_colors[idx % len(self.keypoint_colors)]
                    if kp_conf > 0.2:
                        point_3d = self.deproject_to_3d(kp_x, kp_y, depth_frame, depth_intrinsics)
                        if point_3d:
                            points_3d.append(point_3d)
                            # Save each keypoint as a dict with its 3D coordinates and confidence
                            current_keypoints[idx] = {
                                "idx": idx,
                                "point": point_3d,
                                "confidence": kp_conf
                            }
                            cv2.circle(color_image, (kp_x, kp_y), 5, color, -1)
                            cv2.putText(color_image, f"conf: {kp_conf:.2f}", (kp_x, kp_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # print(f"Keypoint Dict: {current_keypoints}")

                if len(points_3d) == 3:
                    surface_normal = np.array(points_3d[0]) / np.linalg.norm(points_3d[0])
                    adjusted_point_0 = np.array(points_3d[0]) + surface_normal * object_radius
                    reference_frame = self.calculate_reference_frame(adjusted_point_0, points_3d[1], points_3d[2])
                    self.draw_reference_frame(
                        color_image,
                        reference_frame["origin"],
                        reference_frame["x_axis"],
                        reference_frame["y_axis"],
                        reference_frame["z_axis"],
                        depth_intrinsics
                    )

        # Update the last keypoint set and optionally store it in history
        self.last_keypoint_set = current_keypoints
        self.keypoints_history.append(current_keypoints)
        return color_image

    def get_last_keypoint_set(self):
        """Returns the most recent set of 3D keypoints with their confidence."""
        return self.last_keypoint_set

    def get_last_frame(self):
        return self.last_frame

    def run(self):
        try:
            while True:
                image = self.process_frame()
                self.last_frame = image
                if image is None:
                    continue
                if self.show_frame:
                    cv2.namedWindow('RealSense Tomato Detection', cv2.WINDOW_NORMAL)
                    resized_image = cv2.resize(image, (1280, 960))
                    cv2.imshow('RealSense Tomato Detection', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = TomatoDetector(model_path='models/keypoints_new.pt', show_frame=True)
    #detector.run()
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


    # Parse keypoints
    # keypoints = detector.get_last_keypoint_set()  # Retrieve the latest keypoint set
    #
    # if keypoints:
    #     for idx, kp in enumerate(keypoints):
    #         point = kp["point"]         # This gives you the 3D coordinates, e.g., [x, y, z]
    #         confidence = kp["confidence"] # This gives you the confidence value
    #         print(f"Keypoint {idx}: Coordinates = {point}, Confidence = {confidence}")
    # else:
    #     print("No keypoints detected in the current frame.")