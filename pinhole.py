import numpy as np


# [LEFT_CAM_HD]
# fx=533.895
# fy=534.07
# cx=632.69
# cy=379.6325
# k1=-0.0489421
# k2=0.0208547
# p1=0.000261529
# p2=-0.000580449
# k3=-0.00836067
#
# [RIGHT_CAM_HD]
# fx=532.225
# fy=532.47
# cx=645.515
# cy=362.0185
# k1=-0.0463267
# k2=0.0195163
# p1=0.000313832
# p2=-8.13248e-05
# k3=-0.00854262




class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, size_x, size_y):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y

    def project(self, point_3d):
        point = np.zeros(3)
        point[0] = point_3d[0] * self.fx / point_3d[2] + self.cx
        point[1] = point_3d[1] * self.fy / point_3d[2] + self.cy
        point[2] = point_3d[2]
        return point

    def back_project(self, point, depth_value):
        point_3d = np.zeros(3)
        point_3d[2] = depth_value
        point_3d[0] = (point[0] - self.cx) * point_3d[2] / self.fx
        point_3d[1] = (point[1] - self.cy) * point_3d[2] / self.fy
        return point_3d

    def matrix(self):
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = self.fx
        cam_matrix[1, 1] = self.fy
        cam_matrix[0, 2] = self.cx
        cam_matrix[1, 2] = self.cy
        cam_matrix[2, 2] = 1.0
        return cam_matrix
