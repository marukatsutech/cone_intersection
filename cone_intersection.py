""" Intersection of Cone and Plane (Cone and circle, ellipse, parabola, hyperbola) """
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation
import mpl_toolkits.mplot3d.art3d as art3d


""" Global variables """

""" Animation control """
is_play = False

""" Axis vectors """
vector_x_axis = np.array([1., 0., 0.])
vector_y_axis = np.array([0., 1., 0.])
vector_z_axis = np.array([0., 0., 1.])

""" Cone parameter """
height_cone = 2.
direction_theta = 0.
direction_phi = 0.
angle_cone_deg = 45.
apex_cone = np.array([0., 0., height_cone])
length_slant_line = 8.

""" Create figure and axes """
title_ax0 = "Intersection of Cone and Plane"
title_tk = title_ax0

x_min = -4.
x_max = 4.
y_min = -4.
y_max = 4.
z_min = 0.
z_max = 4.

fig = Figure()
ax0 = fig.add_subplot(111, projection="3d")
ax0.set_box_aspect((8, 8, 4))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

""" Embed in Tkinter """
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

""" Classes and functions """


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


class Arrow:
    def __init__(self, ax, point_start, theta, phi, col, label):
        self.ax = ax
        self.x, self.y, self.z = point_start[0], point_start[1], point_start[2]
        self.theta_init = theta
        self.phi_init = phi
        self.r = 1.
        self.vector = spherical_to_cartesian(self.r, self.theta_init, self.phi_init)
        self.col = col
        self.label = label
        self.quiver = self.ax.quiver(self.x, self.y, self.z, self.vector[0], self.vector[1], self.vector[2],
                                     length=1, color=self.col, normalize=True, label=self.label)
        self.vector_init = np.array([self.vector[0], self.vector[1], self.vector[2]])
        self.is_rotate = True

    def update_quiver(self):
        if hasattr(self, 'quiver'):
            self.quiver.remove()
        self.quiver = self.ax.quiver(self.x, self.y, self.z, self.vector[0], self.vector[1], self.vector[2],
                                     length=1, color=self.col, normalize=True, label=self.label)

    def rotate(self, theta=None, phi=None):
        if not self.is_rotate:
            return
        if theta is not None:
            self.theta_init = theta
        if phi is not None:
            self.phi_init = phi
        vector_rotated = spherical_to_cartesian(self.r, self.theta_init, self.phi_init)
        self.vector = vector_rotated[0], vector_rotated[1], vector_rotated[2]
        self.update_quiver()

    def set_rotate(self, flag):
        self.is_rotate = flag

    def set_z(self, z):
        self.z = z
        self.update_quiver()

    def get_vector(self):
        return self.vector

    def get_point(self):
        return np.array([self.x, self.y, self.z])


class ConeSurfaceVectorGenerator:
    def __init__(self, vector, angle_deg, resolution):
        self.vector = vector / np.linalg.norm(vector)
        self.angle = np.deg2rad(180. - angle_deg)
        self.resolution = resolution

        self.base_vector = np.array([0., 0., 1.])
        self.rotation = Rotation.align_vectors([self.vector], [self.base_vector])[0]
        self._generate_vectors()

    def _generate_vectors(self):
        self.vectors = []

        for i in range(self.resolution):
            theta = 2 * np.pi * i / self.resolution

            x = np.sin(self.angle) * np.cos(theta)
            y = np.sin(self.angle) * np.sin(theta)
            z = np.cos(self.angle)
            vector = np.array([x, y, z])

            vector_rotated = self.rotation.apply(vector)
            self.vectors.append(vector_rotated)

        return np.array(self.vectors)

    def set_angle(self, angle_deg):
        self._generate_vectors()
        self.angle = np.deg2rad(180. - angle_deg)

    def set_vector(self, new_vector):
        self.vector = new_vector / np.linalg.norm(new_vector)
        self.rotation = Rotation.align_vectors([self.vector], [self.base_vector])[0]

    def get_vectors(self):
        return self.vectors

    def get_rotate_vectors(self, theta, phi):
        rot_matrix_theta = Rotation.from_rotvec(theta * vector_y_axis)
        rot_matrix_phi = Rotation.from_rotvec(phi * vector_z_axis)
        rotated_vectors = []
        for vector in self.vectors:
            vector_rotated_theta = rot_matrix_theta.apply(vector)
            vector_rotated = rot_matrix_phi.apply(vector_rotated_theta)
            rotated_vectors.append(vector_rotated)
        return rotated_vectors


class ConePlotter:
    def __init__(self, ax, apex_point, cone_vectors, length_slant_line, line_width, color):
        self.ax = ax
        self.apex_point = apex_point
        self.cone_vectors = cone_vectors
        self.length_slant_line = length_slant_line
        self.line_width = line_width
        self.color = color

        self.cone_plots = []
        self.base_line_points = []
        self.base_line_plot = None

        self.plot_cone()

    def plot_cone(self):
        for vector in self.cone_vectors:
            if vector[2] > 0:
                scale_factor_z = np.inf
            elif vector[2] != 0:
                scale_factor_z = self.apex_point[2] / -vector[2]
            else:
                scale_factor_z = np.inf

            if scale_factor_z < self.length_slant_line:
                end_point = [self.apex_point[0] + vector[0] * scale_factor_z,
                             self.apex_point[1] + vector[1] * scale_factor_z,
                             0]
                self.base_line_points.append(end_point[:2])  # (x, y)を保存
            else:
                end_point = [self.apex_point[0] + vector[0] * self.length_slant_line,
                             self.apex_point[1] + vector[1] * self.length_slant_line,
                             self.apex_point[2] + vector[2] * self.length_slant_line]

            plot = art3d.Line3D([self.apex_point[0], end_point[0]],
                                [self.apex_point[1], end_point[1]],
                                [self.apex_point[2], end_point[2]], linewidth=self.line_width, color=self.color)
            self.ax.add_line(plot)
            self.cone_plots.append(plot)

        self.plot_base()

    def plot_base(self):
        if self.base_line_points:
            base_line_array = np.array(self.base_line_points)
            if self.base_line_plot is None:
                self.base_line_plot, = self.ax.plot(base_line_array[:, 0], base_line_array[:, 1], np.zeros_like(base_line_array[:, 0]), color="r", linestyle="-")
            else:
                self.base_line_plot.set_data(base_line_array[:, 0], base_line_array[:, 1])
                self.base_line_plot.set_3d_properties(np.zeros_like(base_line_array[:, 0]))

    def re_plot(self, apex_point, cone_vectors):
        self.apex_point = apex_point
        self.cone_vectors = cone_vectors
        self.base_line_points = []

        for plot, vector in zip(self.cone_plots, self.cone_vectors):
            if vector[2] > 0:
                scale_factor_z = np.inf
            elif vector[2] != 0:
                scale_factor_z = self.apex_point[2] / -vector[2]
            else:
                scale_factor_z = np.inf

            if scale_factor_z < self.length_slant_line:
                end_point = [self.apex_point[0] + vector[0] * scale_factor_z,
                             self.apex_point[1] + vector[1] * scale_factor_z,
                             0]
                self.base_line_points.append(end_point[:2])  # (x, y)を保存
            else:
                end_point = [self.apex_point[0] + vector[0] * self.length_slant_line,
                             self.apex_point[1] + vector[1] * self.length_slant_line,
                             self.apex_point[2] + vector[2] * self.length_slant_line]

            plot.set_data([self.apex_point[0], end_point[0]], [self.apex_point[1], end_point[1]])
            plot.set_3d_properties([self.apex_point[2], end_point[2]])

        self.plot_base()


def rotate(theta, phi):
    global direction_theta, direction_phi
    direction_theta, direction_phi = theta, phi
    arrow.rotate(theta=np.deg2rad(direction_theta), phi=np.deg2rad(direction_phi))
    cone_vectors = cone_vector_generator.get_rotate_vectors(np.deg2rad(direction_theta), np.deg2rad(direction_phi))
    cone_plotter.re_plot(apex_cone, cone_vectors)


def set_height(height):
    global apex_cone
    arrow.set_z(height)
    apex_cone[2] = height
    cone_vectors = cone_vector_generator.get_rotate_vectors(np.deg2rad(direction_theta), np.deg2rad(direction_phi))
    cone_plotter.re_plot(apex_cone, cone_vectors)


def set_angle(angle):
    cone_vector_generator.set_angle(angle)
    cone_vector_generator.get_vectors()
    cone_vectors = cone_vector_generator.get_rotate_vectors(np.deg2rad(direction_theta), np.deg2rad(direction_phi))
    cone_plotter.re_plot(apex_cone, cone_vectors)


def create_parameter_setter():
    frm_spin_cone = ttk.Labelframe(root, relief="ridge", text="Cone", labelanchor='n')
    frm_spin_cone.pack(side='left', fill=tk.Y)

    lbl_cone_height = tk.Label(frm_spin_cone, text="Height")
    lbl_cone_height.pack(side="left")

    var_cone_height = tk.StringVar(root)
    var_cone_height.set(str(height_cone))
    spn_cone_height = tk.Spinbox(
        frm_spin_cone, textvariable=var_cone_height, format="%.1f", from_=1, to=4, increment=.1,
        command=lambda: set_height(float(var_cone_height.get())), width=5
    )
    spn_cone_height.pack(side="left")

    lbl_cone_angle = tk.Label(frm_spin_cone, text="Angle")
    lbl_cone_angle.pack(side="left")

    var_cone_angle = tk.StringVar(root)
    var_cone_angle.set(str(angle_cone_deg))
    spn_cone_angle = tk.Spinbox(
        frm_spin_cone, textvariable=var_cone_angle, format="%.1f", from_=0, to=180, increment=1,
        command=lambda: set_angle(float(var_cone_angle.get())), width=5
    )
    spn_cone_angle.pack(side="left")

    lbl_cone_dir_theta = tk.Label(frm_spin_cone, text="Direction(theta)")
    lbl_cone_dir_theta.pack(side="left")
    var_cone_dir_theta = tk.StringVar(root)
    var_cone_dir_theta.set(str(direction_theta))
    spn_cone_dir_theta = tk.Spinbox(
        frm_spin_cone, textvariable=var_cone_dir_theta, format="%.1f", from_=-360, to=360, increment=1,
        command=lambda: rotate(float(var_cone_dir_theta.get()), direction_phi), width=5
    )
    spn_cone_dir_theta.pack(side="left")

    lbl_cone_dir_phi = tk.Label(frm_spin_cone, text="Direction(phi)")
    lbl_cone_dir_phi.pack(side="left")
    var_cone_dir_phi = tk.StringVar(root)
    var_cone_dir_phi.set(str(direction_phi))
    spn_cone_dir_phi = tk.Spinbox(
        frm_spin_cone, textvariable=var_cone_dir_phi, format="%.1f", from_=-360, to=360, increment=1,
        command=lambda: rotate(direction_theta, float(var_cone_dir_phi.get())), width=5
    )
    spn_cone_dir_phi.pack(side="left")


def update_diagrams():
    pass


def reset():
    global is_play
    is_play = False
    # cnt.reset()
    # update_diagrams()


def switch():
    global is_play
    is_play = not is_play


def update(f):
    if is_play:
        # cnt.count_up()
        # update_diagrams()
        pass


""" main loop """
if __name__ == "__main__":
    arrow = Arrow(ax0, apex_cone, 0., 0., "blue", "Direction")
    cone_vector_generator = ConeSurfaceVectorGenerator(vector_z_axis, angle_cone_deg, 180)
    cone_vectors = cone_vector_generator.get_vectors()
    cone_plotter = ConePlotter(ax0, apex_cone, cone_vectors, length_slant_line, 0.5, "green")
    create_parameter_setter()
    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()
