#%%
import matplotlib.pyplot as plt
import numpy as np

import pylab as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import h5py
# from matplotlib.colors import Normalize, LinearSegmentedColormap

def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data = hdf["data"]
        cavity_drive_I = hdf["x"]
        cavity_drive_Q = hdf[f"y"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data[:, :, :]),
            cavity_drive_I[:],
            cavity_drive_Q[:],
        )

def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x_rot = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    y_rot = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    return A * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)) + offset

    
def make_the_plot(filename: str):
    (
        data,
        cavity_drive_I,
        cavity_drive_Q,
    ) = read_hdf5_file(filename)
    displacement_scale = 1.5
    ### Fit vacuum to 2D gaussian
    vacuum_data = data[:, :, 0] - data[:, :, 1]
    x_grid, y_grid = np.meshgrid(cavity_drive_I, cavity_drive_Q)
    xy = np.vstack((x_grid.ravel(), y_grid.ravel()))
    initial_guess = [100e-6, 0, 0, 1, 1, 0, 0]
    params_opt, _ = curve_fit(
        lambda xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset: gaussian_2d(
            xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset
        ),
        xy,
        vacuum_data.ravel(),
        p0=initial_guess,
    )
    amplitude_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = (params_opt)
    parity_offset = offset_fit
    parity_rescaling = amplitude_fit
    # parity_corrected_0 = (data[:, :, 0] - data[:, :, 1] - parity_offset) / parity_rescaling
    parity_corrected_1 = (data[:, :, 2] - data[:, :, 3] - parity_offset) / parity_rescaling
    #### Reconstruction
    DIM = 7
    displacements_I = cavity_drive_I * displacement_scale
    displacements_Q = cavity_drive_Q * displacement_scale
    my_shape = parity_corrected_1.shape
    new_displacements = np.empty(shape=my_shape, dtype=np.complex128)
    for ii in range(my_shape[0]):
        for jj in range(my_shape[1]):
            new_displacements[ii, jj] = displacements_I[ii] + 1j * displacements_Q[jj]
    # flat_displacements = new_displacements.flatten()
    # flat_parities = parity_corrected_1.flatten()
    # unprocessed_dm = reconstruct_wigner(flat_parities, flat_displacements, DIM)
    # rho = qutip.Qobj(unprocessed_dm).unit()
    # rotation_angle_14 = np.angle(rho.full()[1, 4]) / 2 / np.pi
    # rho_MLE = PSD_MLE_rho(rho=unprocessed_dm).unit()
    CM = 1/2.54
    # fig = plt.figure(figsize=(2.0924*CM, 2.0924*CM))
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_axis_off()
    # my_x = cavity_drive_I * displacement_scale
    # my_y = cavity_drive_Q * displacement_scale
    # my_z = parity_corrected_1.T
    # print("Length x: {}, Length y: {}".format(len(my_x), len(my_y)))
    # ax.pcolormesh(
    #     my_x,
    #     my_y,
    #     my_z,
    #     cmap="bwr",
    #     vmax=1,
    #     vmin=-1,
    #     rasterized=True # NOTE new
    # )
    # plt.gca().set_aspect("equal")
    rot_angle = 64.98  # degrees (counter-clockwise)
    theta = np.deg2rad(rot_angle)

    fig = plt.figure(figsize=(2.0924 * CM, 2.0924 * CM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
    ax.set_axis_off()

    my_x = cavity_drive_I * displacement_scale
    my_y = cavity_drive_Q * displacement_scale
    my_z = parity_corrected_1.T

    # 1. Store original square limits before rotating
    x_min, x_max = my_x.min(), my_x.max()
    y_min, y_max = my_y.min(), my_y.max()

    # 2. Build 2D coordinate grids (safe whether my_x/my_y are 1D or already 2D)
    if my_x.ndim == 1 and my_y.ndim == 1:
        X, Y = np.meshgrid(my_x, my_y)
    else:
        X, Y = my_x, my_y

    # 3. Rotate coordinates around the center of the grid (0, 0 if symmetric)
    x0 = 0.5 * (x_min + x_max)
    y0 = 0.5 * (y_min + y_max)

    X_rot = (X - x0) * np.cos(theta) - (Y - y0) * np.sin(theta) + x0
    Y_rot = (X - x0) * np.sin(theta) + (Y - y0) * np.cos(theta) + y0

    # 4. Plot the rotated mesh
    ax.pcolormesh(
        X_rot,
        Y_rot,
        my_z,
        cmap="bwr",
        vmax=1,
        vmin=-1,
        shading="auto",
        rasterized=True,
    )

    # 5. Lock limits to the original square so corners stay white and scale is preserved
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    return fig

#%%
path = "data/"
filename = "20250807_154004_lakeside_0p2p4.h5"
fig = make_the_plot(filename=path+filename)
fig.savefig("Figures/Wigner_024.pdf", dpi=800, pad_inches=0, facecolor="white")
plt.close(fig)