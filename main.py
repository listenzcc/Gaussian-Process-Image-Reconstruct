# %%
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

# %%
image_paths = [e for e in Path('image').iterdir()]
image_paths

# %%


def sample_and_reconstruct(image_path, dst_folder):
    '''

    '''
    # --------------------------------------------------------------------------------
    # Read the image, and shrink the image to 200 px width
    image_name = image_path.name
    print(image_name, image_path)

    raw_image_mat = np.array(Image.open(image_path))
    raw_width = raw_image_mat.shape[1]

    # Make the image is not too small
    assert raw_width > 200, 'Too small the image is.'

    down_sample = int(raw_width / 200)
    raw_image_mat = raw_image_mat[::down_sample, ::down_sample, :]
    print(raw_image_mat.shape)

    # --------------------------------------------------------------------------------
    # Random select n=2000 pixels from the image
    n = 2000

    xgrid, ygrid = np.meshgrid(
        range(raw_image_mat.shape[1]), range(raw_image_mat.shape[0]))
    print(xgrid.shape, ygrid.shape, xgrid, ygrid)

    a = np.ravel(xgrid)[:, np.newaxis]
    b = np.ravel(ygrid)[:, np.newaxis]
    xy_grid_all = np.concatenate([a, b], axis=1)

    np.random.shuffle(xy_grid_all)
    xy_grid_select = xy_grid_all[:n]
    data_select = raw_image_mat[xy_grid_select[:, 1], xy_grid_select[:, 0], :]
    print(xy_grid_all.shape, xy_grid_select.shape, data_select.shape)

    # --------------------------------------------------------------------------------
    # Train and predict using the GaussianProcess
    # The mean variable is the restructured image pixels
    # and the order is as the same as the xy_grid_all
    X = xy_grid_select
    y = data_select

    # kernel = kernels.DotProduct() + kernels.WhiteKernel()
    # kernel = kernels.DotProduct() + kernels.RBF()
    kernel = kernels.Matern(nu=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    gpr.score(X, y)

    pred = gpr.predict(xy_grid_all, return_std=True)
    mean, std = pred
    print(mean.shape, std.shape)

    # --------------------------------------------------------------------------------
    # Display
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    axes[0].imshow(raw_image_mat)
    axes[0].set_title('Raw image')

    data_select = raw_image_mat[xy_grid_select[:, 1], xy_grid_select[:, 0], :]
    d = raw_image_mat - raw_image_mat
    d[xy_grid_select[:, 1], xy_grid_select[:, 0], :] = data_select
    axes[1].imshow(d)
    axes[1].set_title('Sample')

    pred_mat = raw_image_mat - raw_image_mat
    pred_mat[xy_grid_all[:, 1], xy_grid_all[:, 0], :] = mean
    axes[2].imshow(pred_mat)
    axes[2].set_title('Reconstruct')

    fig.tight_layout()

    folder = Path(dst_folder)
    folder.mkdir(exist_ok=True)
    fig.savefig(folder.joinpath(image_name))

    plt.show()


# %%

for path in tqdm(image_paths):
    sample_and_reconstruct(path, dst_folder='reconstruct-matern-nu=0.1')
print('All done.')

# %%
