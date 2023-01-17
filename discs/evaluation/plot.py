"""Plot tools."""

from discs.common import utils
from flax import jax_utils
import jax
import numpy as np
import PIL


def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
  """NumPy utility: tile a batch of images into a single image.
  Args:
    imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
    pad_pixels: int: number of pixels of padding to add around each image
    pad_val: int: padding value
    num_col: int: number of columns in the tiling; defaults to a square
  Returns:
    np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
  """
  if pad_pixels < 0:
    raise ValueError('Expected pad_pixels >= 0')
  if not 0 <= pad_val <= 255:
    raise ValueError('Expected pad_val in [0, 255]')

  imgs = np.asarray(imgs)
  if imgs.dtype != np.uint8:
    raise ValueError('Expected uint8 input')
  # if imgs.ndim == 3:
  #   imgs = imgs[..., None]
  n, h, w, c = imgs.shape
  if c not in [1, 3]:
    raise ValueError('Expected 1 or 3 channels')

  if num_col <= 0:
    # Make a square
    ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
    num_row = ceil_sqrt_n
    num_col = ceil_sqrt_n
  else:
    # Make a B/num_per_row x num_per_row grid
    assert n % num_col == 0
    num_row = int(np.ceil(n / num_col))

  imgs = np.pad(
      imgs,
      pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
                 (pad_pixels, pad_pixels), (0, 0)),
      mode='constant',
      constant_values=pad_val)
  h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
  imgs = imgs.reshape(num_row, num_col, h, w, c)
  imgs = imgs.transpose(0, 2, 1, 3, 4)
  imgs = imgs.reshape(num_row * h, num_col * w, c)

  if pad_pixels > 0:
    imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
  if c == 1:
    imgs = imgs[..., 0]
  return imgs


def plot_shareded_image(
    fname, images, image_height, image_width, image_channel,
    rescale=1.0, max_num_plot=64):
  samples = utils.all_gather(images)
  if jax.process_index() == 0:
    samples = jax.device_get(jax_utils.unreplicate(samples))
    samples = np.reshape(
        samples, [-1, image_height, image_width, image_channel]) * rescale
    images = np_tile_imgs(samples.astype(np.uint8)[:max_num_plot])
    with open(fname, 'wb') as f:
      img = PIL.Image.fromarray(images)
      img.save(f)
