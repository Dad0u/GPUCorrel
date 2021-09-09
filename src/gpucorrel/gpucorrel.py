# coding:utf-8


import warnings
from math import ceil
import numpy as np
import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from .fields import get_field
from .gpucorrel_level import Correl_level

context = None

kernel_path = [
    'kernels.cu',
    'kernels/kernels.cu',
    'gpucorrel/kernels/kernels.cu',
    'src/gpucorrel/kernels/kernels.cu']


def interpNearest(ary, ny, nx):
  """Used to interpolate the mask for each stage."""
  if ary.shape == (ny, nx):
    return ary
  y, x = ary.shape
  rx = x / nx
  ry = y / ny
  out = np.empty((ny, nx), dtype=np.float32)
  for j in range(ny):
    for i in range(nx):
      out[j, i] = ary[int(ry * j + .5), int(rx * i + .5)]
  return out


def find_kernel_file():
  try:
    from gpucorrel import __path__ as gpath
    l = [os.path.join(p,'../kernels/kernels.cu') for p in gpath]+kernel_path
  except ImportError:
    l = kernel_path
  for p in l:
    if os.path.exists(p):
      return p
  raise RuntimeError("Could not locate kernels.cu file,\
please use kernel_file keyword or place kernels.cu next to your Python file")


class GPUCorrel:
  """
  Identify the displacement between two images

  This class is the core of the Correl block.
  It is meant to be efficient enough to run in real-time.

  It relies on Correl_level to perform correlation on different scales.

  # Requirements #
    - The computer must have a Nvidia video card with compute capability >= 3.0
    - CUDA 5.0 or higher (only tested with CUDA 7.5)
    - pycuda 2014.1 or higher (only tested with pycuda 2016.1.1)

  # Presentation #
    This class takes a list of fields. These fields will be the base
      of deformation in which the displacement will be identified.
      When given two images, it will identify the displacement between
      the original and the second image in this base as closely as possible
      lowering square-residual using provided displacements.

    This class is highly flexible and performs on GPU for
    faster operation.

  # Usage #
    At initialization, Correl needs only one unammed argument:
      the working resolution (as a tuple of ints), which is the resolution
      of the images it will be given.
      All the images must have exactly these dimensions.
    NOTE: The dimensions must be given in this order:
      (y,x) (like openCV images)

    At initialization or after, this class takes a reference image.
      The deformations on this image are supposed to be all equal to 0.

    It also needs a number of deformation fields (technically limited
      to ~500 fields, probably much less depending on the resolution
      and the amount of memory on the graphics card).

    Finally, you need to provide the deformed image you want to process
      It will then identify parameters of the sum of fields that lowers the
      square sum of differences between the original image and the
      second one displaced with the resulting field.

    This class will resample the images and perform identification on a
      lower resolution, use the result to initialize the next stage,
      and again util it reaches the last stage. It will then return
      the computed parameters. The number of levels can be set with
      levels=x (see USAGE)

    The latest parameters returned (if any) are used to initialize
      computation when called again, to help identify large displacement.
      It is particularly adapted to slow deformations.

    To lower the residual, this program computes the gradient of each
      parameter and uses Newton method to converge as fast as possible.
      The number of iterations for the resolution can also be set (see USAGE).

    # Parameters #
    ## Single positional arg ##
    - img_size: tuple of 2 ints, (y,x), the working resolution

    The constructor can take a variety of keyword arguments:
    ## Verbose ##
      Use verbose=x to choose the amount of information printed to the console:
    - 0: Nothing except for errors
    - 1: Only important infos and warnings
    - 2: Major info and a few values periodacally (at a bearable rate)
    - 3: Tons of info including details of each iteration

    Note that verbose=3 REALLY slows the program down.
    To be used only for debug.

    ## Fields ##
      Use fields=[...] to set the fields.
        This can be done later with set_fields(), however
        in case when the fields are set later, you need to add fields_count=x
        to specifiy at init the number of expected fields in order to allocate
        all the necessary memory on the device.

      The fields should be given as a list of tuples
        of 2 numpy.ndarrays or gpuarray.GPUArray of the size of the image,
        each array corresponds to the displacement in pixel along
        respectively X and Y

      Alternatively, fields can be given as (y,x,2) ndarrays where
      a[:,:,0] holds the displacement along X and a[:,:,1] along Y

    You can also use a string instead of the tuple for the common fields:

      Rigid body and linear deformations:
      - 'x': Movement along X
      - 'y': Movement along Y
      - 'r': Rotation (in the trigonometric direction)
      - 'exx': Stretch along X
      - 'eyy': Stretch along Y
      - 'exy': Shear
      - 'z': Zoom (dilatation) (=exx+eyy)

      Note that you should not try to identify exx,eyy AND z at the same time
      (one of them is redundant)

      Quadratic deformations:

      These fields are more complicated to interpret but can be useful
        for complicated sollicitations such as biaxial stretch.\n
        U and V represent the displacement along respectively x and y.
      - 'uxx': U(x,y) = x²
      - 'uyy': U(x,y) = y²
      - 'uxy': U(x,y) = xy
      - 'vxx': V(x,y) = x²
      - 'vyy': V(x,y) = y²
      - 'vxy': V(x,y) = xy

      All of these default fields are normalized to have a max displacement
        of 1 pixel and are centered in the middle of the image.
        They are generated to have the size of your image.

      You can mix strings and tuples at your convenience to perform
        your identification.

      Example:
          fields=['x','y',(MyFieldX,MyFieldY)]

      where MyfieldX and MyfieldY are numpy arrays with the same shape
      as the images

      Example of memory usage: On a 2048x2048 image,
      count roughly 180 + 100*fields_count MB of VRAM

    ## Original image ##
    It must be given as a 2D numpy.ndarray. This block works with
      dtype=np.float32. If the dtype of the given image is different,
      it will print a warning and the image will be converted.
      It can be given at init with the kwarg img=MyImage
      or later with set_ref(MyImage).
      Note: You can reset it whenever you want, even multiple times but
      it will reset the def parameters to 0.

    Once fields and original image are set, there is a short
      preparation time before correlation can be performed.
      You can do this preparation yourself by using .prepare().
      If not called, it will be done automatically
      when necessary, inducing a slight overhead at the first
      call of .compute() after setting/updating the fields or original image

    ## Compared image ##
    It can be given directly when querying the displacement as a parameter to
      compute() or before, with set_image().
      You can provide it as a np.ndarray just like orig,
      or as a pycuda.gpuarray.GPUArray.

    ## Editing the behavior ##
    Kwargs:
      levels: <b>int, >= 1, default=5</b>\n
        Number of levels of the pyramid\n
          More levels can help converging with large and quick deformations
          but may fail on images without low spatial frequency.\n
          Less levels -> program will run faster

      resampling_factor: <b>float, > 1, default=2</b>\n
        The resolution will be divided by this parameter between each stage of
          the pyramid.\n
          Low -> Can allow coherence between stages but is more expensive\n
          High -> Reach small resolutions in less levels -> faster but be
          careful not to loose consistency between stages

      iterations: <b>int, x>=1, default=4</b>\n
          The MAXIMUM number of iteration to be run before returning the
          values.\n
          Note that if the residual increases before
          reaching x iterations, the block will return anyways.

      img: <b>numpy.ndarray, img.shape=img_size, default: None</b>\n
          If you want to set the original image at init.

      mask: <b>numpy.ndarray, mask.shape=img_size, default: None</b>\n
        To set the mask, to weight the zone of interest on the images.
        It is particularly useful to prevent undesired effects on the border
        of the images.\n
        If no mask is given, a rectangular mask will be used, with border
        of 5% the size of the image.

      show_diff: <b>Boolean, ,default=False</b>\n
          Will open a cv2 window and print the difference between the
          original and the displaced image after correlation.\n
          128 Gray means no difference,
          lighter means positive and darker negative.

      kernel_file: <b>string, path, default=None</b>\n
        Path of the kernel file to use
        If None, will try several likely locations

      mul: <b>float, > 0, default=3</b>\n
        This parameter is critical.\n
        The direction will be multiplied by this scalar before being added
        to the solution.\n
        It defines how "fast" we move towards the solution.\n
        High value -> Fast convergence but risk to go past the solution
        and diverge (the program does not try to handle this and
        if the residual rises, iterations will stop immediately).\n
        Low value -> Probably more precise but slower and may require
        more iterations.\n
        After multiple tests, 3 was found to be a pretty
        acceptable value. Don't hesitate to adapt it to your case.
        Use verbose=3 and see if the convergence is too slow or too fast.

      use_last: <b> bool, default=True</b>\n
        If True, the last computed field will be used to initialize the field
        on the next call to .compute(). Else, it will be 0 everywhere


  \todo
    This section lists all the considered improvements for this program.
    These features may NOT all be implemented in the future.
    - Allow faster execution by executing the reduction only on a part
         of the images (random or chosen)
    - Restart iterating from 0 once in a while to see if the residual is lower.
         Can be useful to recover when diverged critically due to an incorrect
         image (Shadow, obstruction, flash, camera failure, ...)
  """

  def __init__(self, img_size, **kwargs):
    cuda.init()
    from pycuda.tools import make_default_context
    global context
    context = make_default_context()
    unknown = []
    for k in kwargs.keys():
      if k not in ['verbose', 'levels', 'resampling_factor', 'kernel_file',
                   'iterations', 'show_diff', 'fields_count', 'img',
                   'fields', 'mask', 'mul', 'use_last']:
        unknown.append(k)
    if len(unknown) != 0:
      warnings.warn("Unrecognized parameter" + (
        's: ' + str(unknown) if len(unknown) > 1 else ': ' + unknown[0]),
        SyntaxWarning)
    self.verbose = kwargs.get("verbose", 0)
    self.debug(3, "You set the verbose level to the maximum.\n\
It may help finding bugs or tracking errors but it may also \
impact the program performance as it will print A LOT of \
output and add GPU->CPU copies only to print information.\n\
If it is not desired, consider lowering the verbosity: \
1 or 2 is a reasonable choice, \
0 won't show anything except for errors.")
    self.levels = kwargs.get("levels", 5)
    self.loop = 0
    self.resamplingFactor = kwargs.get("resampling_factor", 2)
    h, w = img_size
    self.nb_iter = kwargs.get("iterations", 4)
    self.debug(1, "Initializing... Master resolution:", img_size,
               "levels:", self.levels, "verbosity:", self.verbose)

    # Computing dimensions of the different levels #
    self.h, self.w = [], []
    for i in range(self.levels):
      self.h.append(int(round(h / (self.resamplingFactor ** i))))
      self.w.append(int(round(w / (self.resamplingFactor ** i))))

    if kwargs.get("fields_count") is not None:
      self.fields_count = kwargs.get("fields_count")
    else:
      try:
        self.fields_count = len(kwargs["fields"])
      except KeyError:
        self.debug(0, "Error! You must provide the number of fields at init. \
Add fields_count=x or directly set fields with fields=list/tuple")
        raise ValueError

    kernel_file = kwargs.get("kernel_file")
    if kernel_file is None:
      kernel_file = find_kernel_file()
    self.debug(3, "Kernel file:", kernel_file)

    # Creating a new instance of Correl_level for each stage #
    self.correl = []
    for i in range(self.levels):
      self.correl.append(Correl_level((self.h[i], self.w[i]),
                                     verbose=self.verbose,
                                     fields_count=self.fields_count,
                                     iterations=self.nb_iter,
                                     show_diff=(i == 0 and kwargs.get(
                                         "show_diff", False)),
                                     mul=kwargs.get("mul", 3),
                                     kernel_file=kernel_file))

    # Set original image if provided #
    if kwargs.get("img") is not None:
      self.set_ref(kwargs.get("img"))

    s = """
texture<float, cudaTextureType2D, cudaReadModeElementType> texFx{0};
texture<float, cudaTextureType2D, cudaReadModeElementType> texFy{0};
__global__ void resample{0}(float* outX, float* outY, const int x, const int y)
{{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idy = blockIdx.y*blockDim.y+threadIdx.y;
  if(idx < x && idy < y)
  {{
    outX[idy*x+idx] = tex2D(texFx{0},(float)idx/x, (float)idy/y);
    outY[idy*x+idx] = tex2D(texFy{0},(float)idx/x, (float)idy/y);
  }}
}}
    """
    self.src = ""
    for i in range(self.fields_count):
      self.src += s.format(i) # Adding textures for the quick fields resampling

    self.mod = SourceModule(self.src)

    self.texFx = []
    self.texFy = []
    self.resampleF = []
    for i in range(self.fields_count):
      self.texFx.append(self.mod.get_texref("texFx%d" % i))
      self.texFy.append(self.mod.get_texref("texFy%d" % i))
      self.resampleF.append(self.mod.get_function("resample%d" % i))
      self.resampleF[i].prepare("PPii", texrefs=[self.texFx[i], self.texFy[i]])

    for t in self.texFx + self.texFy:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0, cuda.address_mode.BORDER)
      t.set_address_mode(1, cuda.address_mode.BORDER)

    # Set fields if provided #
    if kwargs.get("fields") is not None:
      self.set_fields(kwargs.get("fields"))

    if kwargs.get("mask") is not None:
      self.set_mask(kwargs.get("mask"))

    self.use_last = kwargs.get('use_last', True)

  def get_fields(self, y=None, x=None):
    """Returns the fields, resampled to size (y,x)"""
    if x is None or y is None:
      y = self.h[0]
      x = self.w[0]
    outX = gpuarray.empty((self.fields_count, y, x), np.float32)
    outY = gpuarray.empty((self.fields_count, y, x), np.float32)
    grid = (int(ceil(x / 32)), int(ceil(y / 32)))
    block = (int(ceil(x / grid[0])), int(ceil(y / grid[1])), 1)
    for i in range(self.fields_count):
      self.resampleF[i].prepared_call(grid, block,
                                      outX[i, :, :].gpudata,
                                      outY[i, :, :].gpudata,
                                      np.int32(x), np.int32(y))
    return outX, outY

  def debug(self, n, *s):
    """
    To print debug info

    First argument is the level of the message.
    It wil be displayed only if the self.debug is superior or equal
    """
    if n <= self.verbose:
      print("  " * (n - 1) + "[Correl]", *s)

  def set_ref(self, img):
    """
    To set the reference image

    When calling compute, the displacement between the given image and
    this reference will be computed and returned
    """
    self.debug(2, "updating original image")
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert len(img.shape) == 2, "Image must have 2 dimensions (got {})" \
      .format(len(img.shape))
    assert img.shape == (self.h[0], self.w[0]), "Wrong size!"
    if img.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 \
to allow GPU computing (got {}). Converting to float32."
                    .format(img.dtype), RuntimeWarning)
      img = img.astype(np.float32)

    self.correl[0].set_ref(img)
    for i in range(1, self.levels):
      self.correl[i - 1].resample_ref(self.h[i], self.w[i],
                                      self.correl[i].devRef)
      self.correl[i].update_ref()

  def set_fields(self, fields):
    assert self.fields_count == len(fields), \
      "Cannot change the number of fields on the go!"
    # Choosing the right function to copy
    if isinstance(fields[0], str) or isinstance(fields[0][0], np.ndarray):
      toArray = cuda.matrix_to_array
    elif isinstance(fields[0][0], gpuarray.GPUArray):
      toArray = cuda.gpuarray_to_array
    else:
      self.debug(0, "Error ! Incorrect fields argument. \
See docstring of Correl")
      raise ValueError
    # These list store the arrays for the fields texture
    # (to be interpolated quickly for each stage)
    self.fieldsXArray = []
    self.fieldsYArray = []
    for i,f in enumerate(fields):
      if isinstance(f, tuple): # If tuple, check the shape
        assert len(f) == 2,"fields n°{} is invalid".format(i+1)
        assert f[0].shape == f[1].shape == (self.h[0],self.w[0]),\
        "fields n°{} is invalid".format(i+1)
      elif isinstance(f, str): # If string, convert to tuple
        fields[i] = get_field(f.lower(), self.h[0],self.w[0])
      # If (y,x,2) ndarray, check shape and convert to tuple
      elif isinstance(f, np.ndarray):
        assert fields[i].shape == (self.h[0],self.w[0],2),\
            "fields n°{} is invalid".format(i+1)
        fields[i] = (f[:,:,0],f[:,:,1])

      self.fieldsXArray.append(toArray(fields[i][0], "C"))
      self.texFx[i].set_array(self.fieldsXArray[i])
      self.fieldsYArray.append(toArray(fields[i][1], "C"))
      self.texFy[i].set_array(self.fieldsYArray[i])
    for i in range(self.levels):
      self.correl[i].set_fields(*self.get_fields(self.h[i], self.w[i]))

  def prepare(self):
    for c in self.correl:
      c.prepare()
    self.debug(2, "Ready!")

  def save_all_reference_images(self, name="out"):
    import cv2
    self.debug(1, "Saving all images with the name", name + "X.png")
    for i in range(self.levels):
      out = self.correl[i].devRef.get().astype(np.uint8)
      cv2.imwrite(name + str(i) + ".png", out)

  def set_image(self, img_d):
    if img_d.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 "
"to allow GPU computing (got {}). Converting to float32."
                    .format(img_d.dtype), RuntimeWarning)
      img_d = img_d.astype(np.float32)
    self.correl[0].set_image(img_d)
    for i in range(1, self.levels):
      self.correl[i].set_image(
        self.correl[i - 1].resampleD(self.correl[i].h, self.correl[i].w))

  def set_mask(self, mask):
    for l in range(self.levels):
      self.correl[l].set_mask(interpNearest(mask, self.h[l], self.w[l]))

  def compute(self, img_d=None):
    """
    To get the displacement

    This will perform the correlation routine on each stage, initializing with
    the previous values every time it will return the computed parameters
    as a list
    """
    self.loop += 1
    if img_d is not None:
      self.set_image(img_d)
    try:
      assert self.use_last
      disp = self.last / (self.resamplingFactor ** self.levels)
    except (AttributeError, AssertionError):
      disp = np.array([0] * self.fields_count, dtype=np.float32)
    for i in reversed(range(self.levels)):
      disp *= self.resamplingFactor
      self.correl[i].setDisp(disp)
      disp = self.correl[i].compute()
      self.last = disp
    # Every 10 images, print the values (if debug >=2)
    if self.loop % 10 == 0 and self.verbose >= 2:
      self.debug(2, "Loop", self.loop, ", values:", self.correl[0].devX.get(),
                 ", res:", self.correl[0].res / 1e6)
    return disp

  def get_res(self, lvl=0):
    """
    Returns the last residual of the sepcified level (0 by default)

    Usually, the correlation is correct when res < ~1e9-10 but it really
    depends on the images: you need to find the value that suit your own
    images, depending on the resolution, contrast, correlation method etc...
    You can use save_diff to visualize the difference between the
    two images after correlation.
    """
    return self.correl[lvl].res

  def save_diff(self, fname=None, level=0):
    """
    To see the difference between the two images with the computed parameters.

    It writes a single channel picture named "diff.png" where 128 gray is
    exact equality, lighter pixels show positive difference and darker pixels
    a negative difference. Useful to see if correlation succeded and to
    identify the origin of non convergence
    """
    self.correl[level].save_diff(fname)

  def clean(self):
    """Needs to be called at the end, to destroy the context properly"""
    context.pop()
