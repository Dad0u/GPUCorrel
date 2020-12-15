# coding:utf-8


from math import ceil
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
import cv2


class Correl_level:
  """
  Run a correlation routine on an image, at a given resolution.

  This class is intended to be a level of the pyramid in GPUCorrel
  """
  num = 0  # To count the instances so they get a unique number (self.num)

  def __init__(self, img_size, **kwargs):
    self.num = Correl_level.num
    Correl_level.num += 1
    self.verbose = kwargs.get("verbose", 0)
    self.debug(2, "Initializing with resolution", img_size)
    self.h, self.w = img_size
    self._ready = False
    self.nb_iter = kwargs.get("iterations", 5)
    self.show_diff = kwargs.get("show_diff", False)
    if self.show_diff:
      import cv2
      cv2.namedWindow("Residual", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    self.mul = kwargs.get("mul", 3)
    # These two store the values of the last resampled array
    # It is meant to allocate output array only once (see resampleD)
    self.rX, self.rY = -1, -1
    # self.loop will be incremented every time compute is called
    # It will be used to measure performance and output some info
    self.loop = 0

    # Allocating stuff #

    # Grid and block for kernels called with the size of the image #
    # All the images and arrays in the kernels will be in order (x,y)
    self.grid = (int(ceil(self.w / 32)),
                 int(ceil(self.h / 32)))
    self.block = (int(ceil(self.w / self.grid[0])),
                  int(ceil(self.h / self.grid[1])), 1)
    self.debug(3, "Default grid:", self.grid, "block", self.block)

    # We need the number of fields to allocate the G tables #
    self.fields_count = kwargs.get("fields_count")
    if self.fields_count is None:
      self.fields_count = len(kwargs.get("fields")[0])

    # Allocating everything we need #
    self.devG = []
    self.devFieldsX = []
    self.devFieldsY = []
    for i in range(self.fields_count):
      # devG stores the G arrays (to compute the research direction)
      self.devG.append(gpuarray.empty(img_size, np.float32))
      # devFieldsX/Y store the fields value along X and Y
      self.devFieldsX.append(gpuarray.empty((self.h, self.w), np.float32))
      self.devFieldsY.append(gpuarray.empty((self.h, self.w), np.float32))
    # devH Stores the Hessian matrix
    self.H = np.zeros((self.fields_count, self.fields_count), np.float32)
    # And devHi stores its invert
    self.devHi = gpuarray.empty(
        (self.fields_count, self.fields_count), np.float32)
    # devOut is written with the difference of the images
    self.devOut = gpuarray.empty((self.h, self.w), np.float32)
    # devX stores the value of the parameters (what is actually computed)
    self.devX = gpuarray.empty((self.fields_count), np.float32)
    # to store the research direction
    self.devVec = gpuarray.empty((self.fields_count), np.float32)
    # To store the reference image on the device
    self.devRef = gpuarray.empty(img_size, np.float32)
    # To store the gradient along X of the original image on the device
    self.devGradX = gpuarray.empty(img_size, np.float32)
    # And along Y
    self.devGradY = gpuarray.empty(img_size, np.float32)

    # Locating the kernel file #
    kernel_file = kwargs.get("kernel_file")
    if kernel_file is None:
      self.debug(2, "Kernel file not specified")
      kernel_file = "kernels.cu"
    # Reading kernels and compiling module #
    with open(kernel_file, "r") as f:
      self.debug(3, "Sourcing module")
      self.mod = SourceModule(f.read() % (self.w, self.h, self.fields_count))
    # Assigning functions to the kernels #
    # These kernels are defined in data/kernels.cu
    self._resampleRefKrnl = self.mod.get_function('resampleR')
    self._resampleKrnl = self.mod.get_function('resample')
    self._gradientKrnl = self.mod.get_function('gradient')
    self._makeGKrnl = self.mod.get_function('makeG')
    self._makeDiff = self.mod.get_function('makeDiff')
    self._dotKrnl = self.mod.get_function('myDot')
    self._addKrnl = self.mod.get_function('kadd')
    # These ones use pyCuda reduction module to generate efficient kernels
    self._mulRedKrnl = ReductionKernel(np.float32, neutral="0",
                                     reduce_expr="a+b", map_expr="x[i]*y[i]",
                                     arguments="float *x, float *y")
    self._leastSquare = ReductionKernel(np.float32, neutral="0",
                                     reduce_expr="a+b", map_expr="x[i]*x[i]",
                                     arguments="float *x")
    # We could have used use mulRedKrnl(x,x), but this is probably faster ?

    # Getting texture references #
    self.tex = self.mod.get_texref('tex')
    self.tex_d = self.mod.get_texref('tex_d')
    self.texMask = self.mod.get_texref('texMask')
    # Setting proper flags #
    # All textures use normalized coordinates except for the mask
    for t in [self.tex, self.tex_d]:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    for t in [self.tex, self.tex_d, self.texMask]:
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0, cuda.address_mode.BORDER)
      t.set_address_mode(1, cuda.address_mode.BORDER)

    # Preparing kernels for less overhead when called #
    self._resampleRefKrnl.prepare("Pii", texrefs=[self.tex])
    self._resampleKrnl.prepare("Pii", texrefs=[self.tex_d])
    self._gradientKrnl.prepare("PP", texrefs=[self.tex])
    self._makeDiff.prepare("PPPP",texrefs=[self.tex, self.tex_d, self.texMask])
    self._addKrnl.prepare("PfP")
    # Reading original image if provided #
    if kwargs.get("img") is not None:
      self.set_ref(kwargs.get("img"))
    # Reading fields if provided #
    if kwargs.get("fields") is not None:
      self.set_fields(kwargs.get("fields"))
    # Reading mask if provided #
    if kwargs.get("mask") is not None:
      self.set_mask(kwargs.get("mask"))

  def debug(self, n, *s):
    """
    To print debug messages

    First argument is the level of the message.
    The others arguments will be displayed only if
    the self.debug var is superior or equal
    Also, flag and indentation reflect resectively
    the origin and the level of the message
    """
    if n <= self.verbose:
      s2 = ()
      for i in range(len(s)):
        s2 += (str(s[i]).replace("\n", "\n" + (10 + n) * " "),)
      print("  " * (n - 1) + "[Stage " + str(self.num) + "]", *s2)

  def set_ref(self, img):
    """
    To set the original image from a given CPU or GPU array.

    If it is a GPU array, it will NOT be copied.
    Note that the most efficient method is to write directly over
    self.devRef with some kernel and then run self.update_ref()
    """
    assert img.shape == (self.h, self.w), \
      "Got a {} image in a {} correlation routine!".format(
        img.shape, (self.h, self.w))
    if isinstance(img, np.ndarray):
      self.debug(3, "Setting original image from ndarray")
      self.devRef.set(img)
    elif isinstance(img, gpuarray.GPUArray):
      self.debug(3, "Setting original image from GPUArray")
      self.devRef = img
    else:
      self.debug(0, "Error ! Unknown type of data given to set_ref()")
      raise ValueError
    self.update_ref()

  def update_ref(self):
    """Needs to be called after self.img_d has been written directly"""
    self.debug(3, "Updating original image")
    self.array = cuda.gpuarray_to_array(self.devRef, 'C')
    # 'C' order implies tex2D(x,y) will fetch matrix(y,x):
    # this is where x and y are inverted to comlpy with the kernels order
    self.tex.set_array(self.array)
    self._computeGradients()
    self._ready = False

  def _computeGradients(self):
    """Wrapper to call the gradient kernel"""
    self._gradientKrnl.prepared_call(self.grid, self.block,
                                 self.devGradX.gpudata, self.devGradY.gpudata)

  def prepare(self):
    """
    Computes all necessary tables to perform correlation

    This method must be called everytime the original image or fields are set
    If not done by the user, it will be done automatically when needed
    """
    if not hasattr(self, 'maskArray'):
      self.debug(2, "No mask set when preparing, using a basic one, \
with a border of 5% the dimension")
      mask = np.zeros((self.h, self.w), np.float32)
      mask[self.h // 20:-self.h // 20, self.w // 20:-self.w // 20] = 1
      self.set_mask(mask)
    if not self._ready:
      if not hasattr(self, 'array'):
        self.debug(1, "Tried to prepare but original texture is not set !")
      elif not hasattr(self, 'fields'):
        self.debug(1, "Tried to prepare but fields are not set !")
      else:
        self._makeG()
        self._makeH()
        self._ready = True
        self.debug(3, "Ready!")
    else:
      self.debug(1, "Tried to prepare when unnecessary, doing nothing...")

  def _makeG(self):
    for i in range(self.fields_count):
      # Change to prepared call ?
      self._makeGKrnl(self.devG[i].gpudata, self.devGradX.gpudata,
                      self.devGradY.gpudata,
                      self.devFieldsX[i], self.devFieldsY[i],
                      block=self.block, grid=self.grid)

  def _makeH(self):
    for i in range(self.fields_count):
      for j in range(i + 1):
        self.H[i, j] = self._mulRedKrnl(self.devG[i], self.devG[j]).get()
        if i != j:
          self.H[j, i] = self.H[i, j]
    self.debug(3,"Hessian:\n", self.H)
    self.devHi.set(np.linalg.inv(self.H))  # *1e-3)
    # Looks stupid but prevents a useless devHi copy if nothing is printed
    if self.verbose >= 3:
      self.debug(3, "Inverted Hessian:\n", self.devHi.get())

  def resample_ref(self, newY, newX, devOut):
    """
    To resample the original image

    Reads orig.texture and writes the interpolated newX*newY
    image to the devOut array
    """
    grid = (int(ceil(newX / 32)), int(ceil(newY / 32)))
    block = (int(ceil(newX / grid[0])), int(ceil(newY / grid[1])), 1)
    self.debug(3, "Resampling ref texture, grid:", grid, "block:", block)
    self._resampleRefKrnl.prepared_call(self.grid, self.block,
                                         devOut.gpudata,
                                         np.int32(newX), np.int32(newY))
    self.debug(3, "Resampled original texture to", devOut.shape)

  def resampleD(self, newY, newX):
    """Resamples tex_d and returns it in a gpuarray"""
    if (self.rX, self.rY) != (np.int32(newX), np.int32(newY)):
      self.rGrid = (int(ceil(newX / 32)), int(ceil(newY / 32)))
      self.rBlock = (int(ceil(newX / self.rGrid[0])),
                     int(ceil(newY / self.rGrid[1])), 1)
      self.rX, self.rY = np.int32(newX), np.int32(newY)
      self.devROut = gpuarray.empty((newY, newX), np.float32)
    self.debug(3, "Resampling img_d texture to", (newY, newX),
               " grid:", self.rGrid, "block:", self.rBlock)
    self._resampleKrnl.prepared_call(self.rGrid, self.rBlock,
                                     self.devROut.gpudata,
                                     self.rX, self.rY)
    return self.devROut

  def set_fields(self, fieldsX, fieldsY):
    """
    Method to give the fields to identify with the routine.

    This is necessary only once and can be done multiple times, but the routine
    have to be initialized with .prepare(), causing a slight overhead
    Takes a tuple/list of 2 (gpu)arrays[fields_count,x,y] (one for displacement
    along x and one along y)
    """
    self.debug(2, "Setting fields")
    if isinstance(fieldsX, np.ndarray):
      self.devFieldsX.set(fieldsX)
      self.devFieldsY.set(fieldsY)
    elif isinstance(fieldsX, gpuarray.GPUArray):
      self.devFieldsX = fieldsX
      self.devFieldsY = fieldsY
    self.fields = True

  def set_image(self, img_d):
    """
    Set the image to compare with the original

    Note that calling this method is not necessary: you can do .compute(image)
    This will automatically call this method first
    """
    assert img_d.shape == (self.h, self.w), \
      "Got a {} image in a {} correlation routine!".format(
        img_d.shape, (self.h, self.w))
    if isinstance(img_d, np.ndarray):
      self.debug(3, "Creating texture from numpy array")
      self.array_d = cuda.matrix_to_array(img_d, "C")
    elif isinstance(img_d, gpuarray.GPUArray):
      self.debug(3, "Creating texture from gpuarray")
      self.array_d = cuda.gpuarray_to_array(img_d, "C")
    else:
      self.debug(0, "Error ! Unknown type of data given to .set_image()")
      raise ValueError
    self.tex_d.set_array(self.array_d)
    self.devX.set(np.zeros(self.fields_count, dtype=np.float32))

  def set_mask(self, mask):
    self.debug(3, "Setting the mask")
    assert mask.shape == (self.h, self.w), \
      "Got a {} mask in a {} routine.".format(mask.shape, (self.h, self.w))
    if not mask.dtype == np.float32:
      self.debug(2, "Converting the mask to float32")
      mask = mask.astype(np.float32)
    if isinstance(mask, np.ndarray):
      self.maskArray = cuda.matrix_to_array(mask, 'C')
    elif isinstance(mask, gpuarray.GPUArray):
      self.maskArray = cuda.gpuarray_to_array(mask, 'C')
    else:
      self.debug(0, "Error! Mask data type not understood")
      raise ValueError
    self.texMask.set_array(self.maskArray)

  def setDisp(self, X):
    assert X.shape == (self.fields_count,), \
      "Incorrect initialization of the parameters"
    if isinstance(X, gpuarray.GPUArray):
      self.devX = X
    elif isinstance(X, np.ndarray):
      self.devX.set(X)
    else:
      self.debug(0,
          "Error! Unknown type of data given to Correl_level.setDisp")
      raise ValueError

  def save_diff(self,fname=None):
    self._makeDiff.prepared_call(self.grid, self.block,
                                 self.devOut.gpudata,
                                 self.devX.gpudata,
                                 self.devFieldsX.gpudata,
                                 self.devFieldsY.gpudata)
    diff = (self.devOut.get() + 128).astype(np.uint8)
    if fname is None:
      fname = "diff{}-{}.tiff".format(self.num, self.loop)
    cv2.imwrite(fname, diff)

  def compute(self, img_d=None):
    """ The method that actually computes the weight of the fields."""
    self.debug(3, "Calling main routine")
    self.loop += 1
    # self.mul = 3
    if not self._ready:
      self.debug(2, "Wasn't ready ! Preparing...")
      self.prepare()
    if img_d is not None:
      self.set_image(img_d)
    assert hasattr(self, 'array_d'), \
      "Did not set the image, use set_image() before calling compute \
  or give the image as parameter."
    self.debug(3, "Computing first diff table")
    self._makeDiff.prepared_call(self.grid, self.block,
                                 self.devOut.gpudata,
                                 self.devX.gpudata,
                                 self.devFieldsX.gpudata,
                                 self.devFieldsY.gpudata)
    self.res = self._leastSquare(self.devOut).get()
    self.debug(3, "res:", self.res / 1e6)

    # Iterating #
    # Note: I know this section is dense and wrappers for kernel calls could
    # have made things clearer, but function calls in python cause a
    # non-negligeable overhead and this is the critical part.
    # The comments are here to guide you !
    for i in range(self.nb_iter):
      self.debug(3, "Iteration", i)
      for i in range(self.fields_count):
        # Computing the direction of the gradient of each parameters
        self.devVec[i] = self._mulRedKrnl(self.devG[i], self.devOut)
      # Newton method: we multiply the gradient vector by the pre-inverted
      # Hessian, devVec now contains the actual research direction.
      self._dotKrnl(self.devHi, self.devVec,
                    grid=(1, 1), block=(self.fields_count, 1, 1))
      # This line simply adds k times the research direction to devX
      # with a really simple kernel (does self.devX += k*self.devVec)
      self._addKrnl.prepared_call((1, 1), (self.fields_count, 1, 1),
                                  self.devX.gpudata, self.mul,
                                  self.devVec.gpudata)
      # Do not get rid of this condition: it will not change the output but
      # the parameters will be evaluated, this will copy data from the device
      if self.verbose >= 3:
        self.debug(3, "Direction:", self.devVec.get())
        self.debug(3, "New X:", self.devX.get())

      # To get the new residual
      self._makeDiff.prepared_call(self.grid, self.block,
                                   self.devOut.gpudata,
                                   self.devX.gpudata,
                                   self.devFieldsX.gpudata,
                                   self.devFieldsY.gpudata)
      oldres = self.res
      self.res = self._leastSquare(self.devOut).get()
      # If we moved away, revert changes and stop iterating
      if self.res >= oldres:
        self.debug(3, "Diverting from the solution new res={} >= {}!"
                   .format(self.res / 1e6, oldres / 1e6))
        self._addKrnl.prepared_call((1, 1), (self.fields_count, 1, 1),
                                    self.devX.gpudata,
                                    -self.mul,
                                    self.devVec.gpudata)
        self.res = oldres
        self.debug(3, "Undone: X=", self.devX.get())
        break

      self.debug(3, "res:", self.res / 1e6)
    if self.show_diff:
      cv2.imshow("Residual", (self.devOut.get() + 128).astype(np.uint8))
      cv2.waitKey(1)
    return self.devX.get()
