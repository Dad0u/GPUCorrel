GPUCorrel
=========

This project is an implementation of global Digital image Correlation (DIC)
using CUDA.

It takes a list of displacement fields (Numpy arrays of size (y,x,2)),
a reference image and a second image ((y,x) arrays). It will return the
projection of the displacement between the images projected on the given
list of fields.

Requirements
------------
* A Nvidia GPU compatible with CUDA (8.0 or newer)
* CUDA
* Python 3
* PyCUDA
* Numpy

Usage
-----
The class you want to use is GPUCorrel from src/gpudcorrel.py
The files fields.py and kernels.cu must be in the same directory.

```python
# Import the class from the module:
from gpucorrel import GPUCorrel

# Instanciate the class.
# Here, it will work on 640x480 images
# and identify rigid body motions
correl = GPUCorrel((480,640),fields=['x','y','r'])

# Set the reference image
# reference_image must be a 2D Numpy array of the correct shape
# dtype should be float32. If not, the array will be casted into float32
correl.set_ref(reference_image)

# Will prepare the class given the fields and
# the reference image. It may take a few seconds
# This line can be omitted as it will be called automatically if
# necessary when calling the fist .compute()
correl.prepare()

# And let's compute the displacement !
x,y,r = correl.compute(moving_image)

# x,y and r are floats. By default x and y are in pixel and r in degree
```

Here the fields 'x', 'y' and 'r' refer to pre-defined fields.
They are defined in fields.py
To name a few, there are:

* x: Builds an array where f[:,:,0] = 1 and f[:,:,1] = 0 for displacement along X
* y: you guessed it, f[:,:,0] = 0 and f[:,:,1] = 1, for displacement along Y
* r: The rotation. It is scaled so that the result can be interpreted in degrees.
Note the value is a linearization of the actual rotation,
therefore it may not work well for high-amplitude rotations !
* exx: elongation along x. Scaled to be interpreted in percent
* eyy: elongation along y
* exy2: shear strain

You can absolutely use this class with your own defined fields (See example below).
A fields must be a Numpy array with dtype float32 (or it will be cast to float32)
and shape (y,x,2), y and x being respectively the height and width of the images.
For example, f[a,b,0] should hold the displacement along x of the pixel at the line a and column b.
Please note that the algorithm is meant to perform on linearly independant fields.

```python
# This example shows how to create a simple example field
y,x = 480,640
lx = np.linspace(-1,1,x)
ly = np.linspace(-1,1,y)
# A field to identify Z-axis rigid body motion (equivalent to exx+eyy)
xv,yv = np.meshgrid(lx,ly)

my_field = np.empty((y,x,2),dtype=np.float32)
my_field[:,:,0] = xv
my_field[:,:,1] = yv

correl = GPUCorrel((y,x),fields=['x','y',my_field])
```
