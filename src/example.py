from gpucorrel import GPUCorrel
import cv2
import numpy as np
from time import time

# Opening the first camera with openCV
cam = cv2.VideoCapture(0)

# Reading a frame to get the resolution
r,f = cam.read()
# Raise an error if the camera could not be read
assert r,"Could not read camera!"

height,width = f.shape[0:2]
print("Camera resolution : {}x{}".format(width,height))
if f.ndim == 3:
  print("This is a color image source")
  # A fonction to convert color images to grayscale
  convert = lambda img: np.mean(img,axis=2,dtype=np.float32)
elif f.ndim == 2:
  print("This is a grayscale image source")
  # GPUCorrel only works with float32. If the image or fields are any other
  # type, the conversion will be performed implicitly with a warning
  convert = lambda x:x.astype(np.float32)
else:
  raise IOError("Image data not understood")

# Instanciating the class
# We will be computing only rigid body motions
# x and y are the displacement along each direction,
# r is the rotation (linearized so only for small angles)
correl = GPUCorrel((height,width),fields=['x','y','r'])

# We read a few images to let the auto exposure ajust the brightness
for i in range(5):
  r,f = cam.read()
# We set the last one as the reference image
correl.setOrig(convert(f))


def read_image():
  r,f = cam.read()
  assert r,"Error reading the camera"
  return convert(f)


t1 = time()
try:
  while True:
    # Reading a new image
    f = read_image()
    # Compute and print the displacement
    print("x= {:.2f}\ny= {:.2f}\nr= {:.2f}".format(*correl.getDisp(f)))
    t0 = t1
    t1 = time()
    print("{:.3f}ms/loop ({:.2f} fps)\n".format(1000*(t1-t0),1/(t1-t0)))
except KeyboardInterrupt:
  print("Exiting...")
# Don't forget to clean up before exiting
correl.clean()
cam.release()
