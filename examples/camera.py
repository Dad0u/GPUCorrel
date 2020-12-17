from gpucorrel import GPUCorrel
import cv2
import numpy as np
from time import time
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None


class Plotter:
  def __init__(self,*labels):
    print("labels",labels)
    self.labels = labels
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)
    self.lines = []
    for _ in self.labels:
      self.lines.append(self.ax.plot([], [])[0])
    plt.legend(labels, bbox_to_anchor=(-0.03, 1.02, 1.06, .102), loc=3,
               ncol=len(labels), mode="expand", borderaxespad=1)
    plt.xlabel('time (s)')
    plt.grid()
    self.t0 = time()
    plt.draw()
    plt.pause(.001)

  def plot(self,*args):
    assert len(args) == len(self.labels),"Got an invalid number of args"
    t = time()-self.t0
    for l,y in zip(self.lines,args):
      l.set_xdata(np.append(l.get_xdata(),t))
      l.set_ydata(np.append(l.get_ydata(),y))
    self.ax.relim() # Update the window
    self.ax.autoscale_view(True, True, True)
    self.fig.canvas.draw() # Update the graph
    self.fig.canvas.flush_events()


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
fields= ('x','y','r')
correl = GPUCorrel((height,width),fields=list(fields))

# Create class to plot data in real time
if plt is not None:
  p = Plotter(*fields)

# We read a few images to let the auto exposure ajust the brightness
for i in range(5):
  r,f = cam.read()
# We set the last one as the reference image
correl.set_ref(convert(f))


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
    x,y,r = correl.compute(f)
    print("x= {:.2f}\ny= {:.2f}\nr= {:.2f}".format(x,y,r))
    if plt is not None:
      p.plot(x,y,r)
    t0 = t1
    t1 = time()
    print("{:.3f}ms/loop ({:.2f} fps)\n".format(1000*(t1-t0),1/(t1-t0)))
except KeyboardInterrupt:
  print("Exiting...")
# Don't forget to clean up before exiting
correl.clean()
cam.release()
