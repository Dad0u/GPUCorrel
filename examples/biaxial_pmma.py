from glob import glob
import cv2
from time import time

from gpucorrel import GPUCorrel

"""
In order to run this test, the test images must be placed in the folder
img/ alongside this file

To download the images please visit https://doi.org/10.5281/zenodo.4419510

It should be noted that in this example, image loading and processing is
serialized, meaning that performance could be severely impacted by the IO and
CPU of the machine.

For better performance, consider using a dedicated process to load the images.
"""

img_list = sorted(glob('img/img_*.tiff'))
if len(img_list) != 438:
  print("""Could not find the images
To run the test, please download the images
and place them in a folder named img""")

print("Reading reference image...",end='',flush=True)
ref = cv2.imread('img/img_ref_0.00102.tiff',0)
print(" Ok")

print("Preparing...",end='',flush=True)
dic = GPUCorrel(ref.shape,fields=['x','y','r','exx','eyy','exy2'])
dic.set_ref(ref)
dic.prepare()
print(" Ok")

with open("results.csv","w") as f:
  t0 = time()
  for i,img_name in enumerate(img_list,1):
    print(f"Processing image {i}/{len(img_list)}")
    img = cv2.imread(img_name,0)
    r = dic.compute(img)
    f.write(",".join([str(i) for i in r])+'\n')
  t1 = time()

print(f"Processed {len(img_list)} image in {t1-t0:.2f} seconds")
print(f"Average: {len(img_list)/(t1-t0):.2f} fps at resolution {ref.shape}")
dic.clean()
