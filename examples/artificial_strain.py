import numpy as np
import cv2
from gpucorrel import fields, GPUCorrel

h,w = 1080,1920

# Generating a random image and upscaling it to lower spatial frequency
# Not the best speckle but cheap and portable
np.random.seed(1)
imb = cv2.resize(np.random.random((h//10,w//10)),(w,h),cv2.INTER_LANCZOS4)

# The fields we will be using for this test
f = fields.get_fields(['x','y','r','exx','eyy','exy2'],h,w)
x,y,r,exx,eyy,exy2 = [f[:,:,:,i] for i in range(6)]


vec = [.3,2,.02,1,.5,-.2] # We will artificailly apply this transformation
transfo = sum([v*f for v,f in zip(vec,[x,y,r,exx,eyy,exy2])])

xx,yy = np.meshgrid(range(w),range(h))

# Creating the reference image given the deformed image (imb)
# and the displacement field (transfo) with remap
ima = cv2.remap(imb.astype(np.float32),
    (xx+transfo[:,:,0]).astype(np.float32),
    (yy+transfo[:,:,1]).astype(np.float32),cv2.INTER_CUBIC)

# Instanciating the IDIC class, verbose=3 is highly discouraged
# when performance matters !
correl = GPUCorrel((h,w),fields=[x,y,r,exx,eyy,exy2],verbose=3)
# Setting the refrence image
correl.set_ref(ima)
# Preparing and computing
r = correl.compute(imb)
correl.clean()

print("Applied transformation:",vec)
print("Computed transformation:",r)
print("Error:",["{:.2f}%".format(abs(100*(v-i)/v)) for i,v in zip(r,vec)])
