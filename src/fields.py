#coding: utf-8
# Tools to generate and manipulate 2D fields
# Dimension is (h,w,2),
# the third axis contain the displacement along respectively x and y

import numpy as np


def ones(h,w):
  return np.ones((h,w),dtype=np.float32)


def zeros(h,w):
  return np.zeros((h,w),dtype=np.float32)


Z = None


def z(h,w):
  global Z
  if Z is None or Z[0].shape != (h,w):
    sh = 1/(w*w/h/h+1)**.5
    sw = w*sh/h
    Z = np.meshgrid(np.linspace(-sw, sw, w, dtype=np.float32),
                          np.linspace(-sh, sh, h, dtype=np.float32))
  return Z


def get_field(s,h,w):
  """
  Generates a field corresponding to the string s with size (h,w)

  Possible strings are :
  x y r exx eyy exy eyx exy2 r z uxx uyy uxy vxx vyy vxy
  """
  if s == 'x':
    """
    Rigid body motion along x in pixels
    """
    return ones(h,w),zeros(h,w)
  elif s =='y':
    """
    Rigid body motion along y in pixels
    """
    return zeros(h,w),ones(h,w)
  elif s == 'r':
    """
    Rotation in degrees
    """
    u,v = z(h,w)
    # Ratio (angle) of the rotation
    # Should be π/180 to be 1 for 1 deg
    # Z has and amplitude of 1 in the corners
    # 360 because h²+w² is twice the distance center-corner
    r = (h**2+w**2)**.5*np.pi/360
    return v*r,-u*r
  elif s == 'exx':
    """
    Elongation along x in %
    """
    return (np.concatenate((np.linspace(-w/200, w/200, w,
            dtype=np.float32)[np.newaxis, :],)*h, axis=0),
            zeros(h,w))
  elif s == 'eyy':
    """
    Elongation along y in %
    """
    return (zeros(h,w),
            np.concatenate((np.linspace(-h/200, h/200, h,
            dtype=np.float32)[:, np.newaxis],)*w, axis=1))
  elif s == 'exy':
    """
    "Shear" derivative of y along x in %
    """
    return (np.concatenate((np.linspace(-h/200, h/200, h,
            dtype=np.float32)[:, np.newaxis],)*w, axis=1),
            zeros(h,w))
  elif s == 'eyx':
    """
    "Shear" derivative of x along y in %
    """
    return (zeros(h,w),
            np.concatenate((np.linspace(-w/200, w/200, w,
            dtype=np.float32)[np.newaxis, :],)*h, axis=0))
  elif s == 'exy2':
    """
    Sum of the two previous definitions of shear in %
    """
    return ((w/(w*h)**.5)*np.concatenate((np.linspace(-h/200, h/200, h,
            dtype=np.float32)[:, np.newaxis],)*w, axis=1),
            (h/(w*h)**.5)*np.concatenate((np.linspace(-w/200, w/200, w,
            dtype=np.float32)[np.newaxis, :],)*h, axis=0))

  elif s == 'z':
    """
    Zoom in %
    """
    u,v = z(h,w)
    r = (h**2+w**2)**.5/200
    return u*r,v*r
  elif s == 'uxx':
    """
    ux = x²
    """
    return (np.concatenate(((np.linspace(-1, 1, w,dtype=np.float32)**2)
            [np.newaxis, :],) * h, axis=0),
            zeros(h,w))
  elif s == 'uyy':
    """
    ux = y²
    """
    return (np.concatenate(((np.linspace(-1, 1, h,dtype=np.float32)**2)
            [:, np.newaxis],) * w, axis=1),
            zeros(h,w))
  elif s == 'uxy':
    """
    ux = x*y
    """
    return (np.array([[k * j for j in np.linspace(-1, 1, w)]
            for k in np.linspace(-1,1,2)],dtype=np.float32),
            zeros(h,w))
  elif s == 'vxx':
    """
    uy = x²
    """
    return (zeros(h,w),
            np.concatenate(((np.linspace(-1, 1, 2,
            dtype=np.float32)**2)[np.newaxis,:],)*h,axis=0))
  elif s == 'vyy':
    """
    uy = y²
    """
    return (zeros(h,w),
            np.concatenate(((np.linspace(-1, 1, 2,
            dtype=np.float32)**2)[:,np.newaxis],)*w,axis=1))
  elif s == 'vxy':
    """
    uy = x*y
    """
    return (zeros(h,w),
            np.array([[k * j for j in np.linspace(-1, 1, 2)]
            for k in np.linspace(-1,1,2)],dtype=np.float32))
  else:
    print("WTF?",s)
    raise NameError("Unknown field string: "+s)


def get_fields(l,h,w):
  """
  Calls get_field for each string in l

  Returns a single numpy array of dim (h,w,2,N), N being the number of fields
  """
  r = np.empty((h,w,2,len(l)),dtype=np.float32)
  for i,s in enumerate(l):
    if isinstance(s,np.ndarray):
      r[:,:,:,i] = s
    else:
      r[:,:,0,i],r[:,:,1,i] = get_field(s,h,w)
  return r
