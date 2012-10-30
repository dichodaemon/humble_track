#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np

def integral_image( np.ndarray[np.float32_t, ndim=2] image ):
  cdef np.ndarray[np.float32_t, ndim=2] result = image * 0
  cdef int i, j
  cdef float r
  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      r = image[i, j]
      if i > 0:
        r += result[i - 1, j]
      if j > 0:
        r += result[i, j - 1]
      if i > 0 and j > 0:
        r -= result[i - 1, j - 1]
      result[i, j] = r
  return result

