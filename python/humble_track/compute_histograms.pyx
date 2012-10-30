#cython: boundscheck=False
#cython: wraparound=False

from integral_image import *

import time
import numpy as np
cimport numpy as np

def integral_image_2( np.ndarray[np.float32_t, ndim=2] image, int num_texcels ):
  cdef int d0 = image.shape[0]
  cdef int d1 = image.shape[1]
  cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros( ( d0, d1, num_texcels ), np.float32 )
  cdef int i, j, k
  cdef float r
  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      for k in xrange( num_texcels ):
        r = 0
        if image[i, j] == k:
          r += 1
        if i > 0:
          r += result[i - 1, j, k]
        if j > 0:
          r += result[i, j - 1, k]
        if i > 0 and j > 0:
          r -= result[i - 1, j - 1, k]
        result[i, j, k] = r
  return result

def compute_histograms( np.ndarray[np.float32_t, ndim=2] labels, int num_texcels, int window_size ):
  cdef np.ndarray[np.float32_t, ndim=3] integrals
  cdef np.ndarray[np.float32_t, ndim=3] result
  cdef int i, j, k
  cdef int d0 = labels.shape[0]
  cdef int d1 = labels.shape[1]

  integrals = integral_image_2( labels, num_texcels )

  result = np.zeros( ( d0, d1, num_texcels ), np.float32 )

  for i in xrange( window_size, d0 - window_size - 1 ) :
    for j in xrange( window_size, d1 - window_size - 1 ) :
      for k in xrange( num_texcels ):
        result[i, j, k] = integrals[i + window_size, j + window_size, k] +\
                       integrals[i, j, k] -\
                       integrals[i + window_size, j, k] -\
                       integrals[i, j + window_size, k]
  return result
