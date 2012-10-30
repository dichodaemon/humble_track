#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

def compute_texcels( np.ndarray[np.uint8_t, ndim=3] frame, float w1, float w2, float w3 ):
  cdef int d0 = frame.shape[0]
  cdef int d1 = frame.shape[1]
  cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros( ( d0, d1, 11 ), np.float32 )
  cdef int i, j
  cdef float l1
  for i in xrange( 1, frame.shape[0] - 2 ):
    for j in xrange( 1, frame.shape[1] - 2 ):
      result[i, j, 0]  = w1 * frame[i, j, 0]
      result[i, j, 1]  = w2 * frame[i, j, 1]
      result[i, j, 2]  = w2 * frame[i, j, 2]

      l1 = frame[i, j, 0] * 1.
      result[i, j, 3]  = w3 * ( l1 - frame[i - 1, j - 1, 0] )
      result[i, j, 4]  = w3 * ( l1 - frame[i    , j - 1, 0] )
      result[i, j, 5]  = w3 * ( l1 - frame[i + 1, j - 1, 0] )
      result[i, j, 6]  = w3 * ( l1 - frame[i - 1, j, 0] )
      result[i, j, 7]  = w3 * ( l1 - frame[i + 1, j, 0] )
      result[i, j, 8]  = w3 * ( l1 - frame[i - 1, j + 1, 0] )
      result[i, j, 9]  = w3 * ( l1 - frame[i    , j + 1, 0] )
      result[i, j, 10] = w3 * ( l1 - frame[i + 1, j + 1, 0] )
  return result

