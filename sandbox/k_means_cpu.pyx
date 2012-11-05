#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np
import numpy as np

def init_clusters( np.ndarray[np.float32_t, ndim=2] data, int k, int samples = 3 ):
  cdef int i, j
  cdef np.ndarray[np.float32_t, ndim=2] centroids = np.zeros( ( k, data.shape[1] ), np.float32 )
  for i in xrange( k ):
    for j in xrange( samples ):
      centroids[i] += data[np.random.randint( data.shape[0] )]
    centroids /= ( samples * 1. )
  return centroids

def expectation( 
  np.ndarray[np.float32_t, ndim=2] data, 
  np.ndarray[np.float32_t, ndim=2] centroids
) :
  cdef int d0 = data.shape[0]
  cdef int d1 = data.shape[1]
  cdef int k  = centroids.shape[0]
  cdef float distance = 0
  cdef float min_distance
  cdef int i, j, ii
  cdef np.ndarray[np.int32_t, ndim = 1] result = np.ones( ( d0, ), np.int32 )

  for i in xrange( d0 ):
    min_distance = 1E6
    for j in xrange( k ):
      distance = 0
      for ii in xrange( d1 ):
        distance += ( data[i, ii] - centroids[j, ii] ) ** 2
      if distance < min_distance:
        min_distance = distance
        result[i] = j
  return result

def distortion( np.ndarray[np.float32_t, ndim=2] data, np.ndarray[np.float32_t, ndim=2] centroids, np.ndarray[np.int32_t, ndim =  1] assignments):
  cdef int d0 = data.shape[0]
  cdef int d1 = data.shape[1]
  cdef float distance = 0
  cdef float distortion = 0
  cdef int i, j, k

  for i in xrange( d0 ):
    k = assignments[i]
    distance = 0
    for j in xrange( d1 ):
      distance += ( data[i, j] - centroids[k, j] ) ** 2
    distortion += distance
  return ( distortion / d0 ) ** 0.5

def maximization( np.ndarray[np.float32_t, ndim=2] data, np.ndarray[np.int32_t, ndim =  1] assignments, int k ):
  cdef int d0 = data.shape[0]
  cdef int d1 = data.shape[1]
  cdef np.ndarray[np.float32_t, ndim=1] counts = np.zeros( ( k, ), np.float32 )
  cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros( ( k, d1 ), np.float32 )
  cdef int i, j, ii

  for i in xrange( d0 ):
    ii = assignments[i]
    counts[ii] += 1
    for j in xrange( d1 ):
      result[ii, j] += data[i, j]

  for i in xrange( k ):
    for j in xrange( d1 ):
      result[i, j] /= counts[i]
  return result

    

