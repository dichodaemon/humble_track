#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

def find_groups( np.ndarray[np.int32_t, ndim=2] image ):
  cdef int d0 = image.shape[0]
  cdef int d1 = image.shape[1]
  cdef np.ndarray[np.int32_t, ndim=2] groups = np.zeros( ( d0, d1 ), np.int32 )
  cdef np.ndarray[np.int32_t, ndim=1] parent = np.ones( ( 100000, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] index  = np.ones( ( 100000, ), np.int32 ) * -1

  cdef int next_group = 0
  cdef int i, j
  cdef float current
  cdef int group, tgroup

  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      current = image[i, j]
      group = next_group
      if i > 0 and image[i - 1, j] == current:
        group = groups[i - 1, j]
      if j > 0 and image[i, j - 1] == current:
        tgroup = groups[i, j - 1]
        if tgroup < group:
          if parent[group] == -1:
            parent[group] == tgroup
          group = tgroup
        else:
          if parent[tgroup] == -1:
            parent[tgroup] = group
      groups[i, j] = group
      if group == next_group:
        parent[next_group] = next_group
        next_group += 1
  print "groups", next_group    


      
      
