#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np



def foreground_groups( 
  np.ndarray[np.int32_t, ndim=2] foreground, 
  np.ndarray[np.int32_t, ndim=2] groups, 
  np.ndarray[np.int32_t, ndim=1] group_size, 
  int num_groups 
):
  cdef int d0 = foreground.shape[0]
  cdef int d1 = foreground.shape[1]
  cdef np.ndarray[np.int32_t, ndim=1] fg_groups = np.zeros( ( num_groups,), np.int32 )
  cdef np.ndarray[np.int32_t, ndim=2] mask   = np.zeros( ( d0, d1 ), np.int32 )
  cdef np.ndarray[np.int32_t, ndim=1] votes  = np.zeros( ( num_groups,), np.int32 )

  cdef int i, j

  for i in xrange( d0 ):
    for j in xrange( d1 ):
      if foreground[i, j] == 1:
        votes[groups[i, j]] += 1
  
  for i in xrange( num_groups ):
    if group_size[i] > 10 and 1.0 * votes[i] / group_size[i] > 0.6:
      fg_groups[i] = 1


  for i in xrange( d0 ):
    for j in xrange( d1 ):
      if fg_groups[groups[i, j]] == 1:
        mask[i, j] = 1

  return fg_groups, mask


      
      
