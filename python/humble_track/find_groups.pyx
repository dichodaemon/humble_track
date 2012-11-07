#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np



def find_groups( np.ndarray[np.int32_t, ndim=2] image ):
  cdef int big = 50000
  cdef int d0 = image.shape[0]
  cdef int d1 = image.shape[1]
  cdef np.ndarray[np.int32_t, ndim=2] groups = np.zeros( ( d0, d1 ), np.int32 )
  cdef np.ndarray[np.int32_t, ndim=1] parent = np.ones( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] index  = np.ones( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] counts = np.zeros( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] minx = np.ones( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] maxx = np.ones( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] miny = np.ones( ( big, ), np.int32 ) * -1
  cdef np.ndarray[np.int32_t, ndim=1] maxy = np.ones( ( big, ), np.int32 ) * -1

  cdef int i, j
  cdef int changed
  cdef int fg
  cdef int p1, p2
  cdef int g1, g2
  cdef int group_count = 0
  cdef int group

  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      fg = image[i, j]
      p1 = big
      p2 = big
      if i > 0 and image[i - 1, j] == fg:
        g1 = groups[i - 1, j]
        p1 = parent[g1]
      if j > 0 and image[i, j - 1] == fg:
        g2 = groups[i, j - 1]
        p2 = parent[g2]

      if p1 == big and p2 == big:
        groups[i, j] = group_count
        parent[group_count] = group_count
        group_count += 1
      elif p1 < p2:
        groups[i, j] = p1
        if p2 < big:
          parent[g2] = p1
      else:
        groups[i, j] = p2
        if p1 < big:
          parent[g1] = p2

  changed = 1

  while changed == 1:
    changed = 0
    for i in xrange( image.shape[0] ):
      for j in xrange( image.shape[1] ):
        fg = image[i, j]
        p1 = big
        p2 = big
        if i > 0 and image[i - 1, j] == fg:
          g1 = groups[i - 1, j]
          p1 = parent[g1]
        if j > 0 and image[i, j - 1] == fg:
          g2 = groups[i, j - 1]
          p2 = parent[g2]

        if p1 < p2 and p2 < big:
          changed = 1
          groups[i, j] = p1
          parent[g2] = p1
        elif p2 < p1 and p1 < big:
          changed = 1
          groups[i, j] = p2
          parent[g1] = p2

    for i in xrange( image.shape[0] - 1, -1, -1 ):
      for j in xrange( image.shape[1] - 1, -1, -1 ):
        fg = image[i, j]
        p1 = big
        p2 = big
        if i < image.shape[0] - 1 and image[i + 1, j] == fg:
          g1 = groups[i + 1, j]
          p1 = parent[g1]
        if j < image.shape[1] - 1 and image[i, j + 1] == fg:
          g2 = groups[i, j + 1]
          p2 = parent[g2]

        if p1 < p2 and p2 < big:
          changed = 1
          groups[i, j] = p1
          parent[g2] = p1
        elif p2 < p1 and p1 < big:
          changed = 1
          groups[i, j] = p2
          parent[g1] = p2

  group_count = 0
  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      fg = parent[groups[i, j]]
      if index[fg] == -1:
        index[fg] = group_count
        group_count += 1
      group = index[fg]
      groups[i, j] = group
      counts[group] += 1
      if maxx[group] == -1 or i > maxx[group]:
        maxx[group] = i
      if maxy[group] == -1 or j > maxy[group]:
        maxy[group] = j
      if minx[group] == -1 or i < minx[group]:
        minx[group] = i
      if miny[group] == -1 or j < miny[group]:
        miny[group] = j
  
  return group_count, groups, counts, maxx - minx * maxy - miny




      
      
