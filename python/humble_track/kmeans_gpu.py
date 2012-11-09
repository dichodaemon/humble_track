import pyopencl as cl
import numpy as np

import opencl

GROUP_SIZE = 256

class _expectation( opencl.Program ):
  def __init__( self ):
    opencl.Program.__init__( self )
    self.kmeans = self.loadProgram( "kmeans.cl" )

  def __call__( self, data, centroids ):
    mf = cl.mem_flags

    assignments = np.zeros( ( data.shape[0], ), np.int32 )
    assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, assignments.nbytes )
    data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    centroids_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = centroids )

    self.kmeans.expectation(
      self.queue, ( data.shape[0], ), None,
      np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
      data_b, centroids_b, assignments_b 
    )

    cl.enqueue_copy( self.queue, assignments, assignments_b )
    return assignments


class kmeans( opencl.Program ):


  def __init__( self ):
    opencl.Program.__init__( self )
    self.kmeans = self.loadProgram( "kmeans.cl" )

  def reduceFloat( self, buf, size ):
    while True:
      if size % GROUP_SIZE == 0:
        size = size / GROUP_SIZE
      else:
        size = size / GROUP_SIZE + 1

      nsize = size
      if nsize % GROUP_SIZE != 0:
        nsize = size + GROUP_SIZE - size % GROUP_SIZE 

      s1 = self.kmeans.reduceFloat( 
        self.queue, ( nsize, ), ( GROUP_SIZE, ), np.int32( size ), result_b, wait_for = [s1] 
      )
      if size == 1:
        break


  def __call__( self, data, centroids ):
    mf = cl.mem_flags

    assignments = np.zeros( ( data.shape[0], ), np.int32 )
    assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, assignments.nbytes )
    data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    centroids_b = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = centroids )

    for group_size in xrange( GROUP_SIZE, -1, -1 ):
      if data.shape[0] % group_size == 0:
        break

    ngroups = data.shape[0] / group_size

    self.kmeans.kmeans1(
      self.queue, ( data.shape[0], ), (group_size, ),
      np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
      data_b, centroids_b, assignments_b
    )

    cl.enqueue_copy( self.queue, assignments, assignments_b )
    cl.enqueue_copy( self.queue, centroids, centroids_b )
    return assignments, 0

kmeans = kmeans()
expectation = _expectation()

