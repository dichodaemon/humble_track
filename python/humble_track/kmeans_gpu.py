import pyopencl as cl
import numpy as np

import opencl

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

  def __call__( self, data, centroids ):
    mf = cl.mem_flags

    for group_size in xrange( 513, -1, -1 ):
      if data.shape[0] % group_size == 0:
        break

    assignments = np.zeros( ( data.shape[0], ), np.int32 )
    assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, assignments.nbytes )
    data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    centroids_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = centroids )
    buffer_b = cl.Buffer( self.context, mf.READ_WRITE, 4 * data.shape[0] / num_groups )
    

    self.kmeans.kmeans(
      self.queue, ( data.shape[0], ), (group_size, ),
      np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
      data_b, centroids_b, assignments_b, buffer_b
    )

    cl.enqueue_copy( self.queue, assignments, assignments_b )
    return assignments

kmeans = kmeans()
expectation = _expectation()

