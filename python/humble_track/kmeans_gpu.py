import pyopencl as cl
import numpy as np

import opencl

GROUP_SIZE = 512

class _expectation( opencl.Program ):
  def __init__( self ):
    opencl.Program.__init__( self )
    self.kmeans = self.loadProgram( "kmeans.cl" )
    self.data_shape = (0,0)

  def init( self, data ):
    self.data_shape = data.shape
    mf = cl.mem_flags
    self.data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    self.assignments = np.zeros( ( data.shape[0], ), np.int32 )
    self.assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, self.assignments.nbytes )
    self.group_size = GROUP_SIZE / data.shape[1]
    self.size = data.shape[0] + self.group_size - data.shape[0] % self.group_size
    self.data_shape = data.shape

  def __call__( self, data, centroids, refresh = False ):
    if refresh or data.shape != self.data_shape:
      self.init( data )
    mf = cl.mem_flags

    centroids_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = centroids )

    self.kmeans.expectation(
      self.queue, ( data.shape[0], ), None,
      np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
      self.data_b, centroids_b, self.assignments_b 
    )

    cl.enqueue_copy( self.queue, self.assignments, self.assignments_b )
    return self.assignments


class kmeans( opencl.Program ):
  def __init__( self ):
    opencl.Program.__init__( self )
    self.kmeans = self.loadProgram( "kmeans.cl" )

  def reduceFloat( self, buf, size, wait = [] ):
    while True:
      if size % GROUP_SIZE == 0:
        size = size / GROUP_SIZE
      else:
        size = size / GROUP_SIZE + 1

      nsize = size
      if nsize % GROUP_SIZE != 0:
        nsize = size + GROUP_SIZE - size % GROUP_SIZE 

      s1 = self.kmeans.reduceFloat( 
        self.queue, ( nsize, ), ( GROUP_SIZE, ), np.int32( size ), buf, wait_for = wait
      )
      wait = [s1]
      if size == 1:
        break
    return s1


  def __call__( self, data, centroids, iterations = 10 ):
    mf = cl.mem_flags

    assignments = np.zeros( ( data.shape[0], ), np.int32 )
    assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, assignments.nbytes )
    data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    centroids_b = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = centroids )
    buffer_b = cl.Buffer( self.context, mf.READ_WRITE, 4 * data.shape[0] )

    for group_size in xrange( GROUP_SIZE, -1, -1 ):
      if data.shape[0] % group_size == 0:
        break

    ngroups = data.shape[0] / group_size

    wait = []
    for i in xrange( iterations ):
      s1 = self.kmeans.kmeans1(
        self.queue, ( data.shape[0], ), (group_size, ),
        np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
        data_b, centroids_b, assignments_b,
        wait_for = wait
      )
      #for d in xrange( data.shape[1] ):
        #for c in xrange( centroids.shape[0] ):
          #wait = [s1]
          #s1 = self.kmeans.kmeans2(
            #self.queue, ( data.shape[0], ), (group_size, ),
            #np.int32( data.shape[0] ), np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
            #np.int32( d ), np.int32( c ),
            #data_b, assignments_b, buffer_b,
            #wait_for = wait
          #)
          #s1 = self.reduceFloat( buffer_b, data.shape[0], [s1] )

    cl.enqueue_copy( self.queue, assignments, assignments_b )
    cl.enqueue_copy( self.queue, centroids, centroids_b )
    return assignments, 0

kmeans = kmeans()
expectation = _expectation()

