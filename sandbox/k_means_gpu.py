import pyopencl as cl
import numpy as np
from tick_tack import *

import util

class _expectation( util.Program ):
  def __init__( self ):
    util.Program.__init__( self )
    self.kmeans = self.loadProgram( "kmeans.cl" )

  def __call__( self, data, centroids ):
    tick( "Preparation" )
    mf = cl.mem_flags

    assignments = np.zeros( ( data.shape[0], ), np.int32 )
    assignments_b = cl.Buffer( self.context, mf.WRITE_ONLY, assignments.nbytes )
    data_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = data )
    centroids_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = centroids )
    tack( "Preparation" )
    tick( "Execution" )

    self.kmeans.expectation(
      self.queue, ( data.shape[0], ), None,
      np.int32( centroids.shape[1] ), np.int32( centroids.shape[0] ),
      data_b, centroids_b, assignments_b 
    )

    cl.enqueue_copy( self.queue, assignments, assignments_b )
    tack( "Execution" )
    return assignments


expectation = _expectation()

