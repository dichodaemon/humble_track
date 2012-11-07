import pyopencl as cl
import numpy as np
from tick_tack import *

import util

class reduction( util.Program ):
  def __init__( self ):
    util.Program.__init__( self )
    self.reduction = self.loadProgram( "reduction.cl" )
    device = self.context.devices[0]
    print device
    print "Local memory", device.get_info( cl.device_info.LOCAL_MEM_SIZE )

  def v1( self ):
    size = 300000
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v1(
      self.queue, ( values.shape[0], ), None, values_b, result_b
    )

    for i in xrange( 10 ):
      s1 = self.reduction.v1(
        self.queue, ( values.shape[0], ), None, values_b, result_b,
        wait_for = [s1]
      )


    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v2( self ):
    size = 300000
    g_size = 500
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v2(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( 100 ):
      s1 = self.reduction.v2(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )
    s1.wait()

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v3( self ):
    size = 300000
    g_size = 500
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v3(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( 100 ):
      s1 = self.reduction.v3(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )
    s1.wait()

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

def test( r, version, count ):
  value = 0
  s = "v%i" % version
  tick( s )
  for i in xrange( count ):
    value = getattr( r, s )()
  tack( s )
  print "-" * 80
  print value
  for s in stats( s ):
    print s


r = reduction()
count = 100
value = 0

for i in xrange( 2, 4 ):
  test( r, i, 100 )

