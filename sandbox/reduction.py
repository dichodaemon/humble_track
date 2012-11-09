import pyopencl as cl
import numpy as np
from tick_tack import *

import util

size = 300000 * 12
g_size = 250
reps   = 10

class reduction( util.Program ):
  def __init__( self ):
    util.Program.__init__( self )
    self.reduction = self.loadProgram( "reduction.cl" )
    device = self.context.devices[0]
    print device
    print "Local memory", device.get_info( cl.device_info.LOCAL_MEM_SIZE )

  def v1( self ):
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size / g_size, ), np.int32 )
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
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size / g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v2(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( reps ):
      s1 = self.reduction.v2(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v3( self ):
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size / g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v3(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( reps ):
      s1 = self.reduction.v3(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v4( self ):
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size / g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v4(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( reps ):
      s1 = self.reduction.v4(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v4( self ):
    mf = cl.mem_flags

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( size / g_size, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.WRITE_ONLY, result.nbytes )

    s1 = self.reduction.v4(
      self.queue, ( size, ), ( g_size, ), values_b, result_b
    )

    for i in xrange( reps ):
      s1 = self.reduction.v4(
        self.queue, ( size, ), ( g_size, ), values_b, result_b,
        wait_for = [s1]
      )

    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

  def v5( self ):
    mf = cl.mem_flags
    ngroups = size / g_size 

    values = np.ones( ( size, ), np.int32 )
    values_b = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values )
    result = np.zeros( ( ngroups, ), np.int32 )
    result_b = cl.Buffer( self.context, mf.READ_WRITE, result.nbytes )

    s1 = self.reduction.v5a(
      self.queue, ( size, ), ( g_size, ), np.int32( size ), values_b, result_b
    )

    tsize = size
    while True:
      if tsize % g_size == 0:
        tsize = tsize / g_size
      else:
        tsize = tsize / g_size + 1

      nsize = tsize
      if nsize % g_size != 0:
        nsize = tsize + g_size - tsize % g_size 

      s1 = self.reduction.v5b( 
        self.queue, ( nsize, ), ( g_size, ), np.int32( tsize ), result_b, wait_for = [s1] 
      )
      if tsize == 1:
        break
    for i in xrange( reps ):
      s1 = self.reduction.v5a(
        self.queue, ( size, ), ( g_size, ), np.int32( size ), values_b, result_b, wait_for = [s1] 
      )

      tsize = size
      while True:
        if tsize % g_size == 0:
          tsize = tsize / g_size
        else:
          tsize = tsize / g_size + 1

        nsize = tsize
        if nsize % g_size != 0:
          nsize = tsize + g_size - tsize % g_size 

        s1 = self.reduction.v5b( 
          self.queue, ( nsize, ), ( g_size, ), np.int32( tsize ), result_b, wait_for = [s1] 
        )
        if tsize == 1:
          break
    cl.enqueue_copy( self.queue, result, result_b )
    return result[0]

def test( r, version, count ):
  value = 0
  s = "v%i" % version
  tick( s )
  value = getattr( r, s )()
  tack( s )
  print "-" * 80
  print value
  for s in stats( s ):
    print s


r = reduction()
value = 0

for i in xrange( 2, 6 ):
  test( r, i, 100 )

