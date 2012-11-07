import os
import humble_track as ht
import pyopencl as cl
import numpy as np
from math import *

class Program( object ):
  def __init__( self ):
    self.context = cl.create_some_context()
    self.queue =cl.CommandQueue( self.context )

  def loadProgram( self, filename ):
    f = open( os.path.join( ht.BASE_DIR, "opencl", filename ), 'r' )
    fstr = "".join( f.readlines() )
    program = cl.Program( self.context, fstr ).build( "-I %s/opencl/" % ht.BASE_DIR )
    return program

