import numpy as np
from scipy.cluster.vq import kmeans, vq

def create_palette( colors = 16 ):
  samples = np.random.randint( 256, size = ( 1000, 3 ) )
  palette, _ = kmeans( samples, colors )
  palette[0] *= np.zeros( (3, ) )
  return palette

def colorize( image, palette ):
  result = np.zeros( [image.shape[0], image.shape[1], 3] ).astype( np.int8 )
  for k in xrange( palette.shape[0] ):
    result[image % palette.shape[0] == k] = palette[k]
  return result

def flatten( matrix ):
  return matrix.reshape( [matrix.shape[0] * matrix.shape[1], matrix.shape[2]] )

