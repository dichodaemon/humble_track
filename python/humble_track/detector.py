import cv
from util import *
from compute_texcels import *
from compute_histograms import *
from find_groups import *
import numpy as np
from scipy.cluster.vq import kmeans, vq
import logging
logging.basicConfig( format = "%(asctime)-15s - %(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger( __name__ )
logger.setLevel( logging.DEBUG )

class Detector( object ):

  def __init__( 
    self, 
    num_texcels = 16, num_histograms = 7, window_size = 5, 
    w1 = 0.5, w2 = 1.0, w3 = 0.5
   ) :
    self.num_texcels = num_texcels
    self.num_histograms = num_histograms
    self.window_size = window_size
    self.w1 = w1
    self.w2 = w2
    self.w3 = w3

  def learn( self, frame ):
    logger.debug( "Computing Texcels" )
    texcels = compute_texcels( frame, self.w1, self.w2, self.w3 )
    logger.debug( "Computing Texcel codebook" )
    self.tc_codebook, _ = kmeans( flatten( texcels ), self.num_texcels )
    tc_image, _ = vq( flatten( texcels ), self.tc_codebook )
    tc_image = tc_image.astype( np.float32 )
    tc_image = tc_image.reshape( frame.shape[:2] )

    logger.debug( "Computing Histograms" )
    histograms = compute_histograms( tc_image, self.num_texcels, self.window_size )
    logger.debug( "Computing Histogram codebook" )
    self.hg_codebook, _ = kmeans( flatten( histograms ), self.num_histograms )

  def segment( self, frame ):
    logger.debug( "Computing Texcels" )
    texcels = compute_texcels( frame, self.w1, self.w2, self.w3 )
    logger.debug( "Building texcel image" )
    tc_image, _ = vq( flatten( texcels ), self.tc_codebook )
    tc_image = tc_image.astype( np.float32 )
    tc_image = tc_image.reshape( frame.shape[:2] )
    
    logger.debug( "Computing Histograms" )
    histograms = compute_histograms( tc_image, self.num_texcels, self.window_size )
    logger.debug( "Building histogram image" )
    hg_image, _ = vq( flatten( histograms ), self.hg_codebook )
    hg_image = hg_image.astype( np.int32 )
    hg_image = hg_image.reshape( frame.shape[:2] )
    total, groups, counts = find_groups( hg_image )
    print total
    return hg_image

