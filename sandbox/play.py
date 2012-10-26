#!/usr/bin/python

import sys
import cv


class FileSource( object ):
  def __init__( self, filename ):
    self.capture = cv.CaptureFromFile( filename )

  def next( self ):
    result = cv.QueryFrame( self.capture )
    return result


if __name__ == "__main__":
  source = FileSource( sys.argv[1] )

  cv.NamedWindow( "Test" )

  while True:
    cv.ShowImage( "Test", source.next() )
    cv.WaitKey( 1 )

