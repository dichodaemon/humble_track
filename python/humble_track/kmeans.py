import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "..", ".." ) )
sys.path.append( os.path.abspath( os.path.join( BASE_DIR, "python" ) ) )
sys.path.append( os.path.abspath( os.path.join( BASE_DIR, "build" ) ) )

import kmeans_cpu as cpu
import kmeans_gpu as gpu
import numpy as np
from tick_tack import *

init_clusters = cpu.init_clusters

def kmeans_cpu( data, clusters, iterations = None ):
  assignments = cpu.expectation( data, clusters )
  old_distortion = None
  if iterations == None:
    distortion  = cpu.distortion( data, clusters, assignments )
  count = 0
  while ( old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4 )\
      and ( iterations == None or count < iterations ):
    t_clusters = cpu.maximization( data, assignments, clusters.shape[0] )
    assignments = cpu.expectation( data, t_clusters )
    if iterations == None:
      old_distortion = distortion
      distortion = cpu.distortion( data, t_clusters, assignments )
    count += 1
  if iterations != None:
    distortion = cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion

def kmeans_gpu( data, clusters, iterations = None ):
  assignments = gpu.expectation( data, clusters )
  old_distortion = None
  if iterations == None:
    distortion  = cpu.distortion( data, clusters, assignments )
  count = 0
  while ( old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4 )\
      and ( iterations == None or count < iterations ):
    t_clusters = cpu.maximization( data, assignments, clusters.shape[0] )
    assignments = gpu.expectation( data, t_clusters )
    if iterations == None:
      old_distortion = distortion
      distortion = cpu.distortion( data, t_clusters, assignments )
    count += 1
  if iterations != None:
    distortion = cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion


if __name__ == "__main__":

  def generate_data( num_points, num_clusters, dimensions ):
    data  = np.zeros( ( num_points, dimensions ), np.float32 )
    centroids = np.zeros( ( num_clusters, dimensions ), np.float32 )

    for i in xrange( num_clusters ):
      centroids[i] = np.random.uniform( -10, 10, dimensions )
     
    for i in xrange( num_points ):
      data[i] = centroids[np.random.randint( num_clusters )] + np.random.normal( size = dimensions )
    return data, centroids

  num_points = 307200
  num_clusters = 16
  dimensions = 12

  data, _ = generate_data( num_points, num_clusters, dimensions )

  clusters = init_clusters( data, dimensions )

  tick( "CPU" )
  _, distortion = kmeans_cpu( data, clusters, 10 )
  tack( "CPU" )
  print "CPU", distortion

  tick( "GPU" )
  _, distortion = kmeans_gpu( data, clusters, 10 )
  tack( "GPU" )
  print "GPU", distortion

  tick( "FULL_GPU" )
  _, distortion = gpu.kmeans( data, clusters )
  tack( "FULL_GPU" )
  print "FULL_GPU", distortion

  for s in stats( "CPU" ) + stats( "GPU" ) + stats( "FULL_GPU" ):
    print s

