import pyximport; pyximport.install()
from scipy.cluster.vq import kmeans, vq
import k_means_cpu
import k_means_gpu
import numpy as np
from tick_tack import *

def generate_data( num_points, num_clusters, dimensions ):
  data  = np.zeros( ( num_points, dimensions ), np.float32 )
  centroids = np.zeros( ( num_clusters, dimensions ), np.float32 )

  for i in xrange( num_clusters ):
    centroids[i] = np.random.uniform( -10, 10, dimensions )
   
  for i in xrange( num_points ):
    data[i] = centroids[np.random.randint( num_clusters )] + np.random.normal( size = dimensions )
  return data, centroids

def kmeans_cpu( data, clusters ):
  assignments = k_means_cpu.expectation( data, clusters )
  old_distortion = None
  distortion  = k_means_cpu.distortion( data, clusters, assignments )
  while old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4:
    t_clusters = k_means_cpu.maximization( data, assignments, num_clusters )
    assignments = k_means_cpu.expectation( data, t_clusters )
    old_distortion = distortion
    distortion = k_means_cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion

def kmeans_gpu( data, clusters ):
  assignments = k_means_gpu.expectation( data, clusters )
  old_distortion = None
  distortion  = k_means_cpu.distortion( data, clusters, assignments )
  while old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4:
    t_clusters = k_means_cpu.maximization( data, assignments, num_clusters )
    assignments = k_means_gpu.expectation( data, t_clusters )
    old_distortion = distortion
    distortion = k_means_cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion

num_points = 307200
num_clusters = 16
dimensions = 12

data, _ = generate_data( num_points, num_clusters, dimensions )

clusters = k_means_cpu.init_clusters( data, dimensions )

tick( "CPU" )
_, distortion = kmeans_cpu( data, clusters )
tack( "CPU" )
print "CPU", distortion

tick( "GPU" )
_, distortion = kmeans_gpu( data, clusters )
tack( "GPU" )
print "GPU", distortion

tick( "Scipy" )
_, distortion = kmeans( data, clusters )
tack( "Scipy" )
print "Scipy", distortion

for s in stats( "CPU" ) + stats( "GPU" ) + stats( "Scipy" ):
  print s

