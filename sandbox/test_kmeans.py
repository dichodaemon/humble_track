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

def cpu( data, clusters ):
  tick( "Expectation" )
  assignments = k_means_cpu.expectation( data, clusters )
  tack( "Expectation" )
  old_distortion = None
  tick( "Distortion" )
  distortion  = k_means_cpu.distortion( data, clusters, assignments )
  tack( "Distortion" )
  while True:
    tick( "Maximization" )
    t_clusters = k_means_cpu.maximization( data, assignments, num_clusters )
    tack( "Maximization" )
    tick( "Expectation" )
    assignments = k_means_cpu.expectation( data, t_clusters )
    tack( "Expectation" )
    tick( "Distortion" )
    distortion = k_means_cpu.distortion( data, t_clusters, assignments )
    tack( "Distortion" )
    if old_distortion != None and np.abs( old_distortion - distortion ) / old_distortion < 1E-4:
      break
    old_distortion = distortion
  return t_clusters, distortion

def gpu( data, clusters ):
  tick( "gExpectation" )
  assignments = k_means_gpu.expectation( data, clusters )
  tack( "gExpectation" )
  old_distortion = None
  tick( "gDistortion" )
  distortion  = k_means_cpu.distortion( data, clusters, assignments )
  tack( "gDistortion" )
  count = 0
  while True:
    print "GPU---"
    tick( "gMaximization" )
    t_clusters = k_means_cpu.maximization( data, assignments, num_clusters )
    tack( "gMaximization" )
    tick( "gExpectation" )
    assignments = k_means_gpu.expectation( data, t_clusters )
    tack( "gExpectation" )
    tick( "gDistortion" )
    distortion = k_means_cpu.distortion( data, t_clusters, assignments )
    tack( "gDistortion" )
    if count == 0:
      break
    if old_distortion != None and np.abs( old_distortion - distortion ) / old_distortion < 1E-4:
      break
    old_distortion = distortion
    count += 1
  return t_clusters, distortion

num_points = 307200
num_clusters = 16
dimensions = 12

data, _ = generate_data( num_points, num_clusters, dimensions )

clusters = k_means_cpu.init_clusters( data, dimensions )

tick( "CPU" )
_, distortion = cpu( data, clusters )
tack( "CPU" )
print "CPU", distortion

tick( "GPU" )
_, distortion = gpu( data, clusters )
tack( "GPU" )
print "GPU", distortion

tick( "Scipy" )
_, distortion = kmeans( data, clusters )
tack( "Scipy" )
print "Scipy", distortion

for s in stats( "CPU" ) + stats( "GPU" ) + stats( "Scipy" ):
  print s

