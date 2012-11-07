import kmeans_cpu as cpu
import kmeans_gpu as gpu
import numpy as np

init_clusters = cpu.init_clusters

def kmeans_cpu( data, clusters ):
  assignments = cpu.expectation( data, clusters )
  old_distortion = None
  distortion  = cpu.distortion( data, clusters, assignments )
  while old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4:
    t_clusters = cpu.maximization( data, assignments, clusters.shape[0] )
    assignments = cpu.expectation( data, t_clusters )
    old_distortion = distortion
    distortion = cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion

def kmeans_gpu( data, clusters ):
  assignments = gpu.expectation( data, clusters )
  old_distortion = None
  distortion  = cpu.distortion( data, clusters, assignments )
  while old_distortion == None or np.abs( old_distortion - distortion ) / distortion > 1E-4:
    t_clusters = cpu.maximization( data, assignments, clusters.shape[0] )
    assignments = gpu.expectation( data, t_clusters )
    old_distortion = distortion
    distortion = cpu.distortion( data, t_clusters, assignments )
  return t_clusters, distortion
