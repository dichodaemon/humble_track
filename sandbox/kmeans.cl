
__kernel void expectation( 
  int dim, int k, 
  __global float * data, __constant float * centroids, __global int * assignments
) {
  unsigned int index = get_global_id( 0 );

  float minDistance = 1E6;

  for ( int cluster = 0; cluster < k; ++cluster ) {
    float distance = 0;
    for ( int d = 0; d < dim; ++d ) {
      float tmp = data[index * dim + d] - centroids[cluster * dim + d];
      distance += tmp * tmp;
    }
    if ( distance < minDistance ) {
      minDistance = distance;
      assignments[index] = cluster;
    }
  }
}

