
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

__kernel void kmeans( 
  int dim, int k, 
  __global float * data, __global float * centroids, __global int * assignments
) {
  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local float workArray[512];

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

  for ( uint stride = 256; stride > 0; stride /= 2 ) {
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( lId < stride && lId + stride < gSize ) {
      workArray[lId] += workArray[lId + stride];
    }
  }

  if ( lId == 0 ) {
    result[gId] = workArray[0];
  }
  barrier( CLK_LOCAL_MEM_FENCE );

  if ( id == 0 ) {
    for ( int i = 1; i < nGroups; ++i ) {
      result[0] += result[i];
    }
  }
}
