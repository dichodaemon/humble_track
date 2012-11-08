#define GROUP_SIZE 512

void computeExpectation( 
  int index, int dim, int k,
  __global float * data, __global float * centroids, __global int * assignments
) {
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

__kernel void expectation( 
  int dim, int k, 
  __global float * data, __global float * centroids, __global int * assignments,
  __global float * buffer
) {
  unsigned int index = get_global_id( 0 );
  computeExpectation( index, dim, k, data, centroids, assignments );
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


  computeExpectation( id, dim, k, data, centroids, assignments );

  barrier( CLK_LOCAL_MEM_FENCE );

  __local float workArray[GROUP_SIZE];
  __local uint countArray[GROUP_SIZE];

  for ( uint c = 0; c < k; ++c ) {
    uint count = 0;
    if ( assignments[id] == c ) {
      countArray[lId] = 1;
      for ( uint d = 0; d < dim; ++d ) {
        barrier( CLK_LOCAL_MEM_FENCE );
        workArray[lId] = data[id * dim + d];
        for ( uint stride = WORK_SIZE / 2; stride > 0; stride /= 2 ) {
          barrier( CLK_LOCAL_MEM_FENCE );
          if ( lId < stride && lId + stride < gSize ) {
            workArray[lId] += workArray[lId + stride];
          }
        }
    
        if ( lId == 0 ) {
          buffer[gId * dim + d] = workArray[0];
        }
        barrier( CLK_LOCAL_MEM_FENCE );
    
      }
    } else {
      countArray[lId] = 0;
    }
    if ( id == 0 ) {
      for ( int i = 0; i < nGroups; ++i ) {
        centroids[c * dim + d] += buffer[i * dim + c];
      }
    }
  }
}
