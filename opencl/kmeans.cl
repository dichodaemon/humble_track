#define WORK_SIZE 256

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
  __global float * data, __global float * centroids, __global int * assignments
) {
  unsigned int index = get_global_id( 0 );
  computeExpectation( index, dim, k, data, centroids, assignments );
}



void reduceFloatArray( uint groupId, uint localId, uint groupSize, __local float * workArray, __global float * buffer ) 
{
  for ( uint stride = WORK_SIZE / 2; stride > 0; stride /= 2 ) {
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( localId < stride && localId + stride < groupSize ) {
      workArray[localId] += workArray[localId + stride];
    }
  }

  if ( localId == 0 ) {
    buffer[groupId] = workArray[0];
  }
  barrier( CLK_LOCAL_MEM_FENCE );
}

__kernel void reduceFloat( 
  int size, __global int * buffer
) {
  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local float workArray[WORK_SIZE];

  if ( id < size ) {
    workArray[lId] = buffer[gId * gSize + lId];
  } else {
    workArray[lId] = 0;
  }
  reduceFloatArray( gId, lId, gSize, workArray, buffer );
}

__kernel void kmeans1(
  int dim, int k, 
  __global float * data, __global float * centroids, __global int * assignments
) {
  uint id = get_global_id( 0 );
  computeExpectation( id, dim, k, data, centroids, assignments );
}


__kernel void kmeans2(
  int size, int dim, int k, int d, int c, 
  __global float * data, __global float * centroids, __global int * assignments,
  __global float * buffer
) {
  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local float workArray[WORK_SIZE];

  uint index = id * dim + d;

  if ( id < size && assignments[index] == c ) {
    workArray[lId] = data[index];
  } else {
    workArray[lId] = 0;
  }
  reduceFloatArray( gId, lId, gSize, workArray, buffer );
}

