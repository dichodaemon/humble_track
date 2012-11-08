#define WORK_SIZE 512

__kernel void v1( 
  __global int * values, __global int * result
) {
  unsigned int index = get_global_id( 0 );

  result[index] = values[index] + 1;

  barrier( CLK_GLOBAL_MEM_FENCE );

  if ( index == 0 ) {
    for ( uint i = 1; i < get_global_size( 0 ); ++i ) {
      result[0] += result[i];
    }
  }
}

__kernel void v2( 
  __global int * values, __global int * result
) {

  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local int workArray[WORK_SIZE];

  workArray[lId] = values[id] + 1;


  for ( uint stride = 1; stride < WORK_SIZE; stride *= 2 ) {
    barrier( CLK_LOCAL_MEM_FENCE );
    uint index = 2 * stride * lId;
    if ( index + stride < gSize ) {
      workArray[index] += workArray[index + stride];
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

void reduce( uint groupId, uint localId, uint groupSize, __local int * workArray, __global int * result ) 
{
  for ( uint stride = WORK_SIZE / 2; stride > 0; stride /= 2 ) {
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( localId < stride && localId + stride < groupSize ) {
      workArray[localId] += workArray[localId + stride];
    }
  }

  if ( localId == 0 ) {
    result[groupId] = workArray[0];
  }
  barrier( CLK_LOCAL_MEM_FENCE );
}

__kernel void v3( 
  __global int * values, __global int * result
) {
  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local int workArray[WORK_SIZE];

  workArray[lId] = values[id] + 1;

  reduce( gId, lId, gSize, workArray, result );

  if ( id == 0 ) {
    for ( int i = 1; i < nGroups; ++i ) {
      result[0] += result[i];
    }
  }
}


__kernel void v4( 
  __global int * values, __global int * result
) {
  uint id = get_global_id( 0 );
  uint gSize = get_local_size( 0 );
  uint gId = get_group_id( 0 );
  uint lId = get_local_id( 0 );
  uint nGroups = get_num_groups( 0 );

  __local int workArray[WORK_SIZE];

  workArray[lId] = values[id] + 1;


  reduce( gId, lId, gSize, workArray, result );

  while ( gSize * gId < nGroups ) {
    if ( gSize * gId + lId < nGroups ) {
      workArray[lId] = result[gSize * gId + lId];
    } else {
      workArray[lId] = 0;
    }
    reduce( gId, lId, gSize, workArray, result );
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( nGroups <= gSize ) {
      break;
    } else {
      nGroups = nGroups / gSize + ( nGroups % gSize > 0 );
    }
  }
}
