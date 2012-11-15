#define WORK_SIZE 512


__kernel void detect( 
  int levels, __global int * data, __global int * counts, __global float * result
) {
  unsigned int index = get_global_id( 0 );

  float sum = 0;
  for ( int l = 0; l < levels; ++l ) {
    sum += counts[index * levels + l];
  }

  result[index] = 1.0 - counts[index * levels + data[index]] / sum;
  /*if ( result[index] < 0.999 ) {*/
    /*result[index] = 0;*/
  /*}*/

  barrier( CLK_LOCAL_MEM_FENCE );

  counts[index * levels + data[index]] += 1;
}

__kernel void detect1( 
  int levels, __global int * data, __global float * counts, __global float * result
) {
  uint index  = get_global_id( 0 );
  uint base   = index * levels;
  uint lindex = base + data[index];

  float sum = 0;
  float val = counts[lindex];
  for ( int l = 0; l < levels; ++l ) {
    sum += counts[base + l];
    counts[base + l] *= 0.995;
  }
  counts[lindex] += .005;
  if ( sum == 0 ) {
    val = 1;
  } else {
    val /= sum;
  }

  val = 1.0 - val;
  if ( val < 0.99 ) {
    val = 0;
  }

  result[index] = val;
}


