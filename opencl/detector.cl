#define WORK_SIZE 512


__kernel void detect( 
  int levels, __global int * data, __global float * counts, __global float * result
) {
  uint index = get_global_id( 0 );
  uint level = data[index];

  float sum = 0;
  for ( int l = 0; l < levels; ++l ) {
    sum += counts[index * levels + l];
  }

  float pBg = counts[index * levels + level] / sum;
  float pFg = 1 - pBg;
  /*if ( pFg < 0.90 ) {*/
    /*pFg = 0.0;*/
  /*} else {*/
    /*pFg = 1.0;*/
  /*}*/
  result[index] = pFg;
  counts[index * levels + level] += pBg;
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


