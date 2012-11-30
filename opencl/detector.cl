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
  /*if ( pFg < 0.98 ) {*/
    /*pFg = 0.0;*/
  /*} else {*/
    /*pFg = 1.0;*/
  /*}*/

  result[index] = pFg;
  counts[index * levels + level] += max( pBg, 0.1 );
}

__kernel void detect1( 
  int levels, __global int * data, __global float * counts, __global float * result
) {
  uint index = get_global_id( 0 );
  uint level = data[index];

  float sum = 0;
  float val = 0;
  float alpha = 0.999;
  for ( int l = 0; l < levels; ++l ) {
    float v = counts[index * levels + l];
    sum += v;
    if ( l == level ) {
      val = v;
    }
    counts[index * levels + l] *= alpha;
  }

  float pBg = val / sum;
  float pFg = 1 - pBg;
  result[index] = pFg;
  /*counts[index * levels + level] = val * alpha + ( 1 - alpha );*/
  counts[index * levels + level] += ( 1 - alpha );
}

