__kernel void detect( 
  uint width, uint height, uint cellSize, float weight, 
  __global float * data, __global float * counts, __global float * result
) {
  uint x = get_global_id( 0 );
  uint y = get_global_id( 1 );
  uint maxX = min( ( x + 1 ) * cellSize, width );
  uint maxY = min( ( y + 1 ) * cellSize, height );

  float sum = 0;
  int count = 0;
  for ( uint i = x * cellSize; i < maxX; ++i ) {
    for ( uint j = y * cellSize; j < maxY; ++j ) {
      sum += data[i * height + j]; 
      count++;
    }
  }

  sum /= count;

  if ( sum < weight ) {
    sum = 0;
  } else {
    sum = 1;
  }

  for ( uint i = x * cellSize; i < maxX; ++i ) {
    for ( uint j = y * cellSize; j < maxY; ++j ) {
      result[i * height + j] = sum;
    }
  }
}

