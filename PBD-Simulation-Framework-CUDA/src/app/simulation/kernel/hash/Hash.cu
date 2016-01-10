/*

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


// Calculates a 30-bit Morton code for the
// given 3D point located within the cube [0.0, 1023.0].
inline unsigned int mortonCode(float3 pos)
{
    pos.x = min(max(pos.x, 0.0f), 1023.0f);
    pos.y = min(max(pos.y, 0.0f), 1023.0f);
    pos.z = min(max(pos.z, 0.0f), 1023.0f);
    // x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    // y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    // z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    const unsigned int xx = expandBits((unsigned int)pos.x) << 2;
    const unsigned int yy = expandBits((unsigned int)pos.y) << 1;
    const unsigned int zz = expandBits((unsigned int)pos.z);
    //return xx * 4 + yy * 2 + zz;
    return xx + yy + zz;
}


__global__ void ResetCellStarts(unsigned int* cellStarts,
                                const unsigned int gridSize) {

  const unsigned int cell_id = get_global_id(0);

  if( cell_id < gridSize  ) {
    cellStarts[cell_id] = UINT_MAX;
  } 

}


__global__ void HashReorder(
                   uint2* hashIndexPairs,
                   const unsigned int width,
                   const unsigned int numberOfParticles,
                   unsigned int* cellStarts,

                   __read_only image2d_t positions4Read,
                   __write_only image2d_t positions4Write,
                   __read_only image2d_t colors4Read,
                   __write_only image2d_t colors4Write,
                   __read_only image2d_t predictedPositionRead,
                   __write_only image2d_t predictedPositionWrite) {

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = idx % textureWidth;
  const unsigned int y = idx / textureWidth;

  const int2 id = (int2)(get_global_id(0) % width, get_global_id(0) / width);

  const unsigned int global_id = get_global_id(0);

  if( global_id < numberOfParticles ) {
    
    const uint2 hashIndex = hashIndexPairs[global_id];

    // Compute cell starts
    if( global_id == 0 ) {
      cellStarts[hashIndex.x] = 0; 
    } else {
      const uint2 hashIndexPrevious = hashIndexPairs[global_id-1];
      if( hashIndexPrevious.x != hashIndex.x ) {
        cellStarts[hashIndex.x] = global_id;
      }
    }

    // Reorder
    const int2 id_before = (int2)(hashIndex.y % width, hashIndex.y / width);

    write_imagef(positions4Write, id, read_imagef(positions4Read, sampler, id_before));
    write_imagef(colors4Write, id, read_imagef(colors4Read, sampler, id_before));
    write_imagef(predictedPositionWrite, id, read_imagef(predictedPositionRead, sampler, id_before));
    
  } 

}


__global__ void HashParticles(__read_only image2d_t positions4Predicted,
                            __global uint2* hashIndexPairs,
                            const unsigned int width,
                            const unsigned int numberOfParticles) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Natural coordinates
                            CLK_ADDRESS_CLAMP           | // Clamp to zeros
                            CLK_FILTER_NEAREST;           // Don't interpolate

  const int2 id = (int2)(get_global_id(0) % width, get_global_id(0) / width);

  const unsigned int global_id = get_global_id(0);

  if( global_id < numberOfParticles ) {
      
    float4 predictedPosition = read_imagef(positions4Predicted, sampler, id);

    hashIndexPairs[global_id] = (uint2)(mortonCode(predictedPosition.xyz), global_id);
    
  } else {
    hashIndexPairs[global_id] = (uint2)(UINT_MAX, 0);
  }

}


*/
