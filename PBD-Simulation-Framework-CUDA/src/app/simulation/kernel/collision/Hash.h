#ifndef HASH_H
#define HASH_H

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the cube [0.0, 1023.0].
__device__ __forceinline__ unsigned int mortonCode(float4 pos) {
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

__device__ __forceinline__ unsigned int getHash(float4 pos) {
   return mortonCode(pos);
}

#endif // HASH_H