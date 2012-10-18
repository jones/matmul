/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-
   Tristan Jones  : cs61c-du 
*/
#include <string.h>
#include <emmintrin.h> 

// Declare static buffer arrays
static float tnybuffer[16 *  64] __attribute__((aligned(0x1000)));
static float smlbuffer[16 * 128] __attribute__((aligned(0x1000)));
static float medbuffer[16 * 192] __attribute__((aligned(0x1000)));
static float hugbuffer[16 * 300] __attribute__((aligned(0x1000)));


void sgemm( int m, int n, float *A, float *C )
{
  // Zero out C because it may be filled with garbage
  memset(C, 0, n * m * sizeof(float));



  // First copy A into a buffer

}
