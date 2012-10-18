/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-
   Tristan Jones  : cs61c-du 
*/

#include <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

// Declare static buffer arrays
static float tnybuffer[16 *  64] __attribute__((aligned(0x1000)));
static float smlbuffer[16 * 128] __attribute__((aligned(0x1000)));
static float medbuffer[16 * 192] __attribute__((aligned(0x1000)));
static float hugbuffer[16 * 300] __attribute__((aligned(0x1000)));
static float transbuff[100 * 300 * sizeof(float)] __attribute__((aligned(0x1000)));


#pragma GCC optimize (3,"unroll-all-loops")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
  // Zero out C because it may be filled with garbage
  // TODO: possibly this is not needed?
  memset(C, 0, n * m * sizeof(float));

  // Write A transpose into the giant buffer
  // TODO: do this better
  // TODO: add padding

  for (int x_t = 0; x_t < COLS; x_t++) {
    for (int y_t = 0; y_t < ROWS; y_t++) {
      *(transbuff + y_t * COLS + x_t) = *(A + x_t * ROWS + y_t);
    }
  }

  // Now do a bad matrix multiply
  


  
  


}
