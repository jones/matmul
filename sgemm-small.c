nclude <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

static float transbuff[1024 * 1024] __attribute__((aligned(0x1000)));


#pragma GCC optimize (3,"unroll-all-loops")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
  // We should zero C out to save time
  memset(C, 0, n * m * sizeof(float));
  
  // Write the (entire) array into the transpose buffer
  // TODO: this will overflow on large arrays
  for( int x_t = 0; x_t < COLS; x_t++) {
    for( int y_t = 0; y_t < ROWS; y_t++) {
      *(transbuff + COLS* y_t + x_t) = *(A + x_t*ROWS + y_t);
    }
  }
        
  // Mulitply the matrix
  // The good thing about this is that performance doesnt degrade as quickly
  for (int y = 0; y < ROWS; y++){
    for (int x = 0; x < ROWS; x++) {
      for (int i = 0; i < COLS; i++) {
        *(C + x * ROWS + y) += *(transbuff + COLS * y + i) * *(transbuff + COLS * x + i);
      }
    }
  }

}
