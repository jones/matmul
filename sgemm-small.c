#include <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

static float transbuff[1024 * 1024] __attribute__((aligned(0x1000)));


#pragma GCC optimize (2,"unroll-all-loops", "fast-math")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
  // We should zero C because it could cause problems
  memset(C, 0, (n +n%4) * (m+m%4) * sizeof(float));
  
   // Fast pointers
  float* __restrict vptra = A;
  float* __restrict vptrc = C;
  
  __m128 va1   =  _mm_setzero_ps();
  __m128 va2   =  _mm_setzero_ps();
  __m128 vc1   =  _mm_setzero_ps();
  __m128 vc2   =  _mm_setzero_ps();
  __m128 vscal =  _mm_setzero_ps(); // Scalar
  
  for (int x = 0; x < COLS; x++) {
    vptra = A + x*ROWS;
    for (int y = 0; y < ROWS; y++) {
	vptrc = C + y*ROWS;
	vscal = _mm_load1_ps(A + y + x*ROWS);
      for (int i = 0; i < ROWS; i+=8) { 
		
		va1 = _mm_loadu_ps(vptra + i);
		vc1 = _mm_loadu_ps(vptrc + i);
		va1 = _mm_mul_ps(va1, vscal);
		va2 = _mm_loadu_ps(vptra + i + 4);
		vc1 = _mm_add_ps(vc1, va1);
		vc2 = _mm_loadu_ps(vptrc + i + 4);	
        _mm_storeu_ps(vptrc + i, vc1);		
		va2 = _mm_mul_ps(va2, vscal);
		vc2 = _mm_add_ps(vc2, va2);
		_mm_storeu_ps(vptrc + i + 4, vc2);
	  }
	}
  }
   

  
}