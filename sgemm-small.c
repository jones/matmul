#include <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

//#pragma GCC optimize (2,"unroll-all-loops", "fast-math","unsafe-loop-optimizations")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
  // Fast pointers
  float* __restrict vptra = A;
  float* __restrict vptrc = C;
  
  __m128 va1, va2, va3, va4, va5, va6;
  __m128 vc1, vc2, vc3;
  __m128 vs1, vs2;  
  
  for (int x = 0; x < COLS; x+=2) {
    for (int y = 0; y < ROWS; y++) {
	vs1 = _mm_load1_ps(A + y + ROWS*x);
	vs2 = _mm_load1_ps(A + y + ROWS*(x+1));

      for (int i = 0; i < ROWS; i+= 12) { 
		va1 = _mm_loadu_ps(A + x*ROWS + i);
		va2 = _mm_loadu_ps(A + x*ROWS + i + 4);
		va3 = _mm_loadu_ps(A + x*ROWS + i + 8);
        va4 = _mm_loadu_ps(A + ROWS*(x+1) + i);
		va5 = _mm_loadu_ps(A + ROWS*(x+1) + i + 4);
		va6 = _mm_loadu_ps(A + ROWS*(x+1) + i + 8);

		vc1 = _mm_loadu_ps(C + y*ROWS + i);
		vc2 = _mm_loadu_ps(C + y*ROWS + i + 4);
		vc3 = _mm_loadu_ps(C + y*ROWS + i + 8);

		va1 = _mm_mul_ps(va1, vs1);
		va2 = _mm_mul_ps(va2, vs1);
		va3 = _mm_mul_ps(va3, vs1);
		va4 = _mm_mul_ps(va4, vs2);
		va5 = _mm_mul_ps(va5, vs2);
		va6 = _mm_mul_ps(va6, vs2);

		vc1 = _mm_add_ps(vc1, va1);
		vc1 = _mm_add_ps(vc1, va4);
		vc2 = _mm_add_ps(vc2, va2);
		vc2 = _mm_add_ps(vc2, va5);
		vc3 = _mm_add_ps(vc3, va3);
		vc3 = _mm_add_ps(vc3, va6);

		_mm_storeu_ps(C + y*ROWS + i, vc1);
		_mm_storeu_ps(C + y*ROWS + i + 4, vc2);
		_mm_storeu_ps(C + y*ROWS + i + 8, vc3);
	  }
	}
  }

  
}