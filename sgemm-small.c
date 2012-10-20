#include <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

//static float transbuff[1024 * 1024] __attribute__((aligne d(0x1000)));


//#pragma GCC optimize (2,"unroll-all-loops", "fast-math","unsafe-loop-optimizations")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
   // Fast pointers
  float* __restrict vptra = A;
  float* __restrict vptrc = C;
  
  __m128 va1   =  _mm_setzero_ps();
  __m128 va2   =  _mm_setzero_ps();
  __m128 va3   =  _mm_setzero_ps();
  __m128 va4   =  _mm_setzero_ps();
  
  __m128 vc1   =  _mm_setzero_ps();
  __m128 vc2   =  _mm_setzero_ps();
  __m128 vc3   =  _mm_setzero_ps();
  __m128 vc4   =  _mm_setzero_ps();
  
  __m128 vs1 =  _mm_setzero_ps(); // Scalar
  __m128 vs2 =  _mm_setzero_ps(); // Scalar
  
  
  
  for (int x = 0; x < COLS; x+=2) {
    for (int y = 0; y < ROWS; y++) {
	vs1 = _mm_load1_ps(A + y + ROWS*x);
	vs2 = _mm_load1_ps(A + y + ROWS*(x+1));
	
      for (int i = 0; i < ROWS; i+= 8) { 
		va1 = _mm_loadu_ps(A + x*ROWS + i);
		va2 = _mm_loadu_ps(A + x*ROWS + i + 4);
        va3 = _mm_loadu_ps(A + ROWS*(x+1) + i);
		va4 = _mm_loadu_ps(A + ROWS*(x+1) + i + 4);
		
		vc1 = _mm_loadu_ps(C + y*ROWS + i);
		vc2 = _mm_loadu_ps(C + y*ROWS + i + 4);
		
		va1 = _mm_mul_ps(va1, vs1);
		va2 = _mm_mul_ps(va2, vs1);
		va3 = _mm_mul_ps(va3, vs2);
		va4 = _mm_mul_ps(va4, vs2);
		
		vc1 = _mm_add_ps(vc1, va1);
		vc2 = _mm_add_ps(vc2, va2);
		vc1 = _mm_add_ps(vc1, va3);
		vc2 = _mm_add_ps(vc2, va4);
		
		_mm_storeu_ps(C + y*ROWS + i, vc1);
		_mm_storeu_ps(C + y*ROWS + i + 4, vc2);
	  }
	}
  }
  
  
  
  
  /*
  
 
   // Fast pointers
  float* __restrict vptra = A;
  float* __restrict vptrc = C;
  float* __restrict vptraTMP = A;
  float* __restrict vptrcTMP = C;
  
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
	vptraTMP = vptra;
	vptrcTMP = vptrc;
      for (int i = 0; i < ROWS; i+=8) { 
		va1 = _mm_loadu_ps(vptraTMP);
		va2 = _mm_loadu_ps(vptraTMP + 4);

		vc1 = _mm_loadu_ps(vptrcTMP);
		vc2 = _mm_loadu_ps(vptrcTMP+ 4);
		
		va1 = _mm_mul_ps(va1, vscal);
		va2 = _mm_mul_ps(va2, vscal);
		
		vc1 = _mm_add_ps(vc1, va1);
		vc2 = _mm_add_ps(vc2, va2);
		
		_mm_storeu_ps(vptrcTMP, vc1);
		_mm_storeu_ps(vptrcTMP + 4, vc2);
		vptrcTMP += 8;
		vptraTMP += 8;
	  }
	}
  }
   */

  
}