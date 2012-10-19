#include <string.h>
#include <emmintrin.h> 

#define ROWS m
#define COLS n

//static float transbuff[1024 * 1024] __attribute__((aligned(0x1000)));


//#pragma GCC optimize (2,"unroll-all-loops", "fast-math","unsafe-loop-optimizations")
void sgemm( int m, int n, float* __restrict A, float* __restrict C )
{
  
  
  float* __restrict vptra = A;
  float* __restrict vptrc = C;
  
  
  __m128 vscal1 = _mm_setzero_ps(); // vScalars
  __m128 vscal2 = _mm_setzero_ps();
  __m128 vscal3 = _mm_setzero_ps();
  __m128 vscal4 = _mm_setzero_ps();
  
  __m128 va1 = _mm_setzero_ps();  // Values loaded from A
  __m128 va2 = _mm_setzero_ps();
  __m128 va3 = _mm_setzero_ps();
  __m128 va4 = _mm_setzero_ps();
  
  __m128 vtmp1 = _mm_setzero_ps(); // Temporary registers
  __m128 vtmp2 = _mm_setzero_ps();
  __m128 vtmp3 = _mm_setzero_ps();
  __m128 vtmp4 = _mm_setzero_ps();
  
  for (int x = 0; x < COLS; x+=4) {
    for (int y = 0; y < ROWS; y++) {
	  vscal1 = _mm_load1_ps(A + y + ROWS*x);
	  vscal2 = _mm_load1_ps(A + y + ROWS*(x+1));
	  vscal3 = _mm_load1_ps(A + y + ROWS*(x+2));
	  vscal4 = _mm_load1_ps(A + y + ROWS*(x+3));
	  
	  //float *x1 = C + y*ROWS;
	  //float *x2 = C + y*ROWS + 4;
	  //float *x3 = C + y*ROWS + 8;
	  //float *x4 = C + y*ROWS + 12;
	  
	  
	  for (int i = 0; i < ROWS; i+=16) {
	    va1 = _mm_loadu_ps(A + x*ROWS + i);
		va2 = _mm_loadu_ps(A + x*ROWS + i + 4);
		va3 = _mm_loadu_ps(A + x*ROWS + i + 8);
		va4 = _mm_loadu_ps(A + x*ROWS + i + 12);
		
		vtmp1 = _mm_mul_ps(va1, vscal1); // a1*s1
		vtmp2 = _mm_mul_ps(va2, vscal2); // a2*s2
		vtmp3 = _mm_mul_ps(va1, vscal3); // a1*s3
		vtmp4 = _mm_mul_ps(va2, vscal4); // a2*s4
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a1*(s1+s3)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a2*(s2+s4)
		
		vtmp3 = _mm_mul_ps(va1, vscal2); // a1*s2
		vtmp4 = _mm_mul_ps(va2, vscal1); // a2*s1
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a1*(s1+s2+s3)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a2*(s1+s2+s4)
		
		vtmp3 = _mm_mul_ps(va1, vscal4); // a1*s4
		vtmp4 = _mm_mul_ps(va2, vscal3); // a2*s3
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a1*(s1+s2+s3+s4)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a2*(s1+s2+s3+s4)
		
		va1 = _mm_loadu_ps(C + y*ROWS + i); // Load original values
		va2 = _mm_loadu_ps(C + y*ROWS + i + 4);
		
		va1 = _mm_add_ps(va1, vtmp1);
		va2 = _mm_add_ps(va2, vtmp2);
		
	    _mm_storeu_ps(C + y*ROWS + i, va1); // Store into memory
		_mm_storeu_ps(C + y*ROWS + i + 4, va2);
		
		// Begin the second case
		vtmp1 = _mm_mul_ps(va3, vscal1); // a3*s1
		vtmp2 = _mm_mul_ps(va4, vscal2); // a4*s2
		vtmp3 = _mm_mul_ps(va3, vscal3); // a3*s3
		vtmp4 = _mm_mul_ps(va4, vscal4); // a4*s4
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a3*(s1+s3)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a4*(s2+s4)
		
		vtmp3 = _mm_mul_ps(va3, vscal2); // a3*s2
		vtmp4 = _mm_mul_ps(va4, vscal1); // a4*s1
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a3*(s1+s2+s3)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a4*(s1+s2+s4)
		
		vtmp3 = _mm_mul_ps(va3, vscal4); // a3*s4
		vtmp4 = _mm_mul_ps(va4, vscal3); // a4*s3
		
		vtmp1 = _mm_add_ps(vtmp1, vtmp3); // a3*(s1+s2+s3+s4)
		vtmp2 = _mm_add_ps(vtmp2, vtmp4); // a4*(s1+s2+s3+s4)
		
		va1 = _mm_loadu_ps(C + y*ROWS + i + 8); // Load original values
		va2 = _mm_loadu_ps(C + y*ROWS + i + 12);
		
		va1 = _mm_add_ps(va1, vtmp1);
		va2 = _mm_add_ps(va2, vtmp2);
		
		_mm_storeu_ps(C + y*ROWS + i + 8, va1); // Store into memory
		_mm_storeu_ps(C + y*ROWS + i + 12, va2);
		
		//x1 += 16;
		//x2 += 16;
		//x3 += 16;
		//x4 += 16;
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