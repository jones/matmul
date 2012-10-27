/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-sh
   Tristan Jones  : cs61c-du 
*/

#include <emmintrin.h> 

void sgemm( int m, int n, float *A, float *C )
{
  register int mdiv20 = (m/20)*20;
  int ndiv5 = (n/5)*5;

  float *aj, *ai, *ci;
  float *a1, *a2, *a3, *a4, *a5;

  __m128 va1, va2, va3, va4, va5;
  __m128 vc1;



  #pragma omp parallel
  #pragma omp for private (va1, va2, va3, va4, va5, vc1, aj, ai, ci, a1, a2, a3, a4, a5)
  for (int j=0; j<m; j++)
  {
    aj = A + j;
    ci = C + j*m;
    for (int k=0; k<ndiv5; k+=5)
    {
      va1 = _mm_load1_ps(aj+k*m);
      va2 = _mm_load1_ps(aj+(k+1)*m);
      va3 = _mm_load1_ps(aj+(k+2)*m);
      va4 = _mm_load1_ps(aj+(k+3)*m);
      va5 = _mm_load1_ps(aj+(k+4)*m);

      for (int i=0; i<mdiv20; i+=20)
      {     
        float *aii = ai + i;
        float *cii = ci + i;

        __m128 vc1 = _mm_loadu_ps(cii);
        __m128 vc2 = _mm_loadu_ps(cii + 4);
        __m128 vc3 = _mm_loadu_ps(cii + 8);
        __m128 vc4 = _mm_loadu_ps(cii + 12);
        __m128 vc5 = _mm_loadu_ps(cii + 16);

        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(aii),va1));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(aii + m),va2));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(aii + 2*m),va3));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(aii + 3*m),va4));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(aii + 4*m),va5));


        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(aii + 4),va1));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(aii + m + 4),va2));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(aii + 2*m + 4),va3));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(aii + 3*m + 4),va4));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(aii + 4*m + 4),va5));


        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(aii + 8),va1));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(aii + m + 8),va2));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(aii + 2*m + 8),va3));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(aii + 3*m + 8),va4));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(aii + 4*m + 8),va5));


        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(aii + 12),va1));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(aii + m + 12),va2));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(aii + 2*m + 12),va3));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(aii + 3*m + 12),va4));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(aii + 4*m + 12),va5));


        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(aii + 16),va1));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(aii + m + 16),va2));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(aii + 2*m + 16),va3));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(aii + 3*m + 16),va4));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(aii + 4*m + 16),va5));

        _mm_storeu_ps(cii, vc1);
        _mm_storeu_ps(cii + 4, vc2);
        _mm_storeu_ps(cii + 8, vc3);
        _mm_storeu_ps(cii + 12, vc4);
        _mm_storeu_ps(cii + 16, vc5);
      }
    }
  }
}
