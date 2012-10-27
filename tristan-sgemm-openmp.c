



/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-sh
   Tristan Jones  : cs61c-du 
 */

#include <emmintrin.h>

//#pragma GCC optimize (2,"unroll-all-loops", "fast-math","unsafe-loop-optimizations")
void sgemm( int m, int n, float *A, float *C )
{
  register int mdiv20 = (m/20)*20;
  int ndiv5 = (n/5)*5;
  register int val20 = 20;
  int xi;

  float *aj, *ai, *ci;
  float *a1, *a2, *a3, *a4, *a5;

  //float* cptr;
  //float* aii, *aii + m, *aii + 2*m, *aii + 3*m, *aii + 4*m;

  __m128 va1, va2, va3, va4, va5;
  __m128 vc1, vc2, vc3, vc4, vc5;

  #pragma omp parallel
  #pragma omp for private (va1, va2, va3, va4, va5, vc1, vc2, vc3, vc4, vc5, aj, ai, ci, xi, a1, a2, a3, a4, a5)
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

      ai = A + k * m;
      a1 = ai;
      a2 = ai + m;
      a3 = ai + 2*m;
      a4 = ai + 3*m;
      a5 = ai + 4*m;


      for (int i=0; i<mdiv20; i+=val20)
      {     
        //float *aii = ai + i;
        float *cii = ci + i;
        xi = 4;

        vc1 = _mm_loadu_ps(cii);
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1),va1));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2),va2));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3),va3));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4),va4));
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5),va5));
        _mm_storeu_ps(cii, vc1);


        vc2 = _mm_loadu_ps(cii + xi);
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(cii + xi, vc2);
        xi += 4;

        vc3 = _mm_loadu_ps(cii + xi);
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc3 = _mm_add_ps(vc3, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(cii + xi, vc3);
        xi += 4;

        vc4 = _mm_loadu_ps(cii + xi);
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc4 = _mm_add_ps(vc4, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(cii + xi, vc4);
        xi += 4;

        vc5 = _mm_loadu_ps(cii + xi);
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc5 = _mm_add_ps(vc5, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(cii + xi, vc5);

        a1 += val20;
        a2 += val20;
        a3 += val20;
        a4 += val20;
        a5 += val20;
      }
    }
  }
}





