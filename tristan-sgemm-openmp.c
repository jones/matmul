/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-sh
   Tristan Jones  : cs61c-du 
*/
#include <emmintrin.h>

#pragma GCC optimize ("unsafe-loop-optimizations", "fast-math", "fp-contract=on","ira-loop-pressure","sched-pressure")
void sgemm( int m, int n, float *A, float *C )
{
  int mdiv20 = (m/20)*20;
  int ndiv5 = (n/5)*5;
  register int xi;

  float *aj, *ai, *ci;
  float *a1, *a2, *a3, *a4, *a5, *c1;

  __m128 va1, va2, va3, va4, va5;
  __m128 vc1, vc2;  

  #pragma omp parallel for private (va1, va2, va3, va4, va5, vc1, vc2, aj, ai, ci, xi, a1, a2, a3, a4, a5, c1)
  for (int j=0; j<m; j++)
  {
    aj = A + j;
    ci = C + j*m;
    for (int k=0; k<ndiv5; k+=5)
    {
      int km = k * m;
      va1 = _mm_load1_ps(aj+km);
      va2 = _mm_load1_ps(aj+km+m);
      va3 = _mm_load1_ps(aj+km+2*m);
      va4 = _mm_load1_ps(aj+km+3*m);
      va5 = _mm_load1_ps(aj+km+4*m);

      ai = A + k * m;
      c1 = ci;
      a1 = ai;
      a2 = ai + m;
      a3 = ai + 2*m;
      a4 = ai + 3*m;
      a5 = ai + 4*m;

      for (int i=0; i<mdiv20; i+=20)
      {	
        xi = 4;

        vc1 = _mm_loadu_ps(c1);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1),va1));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2),va2));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3),va3));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4),va4));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5),va5));
        _mm_storeu_ps(c1, vc2);


        vc1 = _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(c1 + xi, vc2);
        xi += 4;

        vc1 = _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(c1 + xi, vc2);
        xi += 4;

        vc1= _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        _mm_storeu_ps(c1 + xi, vc2);
        xi += 4;

        vc1 = _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        a1 += 20;
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + xi),va2));
        a2 += 20;
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + xi),va3));
        a3 += 20;
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + xi),va4));
        a4 += 20;
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + xi),va5));
        a5 += 20;
        _mm_storeu_ps(c1 + xi, vc2);
        c1 += 20;     
      }

      for (int i=mdiv20; i<(m/4)*4; i+=4)
      {
        vc1 = _mm_loadu_ps(c1);
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1),va1));
        a1 += 4;
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2),va2));
        a2 += 4;
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3),va3));
        a3 += 4;
        vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4),va4));
        a4 += 4;
        vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5),va5));
        a5 += 4;
        _mm_storeu_ps(c1, vc2);
        c1 += 4;
      }

      for (int i=(m/4)*4; i<m; i++)
      {
        float sum = (*a1) * (*(aj+km));		
        sum += (*a2) * (*(aj+km+m));
        a1++;		
        sum += (*a3) * (*(aj+km+2*m));
        a2++;
        sum += (*a4) * (*(aj+km+3*m));
        a3++;
        sum += (*a5) * (*(aj+km+4*m));
        a4++;
        *(ci + i) += sum;
        a5++;
      }
    }

    for (int k=ndiv5; k<n; k++)
    {
      int km = k * m;
      va1 = _mm_load1_ps(aj+km);
      ai = A + km;
      c1 = ci;
      a1 = ai;

      for (int i=0; i<mdiv20; i+=20)
      {	
        xi = 4;

        vc1 = _mm_loadu_ps(c1);
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1),va1));
        _mm_storeu_ps(c1, vc1);	

        vc2 = _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        _mm_storeu_ps(c1 + xi, vc2);
        xi += 4;

        vc1 = _mm_loadu_ps(c1 + xi);
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        _mm_storeu_ps(c1 + xi, vc1);
        xi += 4;

        vc2 = _mm_loadu_ps(c1 + xi);
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        _mm_storeu_ps(c1 + xi, vc2);
        xi += 4;

        vc1 = _mm_loadu_ps(c1 + xi);
        vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + xi),va1));
        a1 += 20;
        _mm_storeu_ps(c1 + xi, vc1);
        c1 += 20;
      }

      for (int i=mdiv20; i<(m/4)*4; i+=4)
      {
        vc2 = _mm_loadu_ps(c1);
        vc2 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a1),va1));
        a1 += 4;
        _mm_storeu_ps(c1, vc2);
        c1 += 4;
      }

      for (int i=(m/4)*4; i<m; i++)
      {
        *(ci + i) += (*a1) * (*(aj+km));
        a1++;
      }
    }
  }
}
