/* CS61C Project 3: Matrix Multiply Parallelization
 * Jian Wei Leong : cs61c-sh
 * Tristan Jones  : cs61c-du
 */
#include <emmintrin.h>

#pragma GCC optimize ("unsafe-loop-optimizations", "fast-math", "fp-contract=on", "ira-loop-pressure", "sched-pressure")
void
sgemm(const int m, const int n, float* __restrict__ A, float* __restrict__ C)
{
 float *a1, *a2, *a3, *a4, *a5, *c1;

 __m128 va1, va2, va3, va4, va5;
 __m128 vc1, vc2;

 #pragma omp parallel for private (va1, va2, va3, va4, va5, vc1, vc2, a1, a2, a3, a4, a5, c1) schedule(dynamic, 1)
 for (int j = 0; j < m; j++)
  {
   float* aj = A + j;
   float* ci = C + j * m;
   int k;
   for (k = 0; k < (n / 5) * 5; k += 5)
    {
     int km = k * m;
     va1 = _mm_load1_ps(aj + km);
     va2 = _mm_load1_ps(aj + km + m);
     va3 = _mm_load1_ps(aj + km + 2 * m);
     va4 = _mm_load1_ps(aj + km + 3 * m);
     va5 = _mm_load1_ps(aj + km + 4 * m);

     float* ai = A + k * m;
     c1 = ci;
     a1 = ai;
     a2 = ai + m;
     a3 = ai + 2 * m;
     a4 = ai + 3 * m;
     a5 = ai + 4 * m;

     int i;
     for (i = 0; i < (m / 32) * 32; i += 32)
      {
       vc1 = _mm_loadu_ps(c1);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5), va5));
       _mm_storeu_ps(c1, vc2);

       vc1 = _mm_loadu_ps(c1 + 4);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 4), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 4), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 4), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 4), va5));
       _mm_storeu_ps(c1 + 4, vc2);

       vc1 = _mm_loadu_ps(c1 + 8);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 8), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 8), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 8), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 8), va5));
       _mm_storeu_ps(c1 + 8, vc2);

       vc1 = _mm_loadu_ps(c1 + 12);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 12), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 12), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 12), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 12), va5));
       _mm_storeu_ps(c1 + 12, vc2);

       vc1 = _mm_loadu_ps(c1 + 16);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 16), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 16), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 16), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 16), va5));
       _mm_storeu_ps(c1 + 16, vc2);

       vc1 = _mm_loadu_ps(c1 + 20);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 20), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 20), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 20), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 20), va5));
       _mm_storeu_ps(c1 + 20, vc2);

       vc1 = _mm_loadu_ps(c1 + 24);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 24), va2));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 24), va3));
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 24), va4));
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 24), va5));
       _mm_storeu_ps(c1 + 24, vc2);

       vc1 = _mm_loadu_ps(c1 + 28);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1));
       a1 += 32;
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2 + 28), va2));
       a2 += 32;
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 28), va3));
       a3 += 32;
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4 + 28), va4));
       a4 += 32;
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5 + 28), va5));
       a5 += 32;
       _mm_storeu_ps(c1 + 28, vc2);
       c1 += 32;
      }

     for (; i < (m / 4) * 4; i += 4)
      {
       vc1 = _mm_loadu_ps(c1);
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
       a1 += 4;
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a2), va2));
       a2 += 4;
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
       a3 += 4;
       vc1 = _mm_add_ps(vc2, _mm_mul_ps(_mm_loadu_ps(a4), va4));
       a4 += 4;
       vc2 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a5), va5));
       a5 += 4;
       _mm_storeu_ps(c1, vc2);
       c1 += 4;
      }

     for (; i < m; i++)
      {
       float sum = (*a1) * (*(aj + km));
       sum += (*a2) * (*(aj + km + m));
       a1++;
       sum += (*a3) * (*(aj + km + 2 * m));
       a2++;
       sum += (*a4) * (*(aj + km + 3 * m));
       a3++;
       sum += (*a5) * (*(aj + km + 4 * m));
       a4++;
       *(ci + i) += sum;
       a5++;
      }
    }

   for (; k < n; k++)
    {
     int km = k * m;
     va1 = _mm_load1_ps(aj + km);
     float* ai = A + km;
     c1 = ci;
     a1 = ai;

     int i;
     for (i = 0; i < (m / 32) * 32; i += 32)
      {
       // va2, va3, va4, va5 may be used as temp here
       vc1 = _mm_loadu_ps(c1);
       va2 = _mm_mul_ps(_mm_loadu_ps(a1), va1);
       vc1 = _mm_add_ps(vc1, va2);
       _mm_storeu_ps(c1, vc1);

       vc2 = _mm_loadu_ps(c1 + 4);
       va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1);
       vc2 = _mm_add_ps(vc2, va3);
       _mm_storeu_ps(c1 + 4, vc2);

       vc1 = _mm_loadu_ps(c1 + 8);
       va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1);
       vc1 = _mm_add_ps(vc1, va4);
       _mm_storeu_ps(c1 + 8, vc1);

       vc2 = _mm_loadu_ps(c1 + 12);
       va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1);
       vc2 = _mm_add_ps(vc2, va5);
       _mm_storeu_ps(c1 + 12, vc2);

       vc1 = _mm_loadu_ps(c1 + 16);
       va2 = _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1);
       vc1 = _mm_add_ps(vc1, va2);
       _mm_storeu_ps(c1 + 16, vc1);

       vc2 = _mm_loadu_ps(c1 + 20);
       va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1);
       vc2 = _mm_add_ps(vc2, va3);
       _mm_storeu_ps(c1 + 20, vc2);

       vc1 = _mm_loadu_ps(c1 + 24);
       va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1);
       vc1 = _mm_add_ps(vc1, va4);
       _mm_storeu_ps(c1 + 24, vc1);

       vc2 = _mm_loadu_ps(c1 + 28);
       va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1);
       a1 += 32;
       vc2 = _mm_add_ps(vc2, va5);
       _mm_storeu_ps(c1 + 28, vc2);
       c1 += 32;
      }

     for (; i < (m / 4) * 4; i += 4)
      {
       vc2 = _mm_loadu_ps(c1);
       va3 = _mm_loadu_ps(a1);
       a1 += 4;
       va4 = _mm_mul_ps(va3, va1);
       vc2 = _mm_add_ps(vc2, va4);
       _mm_storeu_ps(c1, vc2);
       c1 += 4;
      }

     for (; i < m; i++)
      {
       *(ci + i) += (*a1) * (*(aj + km));
       a1++;
      }
    }
  }
}
