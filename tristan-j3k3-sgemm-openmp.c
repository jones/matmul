/* CS61C Project 3: Matrix Multiply Parallelization
 * Jian Wei Leong : cs61c-sh
 * Tristan Jones  : cs61c-du
 *
 * Comments:
 * This is an experimental build I made after I realized that most of our writes were L3 cache hits, allowing
 * us to be more liberal on memory stores. It ended up running at ~88 Gflop/s, which is slower than the
 * 'for j (2)' loop ordering. We won't turn this and will just keep this as reference.
 * 
 * -Tristan
 *
 *
 * OMP:
 * - Parallel for j (3)
 * -- for k (3)
 * --- for i (32)
 * --- for i (4)
 * --- for i (1)
 * -- for k (1)
 * --- for i (32)
 * --- for i (4)
 * --- for i (1)
 * - Parallel for j (1)
 * -- for k (4)
 * --- for i (32)
 * --- for i (4)
 * --- for i (1)
 * -- for k (1)
 * --- for i (32)
 * --- for i (4)
 * --- for i (1)
 */
#include <emmintrin.h>

#pragma GCC optimize ("unsafe-loop-optimizations", "unroll-loops", "fast-math", "fp-contract=on", "ira-loop-pressure", "sched-pressure", "align-loops=16")
void
sgemm(const int m, const int n, float* const __restrict__ A, float* const __restrict__ C)
{

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
   for (int j = 0; j < (m / 3) * 3; j += 3)
    {
     float *a1, *a2, *a3, *r1, *r2, *r3;

     float* const aj = A + j;
     float* const ri = C + j * m;

     int k;
     for (k = 0; k < (n / 3) * 3; k += 3)
      {
       int const km = k * m;
       __m128 vr1, vr2, vr3, vt1, vt2;
       __m128 va1 = _mm_load1_ps(aj + km);
       __m128 vb1 = _mm_load1_ps(aj + km + 1);
       __m128 vc1 = _mm_load1_ps(aj + km + 2);
       __m128 va2 = _mm_load1_ps(aj + km + m);
       __m128 vb2 = _mm_load1_ps(aj + km + m + 1);
       __m128 vc2 = _mm_load1_ps(aj + km + m + 2);
       __m128 va3 = _mm_load1_ps(aj + km + 2 * m);
       __m128 vb3 = _mm_load1_ps(aj + km + 2 * m + 1);
       __m128 vc3 = _mm_load1_ps(aj + km + 2 * m + 2);

       float* const ai = A + km;
       r1 = ri;
       r2 = ri + m;
       r3 = ri + 2 * m;

       a1 = ai;
       a2 = ai + m;
       a3 = ai + 2 * m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // Offset 0
         vt1 = _mm_loadu_ps(a1);
         vr1 = _mm_loadu_ps(r1);
         vr2 = _mm_loadu_ps(r2);
         vr3 = _mm_loadu_ps(r3);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         vt2 = _mm_loadu_ps(a2);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc2));
         vt1 = _mm_loadu_ps(a3);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc3));
         _mm_storeu_ps(r1, vr1);
         _mm_storeu_ps(r2, vr2);
         _mm_storeu_ps(r3, vr3);

         // Offset 4
         vt2 = _mm_loadu_ps(a1 + 4);
         vr1 = _mm_loadu_ps(r1 + 4);
         vr2 = _mm_loadu_ps(r2 + 4);
         vr3 = _mm_loadu_ps(r3 + 4);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         vt1 = _mm_loadu_ps(a2 + 4);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc2));
         vt2 = _mm_loadu_ps(a3 + 4);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc3));
         _mm_storeu_ps(r1 + 4, vr1);
         _mm_storeu_ps(r2 + 4, vr2);
         _mm_storeu_ps(r3 + 4, vr3);

         // Offset 8
         vt1 = _mm_loadu_ps(a1 + 8);
         vr1 = _mm_loadu_ps(r1 + 8);
         vr2 = _mm_loadu_ps(r2 + 8);
         vr3 = _mm_loadu_ps(r3 + 8);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         vt2 = _mm_loadu_ps(a2 + 8);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc2));
         vt1 = _mm_loadu_ps(a3 + 8);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc3));
         _mm_storeu_ps(r1 + 8, vr1);
         _mm_storeu_ps(r2 + 8, vr2);
         _mm_storeu_ps(r3 + 8, vr3);

         // Offset 12
         vt2 = _mm_loadu_ps(a1 + 12);
         vr1 = _mm_loadu_ps(r1 + 12);
         vr2 = _mm_loadu_ps(r2 + 12);
         vr3 = _mm_loadu_ps(r3 + 12);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         vt1 = _mm_loadu_ps(a2 + 12);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc2));
         vt2 = _mm_loadu_ps(a3 + 12);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc3));
         _mm_storeu_ps(r1 + 12, vr1);
         _mm_storeu_ps(r2 + 12, vr2);
         _mm_storeu_ps(r3 + 12, vr3);

         // Offset 16
         vt1 = _mm_loadu_ps(a1 + 16);
         vr1 = _mm_loadu_ps(r1 + 16);
         vr2 = _mm_loadu_ps(r2 + 16);
         vr3 = _mm_loadu_ps(r3 + 16);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         vt2 = _mm_loadu_ps(a2 + 16);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc2));
         vt1 = _mm_loadu_ps(a3 + 16);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc3));
         _mm_storeu_ps(r1 + 16, vr1);
         _mm_storeu_ps(r2 + 16, vr2);
         _mm_storeu_ps(r3 + 16, vr3);

         // Offset 20
         vt2 = _mm_loadu_ps(a1 + 20);
         vr1 = _mm_loadu_ps(r1 + 20);
         vr2 = _mm_loadu_ps(r2 + 20);
         vr3 = _mm_loadu_ps(r3 + 20);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         vt1 = _mm_loadu_ps(a2 + 20);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc2));
         vt2 = _mm_loadu_ps(a3 + 20);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc3));
         _mm_storeu_ps(r1 + 20, vr1);
         _mm_storeu_ps(r2 + 20, vr2);
         _mm_storeu_ps(r3 + 20, vr3);

         // Offset 24
         vt1 = _mm_loadu_ps(a1 + 24);
         vr1 = _mm_loadu_ps(r1 + 24);
         vr2 = _mm_loadu_ps(r2 + 24);
         vr3 = _mm_loadu_ps(r3 + 24);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         vt2 = _mm_loadu_ps(a2 + 24);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc2));
         vt1 = _mm_loadu_ps(a3 + 24);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc3));
         _mm_storeu_ps(r1 + 24, vr1);
         _mm_storeu_ps(r2 + 24, vr2);
         _mm_storeu_ps(r3 + 24, vr3);

         // Offset 28
         vt2 = _mm_loadu_ps(a1 + 28);
         a1 += 32;
         vr1 = _mm_loadu_ps(r1 + 28);
         r1 += 32;
         vr2 = _mm_loadu_ps(r2 + 28);
         r2 += 32;
         vr3 = _mm_loadu_ps(r3 + 28);
         r3 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         vt1 = _mm_loadu_ps(a2 + 28);
         a2 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc2));
         vt2 = _mm_loadu_ps(a3 + 28);
         a3 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc3));
         _mm_storeu_ps(r1 - 4, vr1);
         _mm_storeu_ps(r2 - 4, vr2);
         _mm_storeu_ps(r3 - 4, vr3);

        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vt1 = _mm_loadu_ps(a1);
         a1 += 4;
         vr1 = _mm_loadu_ps(r1);
         r1 += 4;
         vr2 = _mm_loadu_ps(r2);
         r2 += 4;
         vr3 = _mm_loadu_ps(r3);
         r3 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         vt2 = _mm_loadu_ps(a2);
         a2 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va2));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb2));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc2));
         vt1 = _mm_loadu_ps(a3);
         a3 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va3));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb3));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc3));
         _mm_storeu_ps(r1 - 4, vr1);
         _mm_storeu_ps(r2 - 4, vr2);
         _mm_storeu_ps(r3 - 4, vr3);
        }

       for (; i < m; i++)
        {
         float sum1, sum2, sum3, fa1, fa2, fa3;
         fa1 = *a1;
         sum1 = fa1 * (*(aj + km));
         sum2 = fa1 * (*(aj + km + 1));
         sum3 = fa1 * (*(aj + km + 2));
         a1++;

         fa2 = *a2;
         sum1 += fa2 * (*(aj + km + m));
         sum2 += fa2 * (*(aj + km + m + 1));
         sum3 += fa2 * (*(aj + km + m + 2));
         a2++;

         fa3 = *a3;
         sum1 += fa3 * (*(aj + km + 2 * m));
         sum2 += fa3 * (*(aj + km + 2 * m + 1));
         sum3 += fa3 * (*(aj + km + 2 * m + 2));
         a3++;

         *(ri + i) += sum1;
         *(ri + i + m) += sum2;
         *(ri + i + 2 * m) += sum3;
        }
      }

     for (; k < n; k++)
      {
       int const km = k * m;
       __m128 vr1, vr2, vr3, vt1, vt2;
       __m128 va1 = _mm_load1_ps(aj + km);
       __m128 vb1 = _mm_load1_ps(aj + km + 1);
       __m128 vc1 = _mm_load1_ps(aj + km + 2);

       r1 = ri;
       r2 = ri + m;
       r3 = ri + 2 * m;
       a1 = A + km;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // Offset 0
         vt1 = _mm_loadu_ps(a1);
         vr1 = _mm_loadu_ps(r1);
         vr2 = _mm_loadu_ps(r2);
         vr3 = _mm_loadu_ps(r3);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         _mm_storeu_ps(r1, vr1);
         _mm_storeu_ps(r2, vr2);
         _mm_storeu_ps(r3, vr3);

         // Offset 4
         vt2 = _mm_loadu_ps(a1 + 4);
         vr1 = _mm_loadu_ps(r1 + 4);
         vr2 = _mm_loadu_ps(r2 + 4);
         vr3 = _mm_loadu_ps(r3 + 4);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         _mm_storeu_ps(r1 + 4, vr1);
         _mm_storeu_ps(r2 + 4, vr2);
         _mm_storeu_ps(r3 + 4, vr3);

         // Offset 8
         vt1 = _mm_loadu_ps(a1 + 8);
         vr1 = _mm_loadu_ps(r1 + 8);
         vr2 = _mm_loadu_ps(r2 + 8);
         vr3 = _mm_loadu_ps(r3 + 8);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         _mm_storeu_ps(r1 + 8, vr1);
         _mm_storeu_ps(r2 + 8, vr2);
         _mm_storeu_ps(r3 + 8, vr3);

         // Offset 12
         vt2 = _mm_loadu_ps(a1 + 12);
         vr1 = _mm_loadu_ps(r1 + 12);
         vr2 = _mm_loadu_ps(r2 + 12);
         vr3 = _mm_loadu_ps(r3 + 12);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         _mm_storeu_ps(r1 + 12, vr1);
         _mm_storeu_ps(r2 + 12, vr2);
         _mm_storeu_ps(r3 + 12, vr3);

         // Offset 16
         vt1 = _mm_loadu_ps(a1 + 16);
         vr1 = _mm_loadu_ps(r1 + 16);
         vr2 = _mm_loadu_ps(r2 + 16);
         vr3 = _mm_loadu_ps(r3 + 16);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         _mm_storeu_ps(r1 + 16, vr1);
         _mm_storeu_ps(r2 + 16, vr2);
         _mm_storeu_ps(r3 + 16, vr3);

         // Offset 20
         vt2 = _mm_loadu_ps(a1 + 20);
         vr1 = _mm_loadu_ps(r1 + 20);
         vr2 = _mm_loadu_ps(r2 + 20);
         vr3 = _mm_loadu_ps(r3 + 20);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         _mm_storeu_ps(r1 + 20, vr1);
         _mm_storeu_ps(r2 + 20, vr2);
         _mm_storeu_ps(r3 + 20, vr3);

         // Offset 24
         vt1 = _mm_loadu_ps(a1 + 24);
         vr1 = _mm_loadu_ps(r1 + 24);
         vr2 = _mm_loadu_ps(r2 + 24);
         vr3 = _mm_loadu_ps(r3 + 24);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         _mm_storeu_ps(r1 + 24, vr1);
         _mm_storeu_ps(r2 + 24, vr2);
         _mm_storeu_ps(r3 + 24, vr3);

         // Offset 28
         vt2 = _mm_loadu_ps(a1 + 28);
         a1 += 32;
         vr1 = _mm_loadu_ps(r1 + 28);
         r1 += 32;
         vr2 = _mm_loadu_ps(r2 + 28);
         r2 += 32;
         vr3 = _mm_loadu_ps(r3 + 28);
         r3 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt2, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt2, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt2, vc1));
         _mm_storeu_ps(r1 - 4, vr1);
         _mm_storeu_ps(r2 - 4, vr2);
         _mm_storeu_ps(r3 - 4, vr3);

        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vt1 = _mm_loadu_ps(a1);
         a1 += 4;
         vr1 = _mm_loadu_ps(r1);
         r1 += 4;
         vr2 = _mm_loadu_ps(r2);
         r2 += 4;
         vr3 = _mm_loadu_ps(r3);
         r3 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(vt1, va1));
         vr2 = _mm_add_ps(vr2, _mm_mul_ps(vt1, vb1));
         vr3 = _mm_add_ps(vr3, _mm_mul_ps(vt1, vc1));
         _mm_storeu_ps(r1 - 4, vr1);
         _mm_storeu_ps(r2 - 4, vr2);
         _mm_storeu_ps(r3 - 4, vr3);
        }

       for (; i < m; i++)
        {
         float sum1, sum2, sum3, fa1;
         fa1 = *a1;
         sum1 = fa1 * (*(aj + km));
         sum2 = fa1 * (*(aj + km + 1));
         sum3 = fa1 * (*(aj + km + 2));
         a1++;

         *(ri + i) += sum1;
         *(ri + i + m) += sum2;
         *(ri + i + 2 * m) += sum3;
        }
      }
    }

#pragma omp for schedule(dynamic, 1)
   for (int j = (m / 3) * 3; j < m; j++)
    {
     float *a1, *a2, *a3, *a4, *r1;
     float* const aj = A + j;
     float* const ri = C + j * m;

     int k;
     for (k = 0; k < (n / 4) * 4; k += 4)
      {
       int const km = k * m;
       __m128 vr1;
       __m128 va1 = _mm_load1_ps(aj + km);
       __m128 va2 = _mm_load1_ps(aj + km + m);
       __m128 va3 = _mm_load1_ps(aj + km + 2 * m);
       __m128 va4 = _mm_load1_ps(aj + km + 3 * m);

       float* const ai = A + km;
       r1 = ri;
       a1 = ai;
       a2 = ai + m;
       a3 = ai + 2 * m;
       a4 = ai + 3 * m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         vr1 = _mm_loadu_ps(r1);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4), va4));
         _mm_storeu_ps(r1, vr1);

         vr1 = _mm_loadu_ps(r1 + 4);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 4), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 4), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 4), va4));
         _mm_storeu_ps(r1 + 4, vr1);

         vr1 = _mm_loadu_ps(r1 + 8);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 8), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 8), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 8), va4));
         _mm_storeu_ps(r1 + 8, vr1);

         vr1 = _mm_loadu_ps(r1 + 12);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 12), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 12), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 12), va4));
         _mm_storeu_ps(r1 + 12, vr1);

         vr1 = _mm_loadu_ps(r1 + 16);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 16), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 16), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 16), va4));
         _mm_storeu_ps(r1 + 16, vr1);

         vr1 = _mm_loadu_ps(r1 + 20);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 20), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 20), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 20), va4));
         _mm_storeu_ps(r1 + 20, vr1);

         vr1 = _mm_loadu_ps(r1 + 24);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 24), va2));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 24), va3));
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 24), va4));
         _mm_storeu_ps(r1 + 24, vr1);

         vr1 = _mm_loadu_ps(r1 + 28);
         r1 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1));
         a1 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2 + 28), va2));
         a2 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3 + 28), va3));
         a3 += 32;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4 + 28), va4));
         a4 += 32;
         _mm_storeu_ps(r1 - 4, vr1);

        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vr1 = _mm_loadu_ps(r1);
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
         a1 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a2), va2));
         a2 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
         a3 += 4;
         vr1 = _mm_add_ps(vr1, _mm_mul_ps(_mm_loadu_ps(a4), va4));
         a4 += 4;
         _mm_storeu_ps(r1, vr1);
         r1 += 4;
        }

       for (; i < m; i++)
        {
         // The *(aj + km) calls miss the cache about 20% of the time
         // It would run faster with some inline assembly
         float sum = (*a1) * (*(aj + km));
         sum += (*a2) * (*(aj + km + m));
         a1++;
         sum += (*a3) * (*(aj + km + 2 * m));
         a2++;
         sum += (*a4) * (*(aj + km + 3 * m));
         a3++;
         a4++;
         *(ri + i) += sum;
        }
      }

     for (; k < n; k++)
      {
       int const km = k * m;
       __m128 vr1, va2, va3, va4, va5;
       __m128 va1 = _mm_load1_ps(aj + km);
       r1 = ri;
       a1 = A + km;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // va2, va3, va4, va5 may be used as temp here
         vr1 = _mm_loadu_ps(r1);
         va2 = _mm_mul_ps(_mm_loadu_ps(a1), va1);
         vr1 = _mm_add_ps(vr1, va2);
         _mm_storeu_ps(r1, vr1);

         vr1 = _mm_loadu_ps(r1 + 4);
         va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1);
         vr1 = _mm_add_ps(vr1, va3);
         _mm_storeu_ps(r1 + 4, vr1);

         vr1 = _mm_loadu_ps(r1 + 8);
         va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1);
         vr1 = _mm_add_ps(vr1, va4);
         _mm_storeu_ps(r1 + 8, vr1);

         vr1 = _mm_loadu_ps(r1 + 12);
         va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1);
         vr1 = _mm_add_ps(vr1, va5);
         _mm_storeu_ps(r1 + 12, vr1);

         vr1 = _mm_loadu_ps(r1 + 16);
         va2 = _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1);
         vr1 = _mm_add_ps(vr1, va2);
         _mm_storeu_ps(r1 + 16, vr1);

         vr1 = _mm_loadu_ps(r1 + 20);
         va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1);
         vr1 = _mm_add_ps(vr1, va3);
         _mm_storeu_ps(r1 + 20, vr1);

         vr1 = _mm_loadu_ps(r1 + 24);
         va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1);
         vr1 = _mm_add_ps(vr1, va4);
         _mm_storeu_ps(r1 + 24, vr1);

         vr1 = _mm_loadu_ps(r1 + 28);
         r1 += 32;
         va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1);
         a1 += 32;
         vr1 = _mm_add_ps(vr1, va5);
         _mm_storeu_ps(r1 - 4, vr1);
        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vr1 = _mm_loadu_ps(r1);
         va3 = _mm_loadu_ps(a1);
         a1 += 4;
         va4 = _mm_mul_ps(va3, va1);
         vr1 = _mm_add_ps(vr1, va4);
         _mm_storeu_ps(r1, vr1);
         r1 += 4;
        }

       for (; i < m; i++)
        {
         *(ri + i) += (*a1) * (*(aj + km));
         a1++;
        }
      }
    }
  }
}
