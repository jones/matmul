/* CS61C Project 3: Matrix Multiply Parallelization
 * Jian Wei Leong : cs61c-sh
 * Tristan Jones  : cs61c-du
 */
#include <emmintrin.h>

#pragma GCC optimize ("unsafe-loop-optimizations", "unroll-loops", "fast-math", "fp-contract=on", "ira-loop-pressure", "sched-pressure", "align-loops=16")
void
sgemm(const int m, const int n, float* const __restrict__ A, float* const __restrict__ C)
{

#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
   for (int j = 0; j < (m / 2) * 2; j += 2)
    {
     float *a1, *a2, *a3, *a4, *c1, *c2;

     float* const aj = A + j;
     float* const ci = C + j * m;

     int k;
     for (k = 0; k < (n / 4) * 4; k += 4)
      {
       float* const ajkm = aj + k * m;
       float* const ai = A + k * m;
       
       __m128 vc1, vc2, vt1, vt2;
       __m128 va1 = _mm_load1_ps(ajkm);
       __m128 vb1 = _mm_load1_ps(ajkm + 1);
       __m128 va2 = _mm_load1_ps(ajkm + m);
       __m128 vb2 = _mm_load1_ps(ajkm + m + 1);
       __m128 va3 = _mm_load1_ps(ajkm + 2 * m);
       __m128 vb3 = _mm_load1_ps(ajkm + 2 * m + 1);
       __m128 va4 = _mm_load1_ps(ajkm + 3 * m);
       __m128 vb4 = _mm_load1_ps(ajkm + 3 * m + 1);

       c1 = ci;
       c2 = ci + m;

       a1 = ai;
       a2 = ai + m;
       a3 = ai + 2 * m;
       a4 = ai + 3 * m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // Offset 0
         vt1 = _mm_loadu_ps(a1);
         vc1 = _mm_loadu_ps(c1);
         vc2 = _mm_loadu_ps(c2);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1, vc1);
         _mm_storeu_ps(c2, vc2);

         // Offset 4
         vt1 = _mm_loadu_ps(a1 + 4);
         vc1 = _mm_loadu_ps(c1 + 4);
         vc2 = _mm_loadu_ps(c2 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 4, vc1);
         _mm_storeu_ps(c2 + 4, vc2);

         // Offset 8
         vt1 = _mm_loadu_ps(a1 + 8);
         vc1 = _mm_loadu_ps(c1 + 8);
         vc2 = _mm_loadu_ps(c2 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 8, vc1);
         _mm_storeu_ps(c2 + 8, vc2);

         // Offset 12
         vt1 = _mm_loadu_ps(a1 + 12);
         vc1 = _mm_loadu_ps(c1 + 12);
         vc2 = _mm_loadu_ps(c2 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 12, vc1);
         _mm_storeu_ps(c2 + 12, vc2);

         // Offset 16
         vt1 = _mm_loadu_ps(a1 + 16);
         vc1 = _mm_loadu_ps(c1 + 16);
         vc2 = _mm_loadu_ps(c2 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 16, vc1);
         _mm_storeu_ps(c2 + 16, vc2);

         // Offset 20
         vt1 = _mm_loadu_ps(a1 + 20);
         vc1 = _mm_loadu_ps(c1 + 20);
         vc2 = _mm_loadu_ps(c2 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 20, vc1);
         _mm_storeu_ps(c2 + 20, vc2);

         // Offset 24
         vt1 = _mm_loadu_ps(a1 + 24);
         vc1 = _mm_loadu_ps(c1 + 24);
         vc2 = _mm_loadu_ps(c2 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 + 24, vc1);
         _mm_storeu_ps(c2 + 24, vc2);

         // Offset 28
         vt1 = _mm_loadu_ps(a1 + 28);
         a1 += 32;
         vc1 = _mm_loadu_ps(c1 + 28);
         c1 += 32;
         vc2 = _mm_loadu_ps(c2 + 28);
         c2 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2 + 28);
         a2 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3 + 28);
         a3 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4 + 28);
         a4 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1 - 4, vc1);
         _mm_storeu_ps(c2 - 4, vc2);
        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vt1 = _mm_loadu_ps(a1);
         a1 += 4;
         vc1 = _mm_loadu_ps(c1);
         vc2 = _mm_loadu_ps(c2);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         vt2 = _mm_loadu_ps(a2);
         a2 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va2));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb2));
         vt1 = _mm_loadu_ps(a3);
         a3 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va3));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb3));
         vt2 = _mm_loadu_ps(a4);
         a4 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt2, va4));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt2, vb4));
         _mm_storeu_ps(c1, vc1);
         c1 += 4;
         _mm_storeu_ps(c2, vc2);
         c2 += 4;
        }

//     float* as;
//     asm("subq $16, %rsp");
//     asm("movsd %0, 0(%rsp)" : : "x" (va1));
//     asm("movw 0(%rsp), a1" : "=r" (as):);
//     asm("addq $16, %rsp");

       for (; i < m; i++)
        {
         float sum1, sum2, fa1, fa2, fa3, fa4;
         
         fa1 = *a1;
         a1++;
         sum1 = fa1 * (*(ajkm));
         sum2 = fa1 * (*(ajkm + 1));

         fa2 = *a2;
         a2++;
         sum1 += fa2 * (*(ajkm + m));
         sum2 += fa2 * (*(ajkm + m + 1));

         fa3 = *a3;
         a3++;
         sum1 += fa3 * (*(ajkm + 2 * m));
         sum2 += fa3 * (*(ajkm + 2 * m + 1));

         fa4 = *a4;
         a4++;
         sum1 += fa4 * (*(ajkm + 3 * m));
         sum2 += fa4 * (*(ajkm + 3 * m + 1));

         *(c1) += sum1;
         c1++;
         *(c2) += sum2;
         c2++;
        }
      }

     for (; k < n; k++)
      {
       float* const ajkm = aj + k * m;
       a1 = A + k * m;
       __m128 vc1, vc2, vt1, vt2;
       __m128 va1 = _mm_load1_ps(ajkm);
       __m128 vb1 = _mm_load1_ps(ajkm + 1);
       c1 = ci;
       c2 = ci + m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // Offset 0
         vt1 = _mm_loadu_ps(a1);
         vc1 = _mm_loadu_ps(c1);
         vc2 = _mm_loadu_ps(c2);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1, vc1);
         _mm_storeu_ps(c2, vc2);

         // Offset 4
         vt1 = _mm_loadu_ps(a1 + 4);
         vc1 = _mm_loadu_ps(c1 + 4);
         vc2 = _mm_loadu_ps(c2 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 4, vc1);
         _mm_storeu_ps(c2 + 4, vc2);

         // Offset 8
         vt1 = _mm_loadu_ps(a1 + 8);
         vc1 = _mm_loadu_ps(c1 + 8);
         vc2 = _mm_loadu_ps(c2 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 8, vc1);
         _mm_storeu_ps(c2 + 8, vc2);

         // Offset 12
         vt1 = _mm_loadu_ps(a1 + 12);
         vc1 = _mm_loadu_ps(c1 + 12);
         vc2 = _mm_loadu_ps(c2 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 12, vc1);
         _mm_storeu_ps(c2 + 12, vc2);

         // Offset 16
         vt1 = _mm_loadu_ps(a1 + 16);
         vc1 = _mm_loadu_ps(c1 + 16);
         vc2 = _mm_loadu_ps(c2 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 16, vc1);
         _mm_storeu_ps(c2 + 16, vc2);

         // Offset 20
         vt1 = _mm_loadu_ps(a1 + 20);
         vc1 = _mm_loadu_ps(c1 + 20);
         vc2 = _mm_loadu_ps(c2 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 20, vc1);
         _mm_storeu_ps(c2 + 20, vc2);

         // Offset 24
         vt1 = _mm_loadu_ps(a1 + 24);
         vc1 = _mm_loadu_ps(c1 + 24);
         vc2 = _mm_loadu_ps(c2 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 + 24, vc1);
         _mm_storeu_ps(c2 + 24, vc2);

         // Offset 28
         vt1 = _mm_loadu_ps(a1 + 28);
         a1 += 32;
         vc1 = _mm_loadu_ps(c1 + 28);
         c1 += 32;
         vc2 = _mm_loadu_ps(c2 + 28);
         c2 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 - 4, vc1);
         _mm_storeu_ps(c2 - 4, vc2);
        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vt1 = _mm_loadu_ps(a1);
         a1 += 4;
         vc1 = _mm_loadu_ps(c1);
         c1 += 4;
         vc2 = _mm_loadu_ps(c2);
         c2 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(vt1, va1));
         vc2 = _mm_add_ps(vc2, _mm_mul_ps(vt1, vb1));
         _mm_storeu_ps(c1 - 4, vc1);
         _mm_storeu_ps(c2 - 4, vc2);
        }

       for (; i < m; i++)
        {
         float sum1, sum2, fa1;
         fa1 = *a1;
         sum1 = fa1 * (*(ajkm));
         sum2 = fa1 * (*(ajkm + 1));
         a1++;

         *(c1) += sum1;
         c1++;
         *(c2) += sum2;
         c2++;
        }
      }

    }

#pragma omp for schedule(dynamic, 1)
   for (int j = (m / 2) * 2; j < m; j++)
    {
     float *a1, *a2, *a3, *a4, *c1;
     float* const aj = A + j;
     float* const ci = C + j * m;

     int k;
     for (k = 0; k < (n / 4) * 4; k += 4)
      {
       float* const ajkm = aj + k * m;
       float* const ai = A + k * m;
       __m128 vc1;
       __m128 va1 = _mm_load1_ps(ajkm);
       __m128 va2 = _mm_load1_ps(ajkm + m);
       __m128 va3 = _mm_load1_ps(ajkm + 2 * m);
       __m128 va4 = _mm_load1_ps(ajkm + 3 * m);

       c1 = ci;
       a1 = ai;
       a2 = ai + m;
       a3 = ai + 2 * m;
       a4 = ai + 3 * m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         vc1 = _mm_loadu_ps(c1);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4), va4));
         _mm_storeu_ps(c1, vc1);

         vc1 = _mm_loadu_ps(c1 + 4);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 4), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 4), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 4), va4));
         _mm_storeu_ps(c1 + 4, vc1);

         vc1 = _mm_loadu_ps(c1 + 8);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 8), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 8), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 8), va4));
         _mm_storeu_ps(c1 + 8, vc1);

         vc1 = _mm_loadu_ps(c1 + 12);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 12), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 12), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 12), va4));
         _mm_storeu_ps(c1 + 12, vc1);

         vc1 = _mm_loadu_ps(c1 + 16);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 16), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 16), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 16), va4));
         _mm_storeu_ps(c1 + 16, vc1);

         vc1 = _mm_loadu_ps(c1 + 20);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 20), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 20), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 20), va4));
         _mm_storeu_ps(c1 + 20, vc1);

         vc1 = _mm_loadu_ps(c1 + 24);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 24), va2));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 24), va3));
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 24), va4));
         _mm_storeu_ps(c1 + 24, vc1);

         vc1 = _mm_loadu_ps(c1 + 28);
         c1 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1));
         a1 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2 + 28), va2));
         a2 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3 + 28), va3));
         a3 += 32;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4 + 28), va4));
         a4 += 32;
         _mm_storeu_ps(c1 - 4, vc1);

        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vc1 = _mm_loadu_ps(c1);
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a1), va1));
         a1 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a2), va2));
         a2 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a3), va3));
         a3 += 4;
         vc1 = _mm_add_ps(vc1, _mm_mul_ps(_mm_loadu_ps(a4), va4));
         a4 += 4;
         _mm_storeu_ps(c1, vc1);
         c1 += 4;
        }

       for (; i < m; i++)
        {
         float sum = (*a1) * (*(ajkm)); // THIS CALL MISSES A LOT
         sum += (*a2) * (*(ajkm + m));
         a1++;
         sum += (*a3) * (*(ajkm + 2 * m));
         a2++;
         sum += (*a4) * (*(ajkm + 3 * m));
         a3++;
         a4++;
         *(ci + i) += sum;
        }
      }

     for (; k < n; k++)
      {
       float* const ajkm = aj + k * m;
       __m128 vc1, va2, va3, va4, va5;
       __m128 va1 = _mm_load1_ps(ajkm);
       c1 = ci;
       a1 = A + k*m;

       int i;
       for (i = 0; i < (m / 32) * 32; i += 32)
        {
         // va2, va3, va4, va5 may be used as temp here
         vc1 = _mm_loadu_ps(c1);
         va2 = _mm_mul_ps(_mm_loadu_ps(a1), va1);
         vc1 = _mm_add_ps(vc1, va2);
         _mm_storeu_ps(c1, vc1);

         vc1 = _mm_loadu_ps(c1 + 4);
         va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 4), va1);
         vc1 = _mm_add_ps(vc1, va3);
         _mm_storeu_ps(c1 + 4, vc1);

         vc1 = _mm_loadu_ps(c1 + 8);
         va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 8), va1);
         vc1 = _mm_add_ps(vc1, va4);
         _mm_storeu_ps(c1 + 8, vc1);

         vc1 = _mm_loadu_ps(c1 + 12);
         va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 12), va1);
         vc1 = _mm_add_ps(vc1, va5);
         _mm_storeu_ps(c1 + 12, vc1);

         vc1 = _mm_loadu_ps(c1 + 16);
         va2 = _mm_mul_ps(_mm_loadu_ps(a1 + 16), va1);
         vc1 = _mm_add_ps(vc1, va2);
         _mm_storeu_ps(c1 + 16, vc1);

         vc1 = _mm_loadu_ps(c1 + 20);
         va3 = _mm_mul_ps(_mm_loadu_ps(a1 + 20), va1);
         vc1 = _mm_add_ps(vc1, va3);
         _mm_storeu_ps(c1 + 20, vc1);

         vc1 = _mm_loadu_ps(c1 + 24);
         va4 = _mm_mul_ps(_mm_loadu_ps(a1 + 24), va1);
         vc1 = _mm_add_ps(vc1, va4);
         _mm_storeu_ps(c1 + 24, vc1);

         vc1 = _mm_loadu_ps(c1 + 28);
         c1 += 32;
         va5 = _mm_mul_ps(_mm_loadu_ps(a1 + 28), va1);
         a1 += 32;
         vc1 = _mm_add_ps(vc1, va5);
         _mm_storeu_ps(c1 - 4, vc1);
        }

       for (; i < (m / 4) * 4; i += 4)
        {
         vc1 = _mm_loadu_ps(c1);
         va3 = _mm_loadu_ps(a1);
         a1 += 4;
         va4 = _mm_mul_ps(va3, va1);
         vc1 = _mm_add_ps(vc1, va4);
         _mm_storeu_ps(c1, vc1);
         c1 += 4;
        }

       for (; i < m; i++)
        {
         *(ci + i) += (*a1) * (*(ajkm));
         a1++;
        }
      }
    }
  }
}
