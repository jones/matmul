/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-
   Tristan Jones  : cs61c-du 
*/

#include <emmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
  int i,j,k,ii;
  int mbtm = (m/12)*12;
  int nbtm = (n/4)*4;
  float* cptr;
  float* aptr1;
  float* aptr2;
  float* aptr3;
  float* aptr4;
  __m128 cv1, tv1, tv2, tv3, tv4;
  __m128 bv1, av1, av2, av3, av4;
  for (k=0; k<nbtm; k+=4){
    for (j=0; j<m; j++){
      tv1 = _mm_load1_ps(A+j+k*m);
      tv2 = _mm_load1_ps(A+j+(k+1)*m);
      tv3 = _mm_load1_ps(A+j+(k+2)*m);
      tv4 = _mm_load1_ps(A+j+(k+3)*m);
      for (i=0; i<mbtm; i+=12){
        aptr1 = A+i+k*m;
        aptr2 = A+i+(k+1)*m;
        aptr3 = A+i+(k+2)*m;
        aptr4 = A+i+(k+3)*m;
        cptr = C+i+j*m;
		
		cv1 = _mm_loadu_ps(cptr);
		av1 = _mm_loadu_ps(aptr1);
		av2 = _mm_loadu_ps(aptr2);
		av3 = _mm_loadu_ps(aptr3);
		av4 = _mm_loadu_ps(aptr4);
		av1 = _mm_mul_ps(av1, tv1);
		av2 = _mm_mul_ps(av2, tv2);
		av3 = _mm_mul_ps(av3, tv3);
		av4 = _mm_mul_ps(av4, tv4);
		bv1 = _mm_add_ps(cv1, av1);
		cv1 = _mm_add_ps(av2, av3);
		bv1 = _mm_add_ps(bv1, av4);
		cv1 = _mm_add_ps(bv1, cv1);
        _mm_storeu_ps(cptr, cv1);       
        aptr1 += 4;
        aptr2 += 4;
        aptr3 += 4;
        aptr4 += 4;
        cptr += 4;
		
        bv1 = _mm_loadu_ps(cptr);
        bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
        bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
        bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
        bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
        _mm_storeu_ps(cptr, bv1);
        aptr1 += 4;
        aptr2 += 4;
        aptr3 += 4;
        aptr4 += 4;
        cptr += 4;
		
        cv1 = _mm_loadu_ps(cptr);
		av1 = _mm_loadu_ps(aptr1);
		av2 = _mm_loadu_ps(aptr2);
		av3 = _mm_loadu_ps(aptr3);
		av4 = _mm_loadu_ps(aptr4);
		av1 = _mm_mul_ps(av1, tv1);
		av2 = _mm_mul_ps(av2, tv2);
		av3 = _mm_mul_ps(av3, tv3);
		av4 = _mm_mul_ps(av4, tv4);
		bv1 = _mm_add_ps(cv1, av1);
		cv1 = _mm_add_ps(av2, av3);
		bv1 = _mm_add_ps(bv1, av4);
		cv1 = _mm_add_ps(bv1, cv1);
        _mm_storeu_ps(cptr, cv1);
      }
    }
  }
  for (k=nbtm; k<n; k++){
    for (j=0; j<m; j++){
      tv1 = _mm_load1_ps(A+j+k*m);
      for (i=0; i<mbtm; i+=12){
        aptr1 = A+i+k*m;
        cptr = C+i+j*m;
        _mm_storeu_ps(cptr, _mm_add_ps(_mm_loadu_ps(cptr), _mm_mul_ps(_mm_loadu_ps(aptr1),tv1)));
        aptr1 += 4;
        cptr += 4;
        _mm_storeu_ps(cptr, _mm_add_ps(_mm_loadu_ps(cptr), _mm_mul_ps(_mm_loadu_ps(aptr1),tv1)));
        aptr1 += 4;
        cptr += 4;
        _mm_storeu_ps(cptr, _mm_add_ps(_mm_loadu_ps(cptr), _mm_mul_ps(_mm_loadu_ps(aptr1),tv1)));
      }
    }
  }
  for (k=0; k<n; k++){
    for (j=0; j<m; j++){
      for (ii=i; ii<m; ii++){
        C[ii+j*m] += A[ii+k*m] * A[j+k*m];
      }
    }
  }
}