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
  __m128 cv, tv1, tv2, tv3, tv4;
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
        cv = _mm_loadu_ps(cptr);
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
        _mm_storeu_ps(cptr, cv);
        aptr1 += 4;
        aptr2 += 4;
        aptr3 += 4;
        aptr4 += 4;
        cptr += 4;
        cv = _mm_loadu_ps(cptr);
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
        _mm_storeu_ps(cptr, cv);
        aptr1 += 4;
        aptr2 += 4;
        aptr3 += 4;
        aptr4 += 4;
        cptr += 4;
        cv = _mm_loadu_ps(cptr);
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
        cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
        _mm_storeu_ps(cptr, cv);
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
