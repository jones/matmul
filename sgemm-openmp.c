/* CS61C Project 3: Matrix Multiply Parallelization 
   Jian Wei Leong : cs61c-sh
   Tristan Jones  : cs61c-du 
*/

#include <emmintrin.h>

//#pragma GCC optimize (2,"unroll-all-loops", "fast-math","unsafe-loop-optimizations")
void sgemm( int m, int n, float *A, float *C )
{
  int i,j,k,ii;
  int ci,cj,ai,aj,ak,cil,cjl,cii;
  int bi=0;
  int mbtm = (m/12)*12;
  int nbtm = (n/4)*4;
  int mblocksize = 24; // Must be a multiple of ai (mblk)
  int nblocksize = 24; // Must be a multiple of aj (nblk)
  int mblkbtm = (m/mblocksize)*mblocksize;
  int nblkbtm = (n/nblocksize)*nblocksize;
  float* cptr;
  float* aptr1;
  float* aptr2;
  float* aptr3;
  float* aptr4;
  __m128 cv, tv1, tv2, tv3, tv4;
  // Let m = #rows, n = #cols, M = mblocksize, N = nblocksize
  for (cj=0; cj<nblkbtm; cj=cjl){ // Do normal Nx_-size blocks
    cjl = cj+nblocksize;
    for (ci=0; ci<mblkbtm; ci=cil){ // Do normal NxM-size blocks
      cil = ci+mblocksize;
      for (ak=0; ak<nbtm; ak+=4){ // Do up to 4k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          tv2 = _mm_load1_ps(A+aj+(ak+1)*m);
          tv3 = _mm_load1_ps(A+aj+(ak+2)*m);
          tv4 = _mm_load1_ps(A+aj+(ak+3)*m);
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i (guaranteed to complete if 12|M)
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
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
      for (; ak<n; ak++){ // Do remaining k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i (guaranteed to complete if 12|M)
            aptr1 = A+ai+ak*m;
            cptr = C+ai+aj*m;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
          }
        }
      }
    }
    for (; ci<m; ci=cil){ // Do edge Nx(m%M)-size block (there will be only 1 left)
      for (ak=0; ak<nbtm; ak+=4){ // Do up to 4k (guaranteed to complete if 4|N)
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          tv2 = _mm_load1_ps(A+aj+(ak+1)*m);
          tv3 = _mm_load1_ps(A+aj+(ak+2)*m);
          tv4 = _mm_load1_ps(A+aj+(ak+3)*m);
          cil = m-12;
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
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
          cil += 12;
          for (; ai<cil; ai++){ // Do remaining i
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
            _mm_storeu_ps(cptr, cv);
          }
        }
      }
    }
  }
  for (; cj<n; cj=cjl){ // Do remaining (n%N)x_-size block (there will be only 1 left)
    cjl = cj+nblocksize;
    for (ci=0; ci<mblkbtm; ci=cil){ // Do edge (n%N)xM-size blocks
      cil = ci+mblocksize;
      for (ak=0; ak<nbtm; ak+=4){ // Do up to 4k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          tv2 = _mm_load1_ps(A+aj+(ak+1)*m);
          tv3 = _mm_load1_ps(A+aj+(ak+2)*m);
          tv4 = _mm_load1_ps(A+aj+(ak+3)*m);
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i (guaranteed to complete if 12|M)
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
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
      for (; ak<n; ak++){ // Do remaining k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i (guaranteed to complete if 12|M)
            aptr1 = A+ai+ak*m;
            cptr = C+ai+aj*m;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
          }
        }
      }
    }
    for (; ci!=m; ci=cil){ // Do edge (n%N)x(m%M)-size block (there will be only 1 left)
      for (ak=0; ak<nbtm; ak+=4){ // Do up to 4k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          tv2 = _mm_load1_ps(A+aj+(ak+1)*m);
          tv3 = _mm_load1_ps(A+aj+(ak+2)*m);
          tv4 = _mm_load1_ps(A+aj+(ak+3)*m);
          cil = m-12;
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
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
          cil += 12;
          for (; ai<cil; ai++){ // Do remaining i
            aptr1 = A+ai+ak*m;
            aptr2 = A+ai+(ak+1)*m;
            aptr3 = A+ai+(ak+2)*m;
            aptr4 = A+ai+(ak+3)*m;
            cptr = C+ai+aj*m;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));
            _mm_storeu_ps(cptr, cv);
          }
        }
      }
      for (; ak<n; ak++){ // Do remaining k
        for (aj=cj; aj<cjl; aj++){
          tv1 = _mm_load1_ps(A+aj+ak*m);
          cil = m-12;
          for (ai=ci; ai<cil; ai+=12){ // Do up to 12i
            aptr1 = A+ai+ak*m;
            cptr = C+ai+aj*m;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
            aptr1 += 4;
            cptr += 4;
            cv = _mm_loadu_ps(cptr);
            cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));
            _mm_storeu_ps(cptr, cv);
          }
          cil += 12;
          for (; ai<cil; ai++){ // Do remaining i
            C[ai+aj*m] += A[ai+ak*m] * A[aj+ak*m];
          }
        }
      }
    }
  }
}
  /*for (cj=0; cj<nblkbtm; cj+=nblocksize){*/
    /*for (ci=0; ci<mblkbtm; ci+=mblocksize){*/
      /*for (ak=0; ak<nbtm; ak++){*/
        /*for (aj=0; aj<m; aj++){*/
          /*tv1 = _mm_load1_ps(A+aj+ak*m);*/
          /*for (ai=0; ai<mbtm; ai+=12){*/
            /*aptr1 = A+ai+ak*m;*/
            /*cptr = C+ci+cj*m;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
          /*}*/
        /*}*/
      /*}*/
    /*}*/
  /*}*/
  /*printf("Finished 1st loop for k=0->%d, j=0->%d, i=0->%d.\n",nbtm,cjl-1,cil-1);*/
  /*printf("Starting 2nd loop for k=0->%d, j=%d->%d, i=%d->%d.\n",nbtm,bj,m-1,bi,mbtm-1);*/
  /*for (; cj<n; cj++){*/
    /*for (; ci<m; ci++){*/
      /*for (ak=0; ak<nbtm; ak++){*/
        /*for (aj=0; aj<m; aj++){*/
          /*tv1 = _mm_load1_ps(A+aj+ak*m);*/
          /*for (ai=0; ai<mbtm; ai+=12){*/
            /*aptr1 = A+ai+ak*m;*/
            /*cptr = C+ci+cj*m;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
          /*}*/
        /*}*/
      /*}*/
    /*}*/
  /*}*/

  /*for (bj=0; bj<mblkbtm; bj=cjl){*/
    /*cjl = bj+blocksize;*/
    /*for (bi=0; bi<mblkbtm; bi=cil){*/
      /*cil = bi+blocksize;*/
      /*for (k=bk; k<bkl; k++){*/
        /*for (j=bj; j<cjl; j++){*/
          /*tv1 = _mm_load1_ps(A+j+k*m);*/
          /*for (i=bi; i<cil; i+=12){*/
            /*aptr1 = A+i+k*m;*/
            /*cptr = C+i+j*m;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
            /*aptr1 += 4;*/
            /*cptr += 4;*/
            /*cv = _mm_loadu_ps(cptr);*/
            /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
            /*_mm_storeu_ps(cptr, cv);*/
          /*}*/
        /*}*/
      /*}*/
    /*}*/
  /*}*/
  /*for (k=0; k<nbtm; k++){*/
    /*for (j=bj; j<m; j++){*/
      /*tv1 = _mm_load1_ps(A+j+k*m);*/
      /*for (i=bi; i<mbtm; i+=12){*/
        /*aptr1 = A+i+k*m;*/
        /*cptr = C+i+j*m;*/
        /*cv = _mm_loadu_ps(cptr);*/
        /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
        /*_mm_storeu_ps(cptr, cv);*/
        /*aptr1 += 4;*/
        /*cptr += 4;*/
        /*cv = _mm_loadu_ps(cptr);*/
        /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
        /*_mm_storeu_ps(cptr, cv);*/
        /*aptr1 += 4;*/
        /*cptr += 4;*/
        /*cv = _mm_loadu_ps(cptr);*/
        /*cv = _mm_add_ps(cv, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
        /*_mm_storeu_ps(cptr, cv);*/
      /*}*/
    /*}*/
  /*}*/

  /*__m128 cv1, tv1, tv2, tv3, tv4;*/
  /*__m128 bv1, av1, av2, av3, av4;*/

        /*aptr1 = A+i+k*m;*/
        /*aptr2 = A+i+(k+1)*m;*/
        /*aptr3 = A+i+(k+2)*m;*/
        /*aptr4 = A+i+(k+3)*m;*/
        /*cptr = C+i+j*m;*/
        /*cv1 = _mm_loadu_ps(cptr);*/
        /*av1 = _mm_loadu_ps(aptr1);*/
        /*av2 = _mm_loadu_ps(aptr2);*/
        /*av3 = _mm_loadu_ps(aptr3);*/
        /*av4 = _mm_loadu_ps(aptr4);*/
        /*av1 = _mm_mul_ps(av1, tv1);*/
        /*av2 = _mm_mul_ps(av2, tv2);*/
        /*av3 = _mm_mul_ps(av3, tv3);*/
        /*av4 = _mm_mul_ps(av4, tv4);*/
        /*bv1 = _mm_add_ps(cv1, av1);*/
        /*cv1 = _mm_add_ps(av2, av3);*/
        /*bv1 = _mm_add_ps(bv1, av4);*/
        /*cv1 = _mm_add_ps(bv1, cv1);*/
        /*_mm_storeu_ps(cptr, cv1);       */
        /*aptr1 += 4;*/
        /*aptr2 += 4;*/
        /*aptr3 += 4;*/
        /*aptr4 += 4;*/
        /*cptr += 4;*/
        /*bv1 = _mm_loadu_ps(cptr);*/
        /*bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr1),tv1));*/
        /*bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr2),tv2));*/
        /*bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr3),tv3));*/
        /*bv1 = _mm_add_ps(bv1, _mm_mul_ps(_mm_loadu_ps(aptr4),tv4));*/
        /*_mm_storeu_ps(cptr, bv1);*/
        /*aptr1 += 4;*/
        /*aptr2 += 4;*/
        /*aptr3 += 4;*/
        /*aptr4 += 4;*/
        /*cptr += 4;*/
        /*cv1 = _mm_loadu_ps(cptr);*/
        /*av1 = _mm_loadu_ps(aptr1);*/
        /*av2 = _mm_loadu_ps(aptr2);*/
        /*av3 = _mm_loadu_ps(aptr3);*/
        /*av4 = _mm_loadu_ps(aptr4);*/
        /*av1 = _mm_mul_ps(av1, tv1);*/
        /*av2 = _mm_mul_ps(av2, tv2);*/
        /*av3 = _mm_mul_ps(av3, tv3);*/
        /*av4 = _mm_mul_ps(av4, tv4);*/
        /*bv1 = _mm_add_ps(cv1, av1);*/
        /*cv1 = _mm_add_ps(av2, av3);*/
        /*bv1 = _mm_add_ps(bv1, av4);*/
        /*cv1 = _mm_add_ps(bv1, cv1);*/
        /*_mm_storeu_ps(cptr, cv1);*/
