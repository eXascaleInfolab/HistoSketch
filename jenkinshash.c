/*
 * jenkinshash.c
 * 
 * Matlab MEX implementation of Jenkins' one-at-a-time hash
 *
 * See http://en.wikipedia.org/wiki/Jenkins_hash_function
 *     http://www.burtleburtle.net/bob/hash/doobs.html
 *
 * Dan Ellis dpwe@ee.columbia.edu 2009-05-09
 */

#include "mex.h"
#include "string.h"
#include <math.h>

typedef unsigned int uint32_t;

#ifndef MWSIZE_MIN
/* matlab R17 -ism? */
typedef unsigned int mwSize;
#endif

uint32_t jenkins_one_at_a_time_hash(unsigned char *key, size_t key_len)
{
    uint32_t hash = 0;
    size_t i;
 
    for (i = 0; i < key_len; i++) {
        hash += key[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int 	i,j;
    long   	pvl, pvb[16];

    if (nrhs < 1){
	mexPrintf("jenkinshash  H = jenkinshash(X)  Jenkins one-at-a-time hash\n");
	mexPrintf("           Each row of X is a sequence of chars or integers, of which only\n");
	mexPrintf("           the lowest 8 bits are taken.  Each element of H is a 32 bit\n");
	mexPrintf("           integer hash based on the row in X.\n");
	return;
    }

    if (nlhs > 0){
	mxArray  *HMatrix;
	int rows, cols, i, j;
	uint32_t *ph;
	unsigned char *px;
	unsigned char *pd;
	mwSize siz;
	int bigendian;
	mxClassID clas;

	int ii = 1;
	if(*(char *)&ii == 1) {
	/*    mexPrintf("littleendian\n");  */
	    bigendian = 0;
	} else {
	/*    mexPrintf("bigendian\n");  */
	    bigendian = 1;
	}

	rows = mxGetM(prhs[0]);
	cols = mxGetN(prhs[0]);
	pd = (unsigned char *)mxGetData(prhs[0]);
	siz = mxGetElementSize(prhs[0]);
    clas = mxGetClassID(prhs[0]);

 	/* mexPrintf("rows=%d cols=%d type=%s\n", rows, cols, mxGetClassName(prhs[0])); */

	HMatrix = mxCreateNumericMatrix(rows, 1, mxUINT32_CLASS, mxREAL);

	ph = (uint32_t *)mxGetData(HMatrix);
	px = (unsigned char *)mxCalloc(cols, sizeof(unsigned char));


	if (clas == mxCHAR_CLASS || clas == mxINT8_CLASS || clas == mxUINT8_CLASS \
	    || clas == mxINT16_CLASS || clas == mxUINT16_CLASS || clas == mxINT32_CLASS \
	    || clas == mxUINT32_CLASS || clas == mxINT64_CLASS || clas == mxUINT64_CLASS) {

		for(i = 0; i < rows; ++i) {
		    for(j = 0; j < cols; ++j) {
				/* just looks at least-significant byte, run-time fix for bigendian */
				px[j] = pd[siz*(i + j*rows) + (siz-1)*bigendian];
		    }
		    ph[i] = jenkins_one_at_a_time_hash(px, cols);
		}
	} else if (clas == mxDOUBLE_CLASS) {
		double *pf = (double *)pd;
		for(i = 0; i < rows; ++i) {
		    for(j = 0; j < cols; ++j) {
				px[j] = (unsigned char)round(pf[(i + j*rows)]);
		    }
		    ph[i] = jenkins_one_at_a_time_hash(px, cols);
		}
	} else if (clas == mxSINGLE_CLASS) {
		float *pf = (float *)pd;
		for(i = 0; i < rows; ++i) {
		    for(j = 0; j < cols; ++j) {
				px[j] = (unsigned char)round(pf[(i + j*rows)]);
		    }
		    ph[i] = jenkins_one_at_a_time_hash(px, cols);
		}
	} else {
	   mexPrintf("error: type %d is not supported\n", clas);
	}

	plhs[0] = HMatrix;
	mxFree(px);
    }
}

