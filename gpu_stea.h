#ifndef STEA_H_
#define STEA_H_

#include <stdint.h>

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
/* Estructuras */
typedef struct {
    ulong k0, k1;
} STEA_KEY;

typedef struct{
    ulong m0, m1;
} STEA_BLOCK;

/*
 * m:    Bloque a Cifrar/Descifrar (64 bit)
 * k:    Key (64 bit)
 */

//Funciones para STEA
void stea_enc(ulong* m, ulong* k);
//void stea_dec(ulong* m, ulong* k);

  /*
  *	Implementacion de la encriptacion de STEA por
  *	Fauzan Mirza <fauzan.mirza@seecs.nust.edu.pk>
  */
void crack_stea (ulong [2], ulong [2], ulong [2]);
void stea_enc (ulong [2], ulong [2], ulong [2]);
void zero(ulong X[2]){
	X[0] = 0;
	X[1] = 0;
}

//GPU


#define STEA_ROUND(block,key) \
{ \
    (block).m0 += (block).m1 ^ (key).k0; \
    (block).m1 += (block).m0 ^ (key).k1; \
}

#define STEA_ROUND2(block,key) \
{ \
	(block).m1 -= (block).m0 ^ (key).k1; \
	(block).m0 -= (block).m1 ^ (key).k0; \
}
__global__ void gpu_stea_enc (STEA_BLOCK *m, STEA_KEY key){
    __shared__ STEA_BLOCK tmp_m;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int y;    
    tmp_m = m[idx];

	for(y = 0; y < 32; y++){
	    STEA_ROUND(tmp_m, key);
	}

  m[idx] = tmp_m;
}

__global__ void gpu_stea_dec (STEA_BLOCK *m, STEA_KEY key){
    STEA_BLOCK tmp_m;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    __syncthreads();
    tmp_m = m[idx];
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    STEA_ROUND2(tmp_m, key); STEA_ROUND2(tmp_m, key);
    m[idx] = tmp_m;
}
//Sirve para ser llamada por un kernel __global__
__device__ float gpu_stea_enc2 (STEA_BLOCK *m, STEA_KEY key){
    STEA_BLOCK tmp_m;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	int y;    
    tmp_m = m[idx];
	__syncthreads();	
	for(y = 0; y < 32; y++){
	    STEA_ROUND(tmp_m, key);
	}
    m[idx] = tmp_m;
	return 0;
}
// Encriptacion para STEA
__device__ float stea_enc2 (ulong P[2], ulong K[2], ulong C[2]){
	ulong temp;
	int i;
	C[0] = P[0], C[1] = P[1];
	for (i=0; i<64; i++)	{
		temp = C[0];
		C[0] = C[1];
		C[1] = temp + (C[1] ^ K[i & 1]);
	}
	return 1;
}

#endif
