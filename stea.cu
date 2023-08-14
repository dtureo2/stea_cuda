/*
	Version: 03.11.2010
	STEA (Simple Tiny Encryption Algorithm).
	STEA sera ejecutado en la GPU. Para cpu ver stea.c o stea_copia.cu
	64 bit Plaintext. Estos serán creados en GPU.
	64 bit Key
	La clave (key) es leida desde key.stea
	El texto cifrado (cipher) es guardado en un archivo llamado ciphercu.stea
	Las funciones de cifrado y descifrado estan en gpu_stea.h

	Mide tiempos
	Compilar con 'clear & nvcc -o custea stea.cu'
	Alternativa 	'clear & nvcc -o gpustea -Xptxas "-v" -maxrregcount=10 stea.cu'
							10 = desired maximum registers / kernel
	gnuplot: plot 'times.tea' with points; set term png; set output 'times.png'; replot; set term x11
		f(x)=a*x+b; fit f(x) 'times.tea' via a,b
		plot 'times.tea' with points, f(x) ---
		plot 'times.tea' with points, f(x)=a*x+b; fit f(x) 'times.tea' via a,b
		gnuplot> plot 'times.tea' with points, f(x)
		gnuplot> set xlabel 'tiempos de CPU'
		gnuplot> set ylabel 'Rondas TEA'
		gnuplot> set title 'Fuerza Bruta sobre TEA'
		gnuplot> plot 'times.tea' title 'test'
		gnuplot> set term png; set output 'tiempos.png'; replot
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda/cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "gpu_stea.h" // funciona para GPU

#define stea_FILENAME "key.stea"
#define out_FILENAME "ciphercu.stea"
#define TIMES "gpu_times.txt"
#define RONDAS 32
#define THREADS_PER_THREADBLOCK 512 //original: 128
#define BLOCKBUFFER_SIZE (655360 * THREADS_PER_THREADBLOCK) // 256 = 2^8
//#define DATASIZE 1024 // siempre mayor que THREADS_PER_THREADBLOCK
#define DATASIZE 2048*2048*2 // siempre mayor que THREADS_PER_THREADBLOCK
#define NUMPT 0xFFFF00 // Para valores sobre 0x000F0000 la GPU es mas rapida
//Para valores mayores seria mejor ocupar una estructura

ulong plain[2]; 
ulong pt[NUMPT][2];
ulong key[2];
ulong key2[NUMPT][2];
ulong cipher[2];
char *text;

FILE *steafile, *outfile, *infile, *times;

void plaintexts(void);
void test_cuda(void);
int gpu_stea(uchar* , size_t , uchar* , STEA_KEY , int );
void gpu_plaintexts();
void plaintexts(void);
//GPU
__global__ void cuda_plaintext( ulong [NUMPT][2], ulong );


int main(){
	ulong count;
	int j, i;
	volatile double t1 = 0, t2;
	STEA_KEY keya;
    STEA_BLOCK *host_databuffer;
	cudaError_t ret;

	count  = NUMPT;
	steafile = fopen (stea_FILENAME, "r"); //lectura de la clave
	if (! steafile) {
		perror ("Error opening " stea_FILENAME);
		exit (2);
	}
	while( !feof(steafile) ){
		fscanf (steafile, "%08lX %08lX", &(key[0]), &(key[1]));
	}

	fclose (steafile);
	printf("Key:%08lX %08lX\n", key[0], key[1]);
	test_cuda(); // Indica si hay errores con la GPU
	t2 = clock();
	gpu_plaintexts(); //creacion de los textos planos
	//plaintexts(); //creacion de los textos planos
	t2 = (clock()-t2)/CLOCKS_PER_SEC;
	ret = cudaMallocHost((void**)(&host_databuffer), DATASIZE * sizeof(STEA_BLOCK));
	if (ret != cudaSuccess){
        printf("Failed to allocate page-locked buffer.\n");
        return EXIT_FAILURE;
    }
	
	printf(" \nt1 enc: %.15f \t t2 PT: %.15f \n", t1, t2);
	times = fopen(TIMES,"a");
	if (! times) {
		perror ("Error creating file " TIMES);
		exit (2);
	}
	fprintf(times,"%.15f\t%.15f\t%d\n", t1, t2, (int)NUMPT);
	fclose(times);

/* 	Rutina para el DESCIFRADO
	ver tea_cuda.cu 
*/
	return 1;
}

int gpu_stea(uchar* mensage, size_t len, uchar* output, STEA_KEY key, int op){
    void* gpu_databuffer;
    cudaEvent_t evt;
    size_t transfer_size, numBufferBlocks, numThreadBlocks;
    cudaError_t ret;
	//test_cuda();// TEST DE CUDA
    numBufferBlocks = len / sizeof(STEA_BLOCK); // lo mismo que DATASIZE
	//printf("\n  number of Buffer Blocks: %d  ", numBufferBlocks);
    if (numBufferBlocks <= 0) return 0;

    cudaMalloc(&gpu_databuffer, BLOCKBUFFER_SIZE * sizeof(STEA_BLOCK));
    while (numBufferBlocks > 0){
		//Nos aseguramos        
		transfer_size = numBufferBlocks > BLOCKBUFFER_SIZE ? BLOCKBUFFER_SIZE : numBufferBlocks;
		//transfer_size =  numBufferBlocks; // no nos aseguramos que lo tranferido sea igual a lo que hay en el buffer
        cudaMemcpy(gpu_databuffer, mensage, transfer_size*sizeof(STEA_BLOCK), cudaMemcpyHostToDevice);

        cudaEventCreate(&evt);
        numThreadBlocks = transfer_size / THREADS_PER_THREADBLOCK;// DATASIZE / THREAD....
		if(op == 0){
			gpu_stea_enc<<<numThreadBlocks, THREADS_PER_THREADBLOCK>>>((STEA_BLOCK *)gpu_databuffer, key);
		}else{
			gpu_stea_dec<<<numThreadBlocks, THREADS_PER_THREADBLOCK>>>((STEA_BLOCK *)gpu_databuffer, key);
		}
        // usleeping() while the kernel is running saves CPU cycles but may decrease performance
        if (cudaEventRecord(evt, NULL) == cudaSuccess)
            while (cudaEventQuery(evt) == cudaErrorNotReady) { usleep(1000); }
        cudaEventDestroy(evt);
        
        ret = cudaGetLastError();
        if (ret != cudaSuccess || cudaThreadSynchronize() != cudaSuccess){
            printf("Kernel failed to run. CUDA threw error message '%s'\n", cudaGetErrorString(ret));
            cudaFree(gpu_databuffer);
            return 0;
        }

        cudaMemcpy(output, gpu_databuffer, transfer_size * sizeof(STEA_BLOCK), cudaMemcpyDeviceToHost);
        
        mensage += transfer_size * sizeof(STEA_BLOCK);
        output += transfer_size * sizeof(STEA_BLOCK);
        numBufferBlocks -= transfer_size;
    }
//	printf("\n");
    cudaFree(gpu_databuffer);
    return 1;
}

/*
	Creacion Plaintexts con GPU
*/
void gpu_plaintexts(){
	ulong num_pt = NUMPT;
	ulong (*gpu_pt)[2];
	int *gpu_band;//, *band;
	cudaError_t ret;
	cudaEvent_t evt;
	size_t transfer_size, numBufferBlocks, numThreadBlocks;
	volatile double t1 = 0;
	
	numBufferBlocks = (ulong) num_pt * 2;
	transfer_size = numBufferBlocks > BLOCKBUFFER_SIZE ? BLOCKBUFFER_SIZE : numBufferBlocks;
	cudaMalloc((void**) &gpu_band, sizeof(int) * 2);
	printf("Creando Textos en Claro con GPU\n");
	ret = cudaMalloc((void**) &gpu_pt, sizeof(ulong)* numBufferBlocks);
	if (ret != cudaSuccess){
        printf("Failed to allocate page-locked buffer.\n");
        return ;
    }
	cudaMemcpy(gpu_pt, pt, sizeof(ulong) * numBufferBlocks, cudaMemcpyHostToDevice);
	numThreadBlocks = transfer_size / THREADS_PER_THREADBLOCK;
	cudaEventCreate(&evt);
	cuda_plaintext<<<numThreadBlocks, THREADS_PER_THREADBLOCK>>>(gpu_pt, num_pt);
	
	if (cudaEventRecord(evt, NULL) == cudaSuccess)
            while (cudaEventQuery(evt) == cudaErrorNotReady) { usleep(1000); }
	cudaEventDestroy(evt);    
	ret = cudaGetLastError();
	if (ret != cudaSuccess || cudaThreadSynchronize() != cudaSuccess){
		printf("Kernel failed to run. CUDA threw error message '%s'\n", cudaGetErrorString(ret));
		cudaFree(gpu_pt);
		return;
	}
	cudaMemcpy(pt, gpu_pt, sizeof(ulong) * transfer_size, cudaMemcpyDeviceToHost);
	cudaFree(gpu_pt);
	printf("Creados %d textos en claro\n\n", (int)num_pt);
}


__global__ void cuda_plaintext( ulong cu_pt[NUMPT][2], ulong Cantidad){
	__shared__ uint threadN;
	ulong tid = blockDim.x + blockIdx.x + threadIdx.x;
	threadN = blockDim.x + gridDim.x;
	__shared__ ulong cero;
	__shared__ ulong k;
	k = tid;
	cero = 0x0;
	cu_pt[k][0] = cero;
	for(uint pos = k; pos < Cantidad; pos += threadN){
		cu_pt[pos][1] = pos;
	}
}

// Se encarga de crear los plaintexts que diga NUMPT
void plaintexts(){
	ulong i, num_pt;
	ulong plaintext[2];
	num_pt = NUMPT;
	plaintext[0] = 0x00000000;
	for(i = 0x0; i < num_pt; i++){
		plaintext[1] = i;
		pt[i][0] = plaintext[0];
		pt[i][1] = plaintext[1];
	}
	printf("Creados %d textos en claro\n\n", (int)num_pt);
}

void test_cuda(){
    cudaError_t ret;
    struct cudaDeviceProp cuda_devprop;
	int cudadev, cudadevcount;
	cudaGetDeviceCount(&cudadevcount);
    
	ret = cudaGetLastError();
    if (ret != cudaSuccess){
        printf("Error en la Tarjeta'%s'\n", cudaGetErrorString(ret));
        return;
    }
    
    printf("  STEA sobre CUDA con %i GPU:\n", cudadevcount);
    for (cudadev = 0; cudadev < cudadevcount; cudadev++){
        cudaGetDeviceProperties(&cuda_devprop, cudadev);
        printf("(%i) '%s'\n\n", cudadev, (char *)&cuda_devprop.name);
    }
    cudaGetDevice(&cudadev);
    if (ret != cudaSuccess){
        printf("Error en la Tarjeta.\n");
        return;
    }
    return;
}
