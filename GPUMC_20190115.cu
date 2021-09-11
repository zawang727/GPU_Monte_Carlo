#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

struct particleposition
{
  double x;
  double y;
  double z;
};

struct particlepositionnext
{
  double x;
  double y;
  double z;
};

struct particlemove_sph
{
  double r;
  double theda;
  double phi;
};

__device__ void position_random_generation (struct particleposition &GPU_pos)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  curandState localState;
  curand_init(37,i, 0, &localState);
  while (true)
	  {
		GPU_pos.x=curand_uniform_double(&localState)-0.5;//sephere radius=0.5 cm
		GPU_pos.y=curand_uniform_double(&localState)-0.5;
		GPU_pos.z=curand_uniform_double(&localState)-0.5;
		if (GPU_pos.x*GPU_pos.x+GPU_pos.y*GPU_pos.y+GPU_pos.z*GPU_pos.z<0.25)
		{break;}
	  }
}

__device__ void particlemove_sphcor_generation (struct particlemove_sph &GPU_movesph)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  curandState localState;
  curand_init(17,i, 0, &localState);
  GPU_movesph.theda=curand_uniform_double(&localState);
  GPU_movesph.theda=GPU_movesph.theda*2-1;
  GPU_movesph.phi=curand_uniform_double(&localState);
  GPU_movesph.phi=GPU_movesph.phi*2*3.1415926536;
  GPU_movesph.r=curand_uniform_double(&localState);
  GPU_movesph.r=1/3.443*-logf(GPU_movesph.r);//r=1/sigma_total*-log(random 0~1)
}

__device__ void particlenextpos (struct particleposition &GPU_pos,struct particleposition &posnew,struct particlemove_sph &GPU_movesph)
{
  posnew.x=GPU_pos.x+GPU_movesph.r*GPU_movesph.theda*cosf(GPU_movesph.phi);
  posnew.y=GPU_pos.y+GPU_movesph.r*GPU_movesph.theda*sinf(GPU_movesph.phi);
  posnew.z=GPU_pos.z+GPU_movesph.r*sinf(acosf(GPU_movesph.theda));
}

__global__ void particlefunc(int *GPU_N,int *absorp_thread,int *scatter_thread,double *track_thread,struct particleposition *GPU_pos,struct particleposition *GPU_posnext,struct particlemove_sph *GPU_movesph,int *stp)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  double absorpportion=0;
  int scatcounter=0;
  if (i<=GPU_N[0])
  {
  curandState localState;
  curand_init(37,i, 0, &localState);
  /*generate start point*/
  stp[i]=1;
  //position_random_generation(GPU_pos[i]);
  GPU_pos[i].x=0;
  GPU_pos[i].y=0;
  GPU_pos[i].z=0;
  /*generate angle and length*/
  while (true)
  {
	  particlemove_sphcor_generation (GPU_movesph[i]);
	  /*calculate new position*/
	  particlenextpos (GPU_pos[i],GPU_posnext[i],GPU_movesph[i]);
	  /*check position*/
	  if (GPU_posnext[i].x*GPU_posnext[i].x+GPU_posnext[i].y*GPU_posnext[i].y+GPU_posnext[i].z*GPU_posnext[i].z<=0.25)/*interact in sphere*/
	  {
		  track_thread[i]=sqrtf((GPU_posnext[i].x-GPU_pos[i].x)*(GPU_posnext[i].x-GPU_pos[i].x)\
		  +(GPU_posnext[i].y-GPU_pos[i].y)*(GPU_posnext[i].y-GPU_pos[i].y)\
		  +(GPU_posnext[i].z-GPU_pos[i].z)*(GPU_posnext[i].z-GPU_pos[i].z));
		  //divide into scattering and absorption
		  absorpportion=curand_uniform_double(&localState);
		  if (absorpportion<0.0064)//absorp XS portion
		  {
			  absorp_thread[i]=1;
			  break;
		  }
		  else
		  {
			  scatter_thread[i]=1;
			  GPU_pos[i].x=GPU_posnext[i].x;
			  GPU_pos[i].y=GPU_posnext[i].y;
			  GPU_pos[i].z=GPU_posnext[i].z;
		  }
	  }/*if end*/
	  else/*die out of sphere*/
	  {
		break;
	  }/*else end*/
	  if (scatcounter>=30) {break;}
	  scatcounter=scatcounter+1;
  }
  
  }/*if end*/
  __syncthreads();
}

int main(int argc, char *argv[])
{
  int N=0,absorption=0, scattering=0;
  double track=0,trackflux;
  int* GPU_N;
  int* GPU_absorption;
  int* GPU_scattering;
  double* GPU_track;
  struct particleposition *pos;
  struct particleposition *posnext;
  struct particlemove_sph *move_sph;
  int* GPU_stp;
  int intsize=sizeof(int);
  int doublesize=sizeof(double);
  int* absorpar;
  int* scatteringar;
  double* trackar;
  cudaEvent_t start,stop;
  float time;
  /*user input*/
  N=1000000;
  printf("Simulate %d particles\n",N);
  /*calculate block and thread number*/
  int blockamount=0;
  blockamount=N/1024+1;
  printf("block %d \n",blockamount);
  /*host array space declaration*/
  absorpar=(int*) malloc(intsize*N);
  scatteringar=(int*) malloc(intsize*N);
  trackar=(double*) malloc(doublesize*N);
  /*cuda computation*/  
  /*decleare menory on device*/
  cudaMalloc((void**) &GPU_N, intsize);
  cudaMalloc((void**) &GPU_absorption, intsize*N);
  cudaMalloc((void**) &GPU_scattering, intsize*N);
  cudaMalloc((void**) &GPU_track, doublesize*N);
  cudaMalloc((void**) &pos, sizeof(particleposition)*N);
  cudaMalloc((void**) &posnext, sizeof(particleposition)*N);
  cudaMalloc((void**) &move_sph, sizeof(particlemove_sph)*N);;
  cudaMalloc((void**) &GPU_stp, intsize*N);
  /*copy data to device*/
  cudaMemcpy(GPU_N,&N,intsize,cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_absorption,absorpar,intsize*N,cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_scattering,scatteringar,intsize*N,cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_track,trackar,doublesize*N,cudaMemcpyHostToDevice);
  /*launch GPU computation*/
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  particlefunc<<<blockamount,1024>>> (GPU_N,GPU_absorption,GPU_scattering,GPU_track,pos,posnext,move_sph,GPU_stp);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("run time=%f ms\n",time);
  /*copy results*/
  cudaMemcpy(scatteringar,GPU_scattering,intsize*N,cudaMemcpyDeviceToHost);
  cudaMemcpy(absorpar,GPU_absorption,intsize*N,cudaMemcpyDeviceToHost);
  cudaMemcpy(trackar,GPU_track,doublesize*N,cudaMemcpyDeviceToHost);
  /*free memories on GPU*/
  cudaFree(GPU_N);
  cudaFree(GPU_absorption);
  cudaFree(GPU_scattering);
  cudaFree(GPU_track);
  cudaFree(move_sph);
  cudaFree(posnext);
  cudaFree(pos);
  cudaFree(GPU_stp);
  /*sum all of the particles*/
  for (int i=0;i<N;++i)
	{
		scattering=scattering+scatteringar[i];
		absorption=absorption+absorpar[i];
		track=track+trackar[i];
	}
  printf ("escape probability=%lf\n",(double)(N-absorption)/N);
  printf ("absorption=%d\n",absorption);
  printf ("scattering=%d\n",scattering);
  printf ("track=%lf\n",track);
  trackflux=track/N;
  printf ("track length estimated flux=%lf\n",trackflux);
  system("PAUSE");	
  return 0;
}
