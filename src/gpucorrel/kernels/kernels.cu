/*
This file defines kernels and textures used in TechCorrel
to perform GPU-accelerated correlation with pyCUDA.
All the kernels must be call with a block*grid size superior or
equal to the dimension of the output
ex: for a 2048*2048 image, block=(32,32,1) grid=(64,64)
*/

#include <cuda.h>
#define WIDTH %d
#define HEIGHT %d
#define PARAMETERS %d // The number of fields

// Original image
texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
// Second image
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_d;
// The mask, to limit the effect of the borders
texture<float, cudaTextureType2D, cudaReadModeElementType> texMask;

// This kernel computes the gradients of the reference image with a Sobel filter
// Note: the outputs are not normalized
__global__ void gradient(float* gradX, float* gradY)
{
  const unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x < WIDTH && y < HEIGHT)
  {
    gradX[x+WIDTH*y] = (
        tex2D(tex, (x+1.5f)/WIDTH, (float)y/HEIGHT)
        +tex2D(tex, (x+1.5f)/WIDTH, (y+1.f)/HEIGHT)
        -tex2D(tex, (x-.5f)/WIDTH, (float)y/HEIGHT)
        -tex2D(tex, (x-.5f)/WIDTH, (y+1.f)/HEIGHT)
         );
    gradY[x+WIDTH*y] = (
        tex2D(tex, (float)x/WIDTH, (y+1.5f)/HEIGHT)
        +tex2D(tex, (x+1.f)/WIDTH, (y+1.5f)/HEIGHT)
        -tex2D(tex, (float)x/WIDTH, (y-.5f)/HEIGHT)
        -tex2D(tex, (x+1.f)/WIDTH, (y-.5f)/HEIGHT)
        );
  }
}

// Kernel to resample the reference image using bilinear interpolation
__global__ void resampleR(float* out, const int w, const int h)
{
  const int idx = threadIdx.x+blockIdx.x*blockDim.x;
  const int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
    out[idx+w*idy] = tex2D(tex,(float)idx/w,(float)idy/h);
}

// To resample the second image...
__global__ void resample(float* out, const int w, const int h)
{
  const int idx = threadIdx.x+blockIdx.x*blockDim.x;
  const int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
    out[idx+w*idy] = tex2D(tex_d,(float)idx/w,(float)idy/h);
}

// This kernel computes the tables that will be used by the correlation
// routine to evaluate the research direction (called G, 1 per parameter)
__global__ void makeG(float* G, const float* gradX, const float* gradY,
                      const float* fieldX, const float* fieldY)
{
  const int idx = threadIdx.x+blockIdx.x*blockDim.x;
  const int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < WIDTH && idy < HEIGHT)
  {
    int id = idx+WIDTH*idy;
    G[id] = gradX[id]*fieldX[id]+gradY[id]*fieldY[id];
  }
}

// The kernel that will write the residual image (the difference between the
// reference image and the second image after deformation)
__global__ void makeDiff(float *out, const float *param,
                         const float *fieldsX, const float *fieldsY)
{
  const int idx = threadIdx.x+blockIdx.x*blockDim.x;
  const int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < WIDTH && idy < HEIGHT)
  {
    const int id = idx+WIDTH*idy;
    float ox = .5f;
    float oy = .5f;
    // First, let's compute the offset we have by adding all the fields
    for(unsigned int i = 0; i < PARAMETERS; i++)
    {
      ox += param[i]*fieldsX[WIDTH*HEIGHT*i+id];
      oy += param[i]*fieldsY[WIDTH*HEIGHT*i+id];
    }
    // The residual in idx,idy is the value of the reference image
    // minus the value of the second image at the new coordinates
    // We multiply this difference by the mask
    out[id] = (
    tex2D(tex,(idx+.5f)/WIDTH,(idy+.5f)/HEIGHT)
    -tex2D(tex_d,(idx+ox)/WIDTH,(idy+oy)/HEIGHT)
    )*tex2D(texMask,idx+.5f,idy+.5f);
  }
}

// Simple matrix-vector dot product (to multiply the inverted Hessian with
// the research direction)
__global__ void myDot(const float *M, float *v)
{
  unsigned int id = threadIdx.x;
  __shared__ float sh_v[PARAMETERS];
  float val = 0;
  sh_v[id] = v[id];
  __syncthreads();
  for(unsigned int i = 0; i < PARAMETERS; i++)
  {
    val += M[id*PARAMETERS+i]*sh_v[i];
  }
  v[id] = val;
}

// Do I really need to explain this one ?
__global__ void kadd(float* v, const float k, const float* v2)
{
  v[threadIdx.x] += k*v2[threadIdx.x];
}
