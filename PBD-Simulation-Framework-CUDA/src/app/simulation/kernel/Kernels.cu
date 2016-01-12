#include "Kernels.h"

#define M_PI 3.14159265359
#define deltaT 0.01f
#define restDensity 1.0f
#define kernelWidth 10.0f
#define cellKernelWidth kernelWidth/2
#define epsilon = 0.00001f

surface<void, cudaSurfaceType2D> positions4;
surface<void, cudaSurfaceType2D> predictedPositions4;
surface<void, cudaSurfaceType2D> velocities4;
surface<void, cudaSurfaceType2D> colors4;
surface<void, cudaSurfaceType2D> densities;

surface<void, cudaSurfaceType2D> positions4Copy;
surface<void, cudaSurfaceType2D> predictedPositions4Copy;
surface<void, cudaSurfaceType2D> velocities4Copy;
surface<void, cudaSurfaceType2D> colors4Copy;

unsigned int* d_cellIds_in;
unsigned int* d_cellIds_out;

unsigned int* d_particleIds_in;
unsigned int* d_particleIds_out;

float* d_lambdas;
float* lambdas;

float* position_deltas;

void* d_sortTempStorage = nullptr;
size_t sortTempStorageBytes = 0;

unsigned int* d_cellStarts;
unsigned int* d_cellEndings;

// TODO: GLÖM INTE ATT DEN SKA VARA SHARED 
float3 gradC = make_float3(0.0f, 0.0f, 0.0f);
// TODO: GLÖM INTE ATT DEN SKA VARA SHARED 
float global_density = 0.0f;
// TODO: GLÖM INTE ATT DEN SKA VARA SHARED 
float global_denominator = 0.0f;

// --------------------------------------------------------------------------

__device__ float poly6(float3 p1,
	float3 p2,
	float h,
	const unsigned int numberOfParticles,
	const unsigned int textureWidth)
{
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {

		float dist = length((p1 - p2));
		if (dist <= 0.0f)
		{
			return 0.0f;
		}

		if (dist <= h)
		{
			return 315.0f * pow(h*h - dist*dist, 3) / (64.0f * M_PI * pow(h, 9));
		}

	}
}

// ---------------------------------------------------------------------------

__device__ float3 spiky(float3 p1,
	float3 p2,
	float h,
	const unsigned int numberOfParticles,
	const unsigned int textureWidth)
{
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	float common = -1.0f;
	float3 v = p1 - p2;
	if (idx < numberOfParticles) {

		float dist = length((v));
		if (dist <= 0.0f)
		{
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		if (dist <= h)
			common = 45.0f* pow(h - dist, 2) / (dist * M_PI *pow(h, 6));
		else
			common = 0.0f;
	}
	return common*v;
}

__global__ void computeLambda(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	float* lambdas)
{
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	float inverseMass = 0.0f;

	float density = 0.0f;
	surf2Dread(densities, density, x, y);

	float4 pred_pos;
	surf2Dread(&predictedPositions4, pred_pos, x, y);
	float3 pred_pos3 = make_float3(pred_pos.x, pred_pos.y, pred_pos.z);


	float4 pred_pos_neigh;
	unsigned int part_id;
	unsigned int m;
	unsigned int n;
	float3 pred_pos_neigh3;

	// Kolla kanterna
	for (size_t i = -cellKernelWidth; i < cellKernelWidth; i++)
	{
		for (size_t j = -cellKernelWidth; j < cellKernelWidth; j++)
		{
			for (size_t k = -cellKernelWidth; k < cellKernelWidth; k++)
			{

				float4 cell_pos = make_float4(pred_pos.x + i, pred_pos.y + j, pred_pos.z + k, 0.0f);
				unsigned int cell_id = mortonCode(cell_pos);
				unsigned int part_id_start = d_cellStarts[cell_id];
				unsigned int part_id_end = d_cellEndings[cell_id];
				for (size_t q = part_id_start; q < part_id_end; q++)
				{
					part_id = d_particleIds_out[q];
					m = (part_id % textureWidth) * sizeof(int);
					n = part_id / textureWidth;

					surf2Dread(&predictedPositions4, pred_pos_neigh, m, n);
					pred_pos_neigh3 = make_float3(pred_pos_neigh);

					global_density += inverseMass*(poly6(pred_pos3, pred_pos_neigh3, kernelWidth, numberOfParticles, textureWidth) / restDensity) - 1;
					global_denominator += pow(length(spiky(pred_pos3, pred_pos_neigh3, kernelWidth, numberOfParticles, textureWidth)), 2) / restDensity;
				}
			}
		}
	}

	lambdas[idx] = -1 * (global_density) / (global_denominator * 2);



	//density = inverseMass*poly6()

}

void cudaCallComputeLambda() {

	auto glShared = GL_Shared::getInstance();
	const auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((*numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);


	computeLambda << < blocks, threads >> >(*numberOfParticles, textureWidth, d_lambdas);
	cudaDeviceSynchronize();

}


__global__ void computePositionDeltas(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	float* lambdas){


	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	float4 pred_pos;
	surf2Dread(&predictedPositions4, pred_pos, x, y);
	float3 pred_pos3 = make_float3(pred_pos.x, pred_pos.y, pred_pos.z);

	float4 pred_pos_neigh;
	unsigned int part_id;
	unsigned int m;
	unsigned int n;
	float3 pred_pos_neigh3;

	float lambda = lambdas[idx];
	float3 position_delta;

	// Kolla kanterna
	for (size_t i = -cellKernelWidth; i < cellKernelWidth; i++)
	{
		for (size_t j = -cellKernelWidth; j < cellKernelWidth; j++)
		{
			for (size_t k = -cellKernelWidth; k < cellKernelWidth; k++)
			{

				float4 cell_pos = make_float4(pred_pos.x + i, pred_pos.y + j, pred_pos.z + k, 0.0f);
				unsigned int cell_id = mortonCode(cell_pos);
				unsigned int part_id_start = d_cellStarts[cell_id];
				unsigned int part_id_end = d_cellEndings[cell_id];
				for (size_t q = part_id_start; q < part_id_end; q++)
				{
					part_id = d_particleIds_out[q];
					m = (part_id % textureWidth) * sizeof(int);
					n = part_id / textureWidth;

					surf2Dread(&predictedPositions4, pred_pos_neigh, m, n);
					pred_pos_neigh3 = make_float3(pred_pos_neigh);

					position_delta += (lambda + lambdas[part_id]) * spiky(pred_pos3, pred_pos_neigh3, kernelWidth, numberOfParticles, textureWidth);
				}
			}
		}
	}

	position_delta = position_delta / restDensity;

}

void cudaCallComputePositionDeltas() {

	auto glShared = GL_Shared::getInstance();
	const auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((*numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);


	computePositionDeltas << < blocks, threads >> >(*numberOfParticles, textureWidth, d_lambdas);
	cudaDeviceSynchronize();

}









// --------------------------------------------------------------------------

__global__ void applyForces(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	const float deltaT) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		const float inverseMass = 1.0f;
		const float gravity = -9.82;

		float4 velocity;
		surf2Dread(&velocity, velocities4, x, y);
		velocity.y += inverseMass * gravity * deltaT;

		float4 position;
		surf2Dread(&position, positions4, x, y);

		float4 predictedPosition = position + velocity * deltaT;
		surf2Dwrite(predictedPosition, predictedPositions4, x, y);
	}
}

void cudaCallApplyForces() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	applyForces << <blocks, threads >> >(numberOfParticles, textureWidth, deltaT);
}

// --------------------------------------------------------------------------

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the cube [0.0, 1023.0].
__device__ __forceinline__ unsigned int mortonCode(float4 pos) {
	pos.x = min(max(pos.x, 0.0f), 1023.0f);
	pos.y = min(max(pos.y, 0.0f), 1023.0f);
	pos.z = min(max(pos.z, 0.0f), 1023.0f);
	// x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	// y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	// z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	const unsigned int xx = expandBits((unsigned int)pos.x) << 2;
	const unsigned int yy = expandBits((unsigned int)pos.y) << 1;
	const unsigned int zz = expandBits((unsigned int)pos.z);
	//return xx * 4 + yy * 2 + zz;
	return xx + yy + zz;
}

__global__ void initializeCellIds(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	unsigned int* cellIdsIn) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		float4 predictedPosition;
		surf2Dread(&predictedPosition, predictedPositions4, x, y);
		cellIdsIn[idx] = mortonCode(predictedPosition);
	}
}

void cudaCallInitializeCellIds() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	initializeCellIds << <blocks, threads >> >(numberOfParticles, textureWidth, d_cellIds_in);
}

// --------------------------------------------------------------------------

void sortIds() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");

	cub::DeviceRadixSort::SortPairs(d_sortTempStorage,
		sortTempStorageBytes,
		d_cellIds_in,
		d_cellIds_out,
		d_particleIds_in,
		d_particleIds_out,
		numberOfParticles);
}

// --------------------------------------------------------------------------

__global__ void copy(const unsigned int numberOfParticles,
	const unsigned int textureWidth) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		float4 data;
		surf2Dread(&data, positions4, x, y);
		surf2Dwrite(data, positions4Copy, x, y);

		surf2Dread(&data, predictedPositions4, x, y);
		surf2Dwrite(data, predictedPositions4Copy, x, y);

		surf2Dread(&data, velocities4, x, y);
		surf2Dwrite(data, velocities4Copy, x, y);

		surf2Dread(&data, colors4, x, y);
		surf2Dwrite(data, colors4Copy, x, y);
	}
}

void cudaCallCopy() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	copy << <blocks, threads >> >(numberOfParticles, textureWidth);
}

// --------------------------------------------------------------------------

__global__ void reorder(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	unsigned int* cellIdsOut,
	unsigned int* particleIdsOut) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		const unsigned int cellId = cellIdsOut[idx];
		const unsigned int particleId = particleIdsOut[idx];

		//const unsigned int cellIdReadX = (cellId % textureWidth) * sizeof(float4);
		//const unsigned int cellIdReadY = cellId / textureWidth;

		const unsigned int particleIdX = (particleId % textureWidth) * sizeof(float4);
		const unsigned int particleIdY = particleId / textureWidth;


		float4 data;
		surf2Dread(&data, positions4Copy, particleIdX, particleIdY);
		surf2Dwrite(data, positions4, x, y);

		surf2Dread(&data, predictedPositions4Copy, particleIdX, particleIdY);
		surf2Dwrite(data, predictedPositions4, x, y);

		surf2Dread(&data, velocities4Copy, particleIdX, particleIdY);
		surf2Dwrite(data, velocities4, x, y);

		surf2Dread(&data, colors4Copy, particleIdX, particleIdY);
		surf2Dwrite(data, colors4, x, y);
	}
}

void cudaCallReorder() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	reorder << <blocks, threads >> >(numberOfParticles, textureWidth, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

void reorderStorage() {
	cudaCallCopy();
	cudaCallReorder();
}

// --------------------------------------------------------------------------

__global__ void resetCellInfo(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	unsigned int* cellStarts,
	unsigned int* cellEndings) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	cellStarts[idx] = UINT_MAX;
	cellEndings[idx] = 0;
}

void cudaCallResetCellInfo() {
	auto glShared = GL_Shared::getInstance();
	const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int maxGrid = *GL_Shared::getInstance().get_unsigned_int_value("maxGrid");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((maxGrid) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	resetCellInfo << <blocks, threads >> >(numberOfParticles, textureWidth, d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------

__global__ void computeCellInfo(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	unsigned int* cellStarts,
	unsigned int* cellEndings,
	unsigned int* cellIdsOut,
	unsigned int* particleIdsOut)  {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	const unsigned int cellId = cellIdsOut[idx];
	const unsigned int particleId = particleIdsOut[idx];

	if (idx < numberOfParticles) {
		if (idx == 0) {
			cellStarts[cellId] = 0;
		}
		else {
			const unsigned int previousCellId = cellIdsOut[idx - 1];
			if (previousCellId != cellId) {
				cellStarts[cellId] = idx;
				cellEndings[previousCellId] = idx;
			}
		}
	}
}

void cudaCallComputeCellInfo() {
	auto glShared = GL_Shared::getInstance();
	auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((*numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	computeCellInfo << <blocks, threads >> >(*numberOfParticles, textureWidth, d_cellStarts, d_cellEndings, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

__global__ void updatePositions(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	const float deltaT) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		float4 position;
		surf2Dread(&position, positions4, x, y);

		float4 predictedPosition;
		surf2Dread(&predictedPosition, predictedPositions4, x, y);

		float4 velocity = (predictedPosition - position) / deltaT;

		surf2Dwrite(predictedPosition, positions4, x, y);
		surf2Dwrite(velocity, velocities4, x, y);
	}
}

void cudaCallUpdatePositions() {
	auto glShared = GL_Shared::getInstance();
	const auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((*numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	updatePositions << <blocks, threads >> >(*numberOfParticles, textureWidth, deltaT);
}

// --------------------------------------------------------------------------

__global__ void initializeParticleIds(const unsigned int numberOfParticles,
	const unsigned int textureWidth,
	unsigned int* particleIdsIn) {
	const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
	const unsigned int x = (idx % textureWidth) * sizeof(float4);
	const unsigned int y = idx / textureWidth;

	if (idx < numberOfParticles) {
		particleIdsIn[idx] = idx;
	}
}

void cudaCallInitializeParticleIds() {
	auto glShared = GL_Shared::getInstance();
	const auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
	const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

	const dim3 blocks((*numberOfParticles) / 128, 1, 1);
	const dim3 threads(128, 1, 1);

	initializeParticleIds << <blocks, threads >> >(*numberOfParticles, textureWidth, d_particleIds_in);
}

// --------------------------------------------------------------------------

void initializeSort() {
	const unsigned int maxParticles = *GL_Shared::getInstance().get_unsigned_int_value("maxParticles");
	CUDA(cudaMalloc((void**)&d_cellIds_in, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_cellIds_out, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_in, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_out, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_lambdas, maxParticles * sizeof(float)));
	cudaCallInitializeParticleIds();


	CUDA(cudaMalloc(&d_sortTempStorage, sortTempStorageBytes));
}

// --------------------------------------------------------------------------

void initializeCellInfo() {
	const unsigned int maxGrid = *GL_Shared::getInstance().get_unsigned_int_value("maxGrid");
	CUDA(cudaMalloc((void**)&d_cellStarts, maxGrid * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_cellEndings, maxGrid * sizeof(unsigned int)));
}


void initializeTexture(surface<void, cudaSurfaceType2D>& surf, const std::string name) {
	auto glShared = GL_Shared::getInstance();
	GLuint gluint = glShared.get_texture(name)->texture_;

	cudaStream_t cudaStream;
	CUDA(cudaStreamCreate(&cudaStream));

	cudaGraphicsResource* resource;
	CUDA(cudaGraphicsGLRegisterImage(&resource,
		gluint,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore));

	CUDA(cudaGraphicsMapResources(1, &resource, cudaStream));

	cudaArray* array;
	CUDA(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

	CUDA(cudaBindSurfaceToArray(surf, array));

	CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
	CUDA(cudaStreamDestroy(cudaStream));
}
#define CUDA_INITIALIZE_SHARED_TEXTURE(name) initializeTexture(name, #name)


void cudaInitializeKernels() {
	CUDA_INITIALIZE_SHARED_TEXTURE(positions4);
	CUDA_INITIALIZE_SHARED_TEXTURE(predictedPositions4);
	CUDA_INITIALIZE_SHARED_TEXTURE(velocities4);
	CUDA_INITIALIZE_SHARED_TEXTURE(colors4);

	CUDA_INITIALIZE_SHARED_TEXTURE(positions4Copy);
	CUDA_INITIALIZE_SHARED_TEXTURE(predictedPositions4Copy);
	CUDA_INITIALIZE_SHARED_TEXTURE(velocities4Copy);
	CUDA_INITIALIZE_SHARED_TEXTURE(colors4Copy);

	initializeSort();
	initializeCellInfo();
}
