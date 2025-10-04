#include "pathtrace.h"

#include <cstdio>
#include <climits>
#include <cuda.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "octree.h"
#include "tiny_gltf.h"

// Controls
#define STREAM_COMPACTION 1
#define SORT_MATERIALS 1
#define OCTREE 1
#define RUSSIAN_ROULETTE 0
// Only start russianroulette after a certain number of bounces
#define BOUNCES_TO_START_RUSSIAN_ROULETTE 2
// min and max values for the termination probability q
#define MIN_Q 0.05
#define MAX_Q 0.70

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

// texture helpers
__device__ __host__ inline void barycentrics(
  const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
  float& x, float& y, float& z)
{
  // convert barycentric coordinates for triangle texturing
  glm::vec3 edge0 = b - a;
  glm::vec3 edge1 = c - a;
  glm::vec3 edge2 = p - a;

  // bary dot products
  float d00 = glm::dot(edge0, edge0);
  float d01 = glm::dot(edge0, edge1);
  float d11 = glm::dot(edge1, edge1);
  float d20 = glm::dot(edge2, edge0);
  float d21 = glm::dot(edge2, edge1);
  float denom = d00 * d11 - d01 * d01;

  if (fabsf(denom) < 1e-20f) { 
    // weird tri don't divide
    x = 1.f; 
    y = z = 0.f; 
    return; 
  }
  
  // compute correct vals
  float v = (d11 * d20 - d01 * d21) / denom;
  float w = (d00 * d21 - d01 * d20) / denom;
  float u = 1.0f - v - w;
  x = u; 
  y = v; 
  z = w;
}

// convert to linear color space
__device__ inline float srgbToLinear(float c) {
  return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
}

// wrap uv coords to 0, 1 just default wrap
__device__ inline glm::vec2 wrapCoord(glm::vec2 uv, int wrap) {
  float newuvx = uv.x - floorf(uv.x);
  float newuvy = uv.y - floorf(uv.y);

  return glm::vec2(newuvx, newuvy);
}

__device__ inline glm::vec4 bilinearTextureSample(const GPUTexture* texture, int textureid, glm::vec2 uv) {
  // check valid texture
  if (textureid < 0) {
    return glm::vec4(1, 1, 1, 1);
  }
  const GPUTexture& t = texture[textureid];
  if (t.width <= 0 || t.height <= 0 || t.rgba == nullptr) {
    return glm::vec4(1, 1, 1, 1);
  }

  // wrap UV coords
  glm::vec2 u = wrapCoord(uv, t.wrapS);
  glm::vec2 v = wrapCoord(glm::vec2(u.y, uv.y), t.wrapT);
  u.y = v.y;

  // set upbilinear sampling bounds
  float x = u.x * (t.width - 1);
  float y = u.y * (t.height - 1);
  int x0 = floorf(x);
  int y0 = floorf(y);
  int x1 = min(x0 + 1, t.width - 1);
  int y1 = min(y0 + 1, t.height - 1);
  float fx = x - x0;
  float fy = y - y0;

  // get pixel colors from texture
  auto lambda = [&](int ix, int iy)->glm::vec4 {
    const unsigned char* p = t.rgba + 4 * (iy * t.width + ix);
    float r = p[0] / 255.f, g = p[1] / 255.f, b = p[2] / 255.f, a = p[3] / 255.f;
    return glm::vec4(r, g, b, a);
    };
  glm::vec4 c00 = lambda(x0, y0);
  glm::vec4 c10 = lambda(x1, y0);
  glm::vec4 c01 = lambda(x0, y1);
  glm::vec4 c11 = lambda(x1, y1);

  // do bilinear sample
  glm::vec4 c0 = glm::vec4(
    c00.x + fx * (c10.x - c00.x),
    c00.y + fx * (c10.y - c00.y),
    c00.z + fx * (c10.z - c00.z),
    c00.w + fx * (c10.w - c00.w)
  );
  glm::vec4 c1 = glm::vec4(
    c01.x + fx * (c11.x - c01.x),
    c01.y + fx * (c11.y - c01.y),
    c01.z + fx * (c11.z - c01.z),
    c01.w + fx * (c11.w - c01.w)
  );
  glm::vec4 c = glm::vec4(
    c0.x + fy * (c1.x - c0.x),
    c0.y + fy * (c1.y - c0.y),
    c0.z + fy * (c1.z - c0.z),
    c0.w + fy * (c1.w - c0.w)
  );
  return c;
}

// actualy get the texture color at the uv coords
__device__ inline glm::vec3 getLinearTextureColor(const GPUTexture* texArr, int texId, glm::vec2 uv) {
  // do bilinear sample
  glm::vec4 c = bilinearTextureSample(texArr, texId, glm::vec2(uv.x, uv.y));
  // convert color  to linear color space
  return glm::vec3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int numNodes = 0;
static int numGeoms = 0;
static int numPrimIndices = 0;

static GPUTexture* dev_textures = nullptr;
static std::vector<unsigned char*> texturesToFree;


// TODO: static variables for device memory, any extra info you need, etc
// ...

// For sorting by material id
static int* dev_materialKeys = NULL;

// For octree
OctreeNode* dev_nodes = nullptr;
int* dev_primIndices = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    numGeoms = scene->geoms.size();

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));
    cudaMemset(dev_materialKeys, 0, pixelcount * sizeof(int));

    if (OCTREE) {
      scene->octree.sendToDevice();
      dev_nodes = scene->octree.dev_nodes;
      numNodes = scene->octree.octreeNodes.size();
      dev_primIndices = scene->octree.dev_primIndices;
      numPrimIndices = scene->octree.octreePrimIndices.size();
    }

    // Textures
    // upload textures to gpu
    if (!scene->textures.empty()) {
      std::vector<GPUTexture> gpuTextures(scene->textures.size());
      texturesToFree.clear(); 
      texturesToFree.reserve(scene->textures.size());

      for (size_t i = 0; i < scene->textures.size(); ++i) {
        const CPUTexture& ct = scene->textures[i];
        gpuTextures[i].width = ct.width;
        gpuTextures[i].height = ct.height;
        // standard wrap type
        gpuTextures[i].wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT;
        gpuTextures[i].wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;

        unsigned char* dev_pixels = nullptr;
        size_t bytes = (size_t)ct.width * ct.height * 4;
        cudaMalloc(&dev_pixels, bytes);
        cudaMemcpy(dev_pixels, ct.rgba.data(), bytes, cudaMemcpyHostToDevice);
        gpuTextures[i].rgba = dev_pixels;
        texturesToFree.push_back(dev_pixels);
      }

      cudaMalloc(&dev_textures, gpuTextures.size() * sizeof(GPUTexture));
      cudaMemcpy(dev_textures, gpuTextures.data(), gpuTextures.size() * sizeof(GPUTexture), cudaMemcpyHostToDevice);
    }


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    cudaFree(dev_materialKeys);
    cudaFree(dev_nodes);
    cudaFree(dev_primIndices);

    // free texture pixels an d textures
    for (auto* pixels : texturesToFree) {
      cudaFree(pixels);
    }
    texturesToFree.clear();
    cudaFree(dev_textures);
    dev_textures = nullptr;


    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
        float jitterX = u01(rng);
        float jitterY = u01(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitterX)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitterY)
        );


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.bouncesCompleted = 0;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections, 
    OctreeNode* nodes,
    int* primIndices,
    int numNodes,
    int numPrimIndices,
    int numGeoms
  
  )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        
        if (OCTREE) {
          octreeCollisionDetection(pathSegment.ray, nodes, numNodes, primIndices, numPrimIndices, geoms, numGeoms, hit_geom_index, t_min, intersect_point, normal);
          if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
            // set materialId to INT_MAX so whenn sorting by material is put at the end
            intersections[path_index].materialId = INT_MAX;
          }
          else {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;

            // default to no UV coords
            intersections[path_index].uv = glm::vec2(0.0f);
            intersections[path_index].hasUV = 0;

            // If hit a triangle compute UV
            const Geom& g = geoms[hit_geom_index];
            if (g.type == TRIANGLE) {
              float bary0;
              float bary1;
              float bary2;
              barycentrics(intersect_point, g.triangleVertices[0], g.triangleVertices[1], g.triangleVertices[2], bary0, bary1, bary2);
              glm::vec2 uv = bary0 * g.triangleUVs[0] + bary1 * g.triangleUVs[1] + bary2 * g.triangleUVs[2];
              intersections[path_index].uv = uv;
              intersections[path_index].hasUV = 1;
            }
          }
        }
        // naive parse through global geoms
        else {
          for (int i = 0; i < geoms_size; i++)
          {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
              t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
              t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE)
            {
              t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
              t_min = t;
              hit_geom_index = i;
              intersect_point = tmp_intersect;
              normal = tmp_normal;
            }
          }

          if (hit_geom_index == -1)
          {
            intersections[path_index].t = -1.0f;
            // set materialId to INT_MAX so whenn sorting by material is put at the end
            intersections[path_index].materialId = INT_MAX;
          }
          else
          {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;

            // default to no UV coords
            intersections[path_index].uv = glm::vec2(0.0f);
            intersections[path_index].hasUV = 0;

            // If hit a triangle compute UV
            const Geom& g = geoms[hit_geom_index];
            if (g.type == TRIANGLE) {
              float bary0;
              float bary1;
              float bary2;
              barycentrics(intersect_point, g.triangleVertices[0], g.triangleVertices[1], g.triangleVertices[2], bary0, bary1, bary2);
              glm::vec2 uv = bary0 * g.triangleUVs[0] + bary1 * g.triangleUVs[1] + bary2 * g.triangleUVs[2];
              intersections[path_index].uv = uv;
              intersections[path_index].hasUV = 1;
            }

          }
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Basic BSDF shader (rays/pathSegments/intersections NOT contiguous in memory by material type)
__global__ void shadeBSDFMaterial
(
  int iter,
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  Material* materials,
  const GPUTexture* textures)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) // if the intersection exists...
    {
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];

      // sample textures if posible
      if (textures != nullptr && material.baseColorTexture >= 0 && (intersection.hasUV != 0.0f)) {
        glm::vec2 uv = intersection.uv;
        glm::vec3 texLinear = getLinearTextureColor(textures, material.baseColorTexture, uv);
        material.color *= texLinear;
      }
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        // terminate path at light source
        pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        // Generate a new ray to continue the path (update ray's PathSegment in place) and add current ray's color (in scatter Ray)
        if (pathSegments[idx].remainingBounces > 0) {

          if (RUSSIAN_ROULETTE) {
            if (pathSegments[idx].bouncesCompleted > BOUNCES_TO_START_RUSSIAN_ROULETTE) {
              // use path's max color component for how impactful the ray is
              float lum = 0.2126f * pathSegments[idx].color.r + 0.7152f * pathSegments[idx].color.g + 0.0722f * pathSegments[idx].color.b;
              lum = fminf(fmaxf(lum, 0.f), 1.f);
              float q = 1 - lum;
              q = fminf(fmaxf(q, MIN_Q), MAX_Q);

              float r = u01(rng);
              if (r < q) {
                // terminate path
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
              }
              else {
                // amplify color to accomodate for terminated rays
                pathSegments[idx].color *= glm::vec3(1.0f / (1.0f - q));
              }
            }
          }

          glm::vec3 intersectionPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
          pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
          scatterRay(pathSegments[idx], intersectionPoint, intersection.surfaceNormal, material, rng);
          // offset next ray to prevent same intersection again
          pathSegments[idx].ray.origin += 0.001f * pathSegments[idx].ray.direction;
          pathSegments[idx].remainingBounces--;
          pathSegments[idx].bouncesCompleted++;
        }
        else {
          // set ray to black if bottomed out
          pathSegments[idx].color = glm::vec3(0.0f);
        }
      }
      // If there was no intersection, color the ray black and set remaining bounces to 0 to signify it's done.
    }
    else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

// Set up the material keys so we can sort by material id
__global__ void createMaterialKeys(int n, const ShadeableIntersection* shadeableintersections, int* materialKeys) 
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < n)
  {
    ShadeableIntersection intersection = shadeableintersections[index];
    materialKeys[index] = intersection.materialId;
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// used for thrust stream compaction
struct isAlive {
  __host__ __device__
  bool operator()(const PathSegment& path)
  {
    return path.remainingBounces > 0;
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int old_num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_nodes,
            dev_primIndices,
            numNodes,
            numPrimIndices,
            numGeoms
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //    iter,
        //    num_paths,
        //    dev_intersections,
        //    dev_paths,
        //    dev_materials
        //  );

        //Sort by Material
        if (SORT_MATERIALS) {
          // Create array of keys based on material id
          dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
          createMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_materialKeys);
          checkCUDAError("Building material keys failed");

          thrust::device_ptr<int> thrust_mat_keys_begin = thrust::device_pointer_cast(dev_materialKeys);
          thrust::device_ptr<ShadeableIntersection> thrust_intersections_begin = thrust::device_pointer_cast(dev_intersections);
          thrust::device_ptr<PathSegment> thrust_paths_begin = thrust::device_pointer_cast(dev_paths);

          // Zip intersections and paths to sort together
          thrust::zip_iterator<thrust::tuple<thrust::device_ptr<ShadeableIntersection>, thrust::device_ptr<PathSegment>>> zip_begin = 
            thrust::make_zip_iterator(thrust::make_tuple(thrust_intersections_begin, thrust_paths_begin));

          // Sort by material id
          thrust::sort_by_key(thrust::device, thrust_mat_keys_begin, thrust_mat_keys_begin + num_paths, zip_begin);
          checkCUDAError("Sorting by material failed");
        }

        shadeBSDFMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures
        );

        // Stream Compaction
        if (STREAM_COMPACTION) {
          dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, isAlive{});
          num_paths = dev_path_end - dev_paths;
        }

        if (num_paths == 0 || depth >= traceDepth) {
          iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
