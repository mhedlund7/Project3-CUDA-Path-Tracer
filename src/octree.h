#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include "intersections.h"

// Bounding Box
struct BBox {
  glm::vec3 min;
  glm::vec3 max;
};

// Make an empty bounding box
BBox inline makeEmptyBBox() {
  BBox box;
  box.min = glm::vec3(FLT_MAX);
  box.max = glm::vec3(-FLT_MAX);
  return box;
}

// Expand bounding box to include another point
void inline expandBBox(BBox &box, const glm::vec3 &point) {
  box.min = glm::min(box.min, point);
  box.max = glm::max(box.max, point);
}

// Expand bounding box to include another bounding box
void inline expandBBox(BBox &box, const BBox &other) {
  box.min = glm::min(box.min, other.min);
  box.max = glm::max(box.max, other.max);
}

// Get center of bounding box
glm::vec3 inline getBBoxCenter(const BBox &box) {
  return (box.min + box.max) * 0.5f;
}

// Octree Node. Stores child nodes as indices into a node array, and primatives (if it's a leaf) as indices into a primative array.
// These arrays are in Octree class and are sent to GPU
struct OctreeNode {
  BBox bbox;
  int children[8]; // index of each child nod, -1 if child doesn't exist

  // both for leaves only
  int firstPrim; // index of this nodes first primative in the octree's primative array
  int primCount; // number of primatives in this node

  // return if node is a leaf
  __host__ __device__ bool isLeaf() const { return primCount > 0; }
};

class Octree 
{
public:
  // build octree from geoms and configuration settings
  void buildOctree(const std::vector<Geom>& geoms, int maxDepth, int maxLeafPrims);

  // GPU functions
  void sendToDevice();
  void freeDevice();

  OctreeNode* dev_nodes = nullptr;
  int* dev_primIndices = nullptr;

  std::vector<OctreeNode> octreeNodes;
  std::vector<int> octreePrimIndices;

private:
  // configuration
  int maxDepth = 12;
  int maxLeafPrims = 16;

  // helper functions
  // create bounding box around an arbitrary geom
  static BBox geomBBox(const Geom& geom);
  // create bounding box around respective geom
  static BBox triangleBBox(const Geom& geom);
  static BBox sphereBBox(const Geom& geom);
  static BBox cubeBBox(const Geom& geom);

  int recursiveBuild(const std::vector<Geom>& geoms, const std::vector<int>& primIndices, const BBox& currBox, int depth);

  // get bounding box of child octant given parent bounding box and octant index (0-7)
  static BBox childBBox(const BBox& parentBox, int octant);
};


// GPU traversal and detection functions


__host__ __device__ inline bool intersectRayBBox(const Ray& r, const BBox& box, float& tmin, float& tmax) {
  // detect if a ray intersects the bounding box using slabs
  float tx1 = (box.min.x - r.origin.x) / r.direction.x;
  float tx2 = (box.max.x - r.origin.x) / r.direction.x;
  tmin = fminf(tx1, tx2);
  tmax = fmaxf(tx1, tx2);

  float ty1 = (box.min.y - r.origin.y) / r.direction.y;
  float ty2 = (box.max.y - r.origin.y) / r.direction.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (box.min.z - r.origin.z) / r.direction.z;
  float tz2 = (box.max.z - r.origin.z) / r.direction.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));
  return tmax >= tmin && tmax >= 0.0f;
}

// Item for the array based stack we use to iterate across octree
__host__ __device__ struct StackItem {
  int nodeIndex;
  float tmin;
  float tmax;
};

// Detect which geom a ray collides with in the octree
__host__ __device__ inline void octreeCollisionDetection(
  const Ray& r,
  const OctreeNode* nodes, int numNodes,
  const int* primIndices, int numPrimIndices,
  const Geom* geoms, int numGeoms,
  int& hitGeomIndex, float& t, glm::vec3& intersect_point,
  glm::vec3& normal)
{
  // make stack for itterative traversal
  hitGeomIndex = -1;
  t = FLT_MAX;
  // use array as a stack
  StackItem stack[64]; // fixed size stack. Can increase if octree depth is large
  int stackSize = 0;

  float tmin;
  float tmax;
  // start from root of octree
  if (numNodes <= 0) {
    // no octree
    return; 
  }
  if (!intersectRayBBox(r, nodes[0].bbox, tmin, tmax)) {
    // ray doesn't hit root bounding box
    return; 
  }
  stack[stackSize] = { 0, tmin, tmax }; // push root node
  stackSize++;

  // itteratively traverse the tree (dfs)
  while (stackSize > 0) {
    // get currNode
    stackSize--;
    const StackItem currItem = stack[stackSize];
    const OctreeNode& currNode = nodes[currItem.nodeIndex];

    if (currNode.isLeaf()) {
      // test intersections with each of the leaf's primatives
      for (int i = 0; i < currNode.primCount; i++) {
        int primIndex = primIndices[currNode.firstPrim + i];
        // test intersection
        Geom geom = geoms[primIndex];
        float candidateT = FLT_MAX;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // check for collision based on prim type
        if (geom.type == CUBE)
        {
          candidateT = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
          candidateT = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == TRIANGLE)
        {
          candidateT = triangleIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (candidateT > 0.0f && t > candidateT)
        {
          t = candidateT;
          hitGeomIndex = primIndex;
          intersect_point = tmp_intersect;
          normal = tmp_normal;
        }
      }
    }

    // push children onto stack
    for (int octant = 0; octant < 8; octant++) {
      int childIndex = currNode.children[octant];
      // check if child exists
      if (childIndex != -1) {
        float ctmin;
        float ctmax;
        // if it intersects the childs bounding box, add to stack
        // also check that the entire bounding box isn't after a already detected collision (ctmin < t)
        if (intersectRayBBox(r, nodes[childIndex].bbox, ctmin, ctmax) && ctmin < t) {
          stack[stackSize] = { childIndex, ctmin, ctmax };
          stackSize++;
        }
      }
    }

  }
}