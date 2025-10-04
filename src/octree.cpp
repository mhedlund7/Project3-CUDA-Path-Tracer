#include "octree.h"
#include <iostream>
#include "intersections.h"

// Create bounding box for different geometries functions
// 

// Helpers
// Apply Matrix to point or vector
__host__ __device__ inline glm::vec3 transformPoint(const glm::mat4& M, const glm::vec3& p) {
  glm::vec4 r = M * glm::vec4(p, 1.0f);
  return glm::vec3(r);
}
__host__ __device__ inline glm::vec3 transformVector(const glm::mat4& M, const glm::vec3& v) {
  glm::vec4 r = M * glm::vec4(v, 0.0f);
  return glm::vec3(r);
}

BBox Octree::triangleBBox(const Geom& g) {
  BBox b = makeEmptyBBox();
  // transform to world space for detection
  expandBBox(b, transformPoint(g.transform, g.triangleVertices[0]));
  expandBBox(b, transformPoint(g.transform, g.triangleVertices[1]));
  expandBBox(b, transformPoint(g.transform, g.triangleVertices[2]));
  // pad for edge cases
  b.min -= glm::vec3(1e-4f);
  b.max += glm::vec3(1e-4f);
  return b;
}

BBox Octree::cubeBBox(const Geom& g) {
  // base cube
  static const glm::vec3 corners[8] = {
    {-0.5f,-0.5f,-0.5f}, {+0.5f,-0.5f,-0.5f},
    {-0.5f,+0.5f,-0.5f}, {+0.5f,+0.5f,-0.5f},
    {-0.5f,-0.5f,+0.5f}, {+0.5f,-0.5f,+0.5f},
    {-0.5f,+0.5f,+0.5f}, {+0.5f,+0.5f,+0.5f},
  };
  BBox b = makeEmptyBBox();
  // expand box to cover all points
  for (int i = 0; i < 8; ++i) {
    expandBBox(b, transformPoint(g.transform, corners[i]));
  }
  // pad
  b.min -= glm::vec3(1e-4f);
  b.max += glm::vec3(1e-4f);
  return b;
}

BBox Octree::sphereBBox(const Geom& g) {

  //base sphere
  const float baseRadius = 0.5f;

  glm::vec3 centerW = transformPoint(g.transform, glm::vec3(0));
  glm::vec3 x = transformVector(g.transform, glm::vec3(1, 0, 0));
  glm::vec3 y = transformVector(g.transform, glm::vec3(0, 1, 0));
  glm::vec3 z = transformVector(g.transform, glm::vec3(0, 0, 1));
  float s = baseRadius * std::max(std::max(glm::length(x), glm::length(y)), glm::length(z));

  // encapsulate sphere around larges direction
  BBox b;
  b.min = centerW - glm::vec3(s);
  b.max = centerW + glm::vec3(s);
  return b;
}

// make bounding box around any supported geom
BBox Octree::geomBBox(const Geom& geom) {
  switch (geom.type) {
    case SPHERE:
      return sphereBBox(geom);
    case CUBE:
      return cubeBBox(geom);
    case TRIANGLE:
      return triangleBBox(geom);
    default:
      return makeEmptyBBox();
  }
}

// Return bounding box for child based on child octant
BBox Octree::childBBox(const BBox& parentBox, int octant) {
  glm::vec3 center = getBBoxCenter(parentBox);
  BBox childBox;
  // min corner

  childBox.min.x = (octant & 1) ? center.x : parentBox.min.x;
  childBox.min.y = (octant & 2) ? center.y : parentBox.min.y;
  childBox.min.z = (octant & 4) ? center.z : parentBox.min.z;

  // max corner
  childBox.max.x = (octant & 1) ? parentBox.max.x : center.x;
  childBox.max.y = (octant & 2) ? parentBox.max.y : center.y;
  childBox.max.z = (octant & 4) ? parentBox.max.z : center.z;
  return childBox;
}

// Build the octree from the provided geoms and configuration parameters
void Octree::buildOctree(const std::vector<Geom>& geoms, int maxDepth, int maxLeafPrims) {
  this->maxDepth = maxDepth;
  this->maxLeafPrims = maxLeafPrims;

  // make array of all primative indices
  std::vector<int> allPrimIndices(geoms.size());
  for (int i = 0; i < geoms.size(); i++) {
    allPrimIndices[i] = i;
  }

  // get root bounding box for the entire scene (around all geoms)
  BBox sceneBox = makeEmptyBBox();
  for (const auto& geom : geoms) {
    BBox geomBox = geomBBox(geom);
    expandBBox(sceneBox, geomBox);
  }

  // recursively build octree
  int rootIndex = 0;
  // clear current settings
  octreeNodes.clear();
  octreePrimIndices.clear();
  rootIndex = recursiveBuild(geoms, allPrimIndices, sceneBox, 0);
  std::cout << "Built octree with " << octreeNodes.size() << " nodes and " << octreePrimIndices.size() << " primatives" << std::endl;
}

// The recursive call to actually build the octree
int Octree::recursiveBuild(const std::vector<Geom>& geoms, const std::vector<int>& primIndices, const BBox& currBox, int depth) {
  // tell parent what index node this child is (returned at end)
  int nodeIndex = this->octreeNodes.size();

  // create new node
  this->octreeNodes.push_back(OctreeNode());
  // access later with this->octreeNodes[nodeIndex] to adjust based on children
  OctreeNode& newNode = this->octreeNodes.back();
  // set bbox
  newNode.bbox.min = currBox.min;
  newNode.bbox.max = currBox.max;
  // initialize children to -1 (don't exist)
  for (int i = 0; i < 8; i++) {
    newNode.children[i] = -1;
  }
  newNode.firstPrim = -1;
  newNode.primCount = 0;

  // leaf node check
  if (depth >= maxDepth || primIndices.size() <= maxLeafPrims) {
    newNode.firstPrim = this->octreePrimIndices.size();
    newNode.primCount = primIndices.size();
    // if leaf add primatives to the octrees primative array which we point to
    for (int primIndex : primIndices) {
      this->octreePrimIndices.push_back(primIndex);
    }
    return nodeIndex;
  }

  // if not a leaf, split into children
  std::vector<int> childPrimIndices[8];
  // create children bounding boxes
  BBox childBBoxes[8];
  for (int octant = 0; octant < 8; octant++) {
    childBBoxes[octant] = childBBox(currBox, octant);
  }
  // see which octants each primative overlaps and add to that child's list
  for (int primIndex : primIndices) {
    // get primative bounding box
    BBox geomBox = geomBBox(geoms[primIndex]);

    for (int octant = 0; octant < 8; octant++) {
      // check if geom box and octant box overlap
      BBox octantBox = childBBoxes[octant];
      bool overlap = (geomBox.min.x <= octantBox.max.x && geomBox.max.x >= octantBox.min.x) &&
        (geomBox.min.y <= octantBox.max.y && geomBox.max.y >= octantBox.min.y) &&
        (geomBox.min.z <= octantBox.max.z && geomBox.max.z >= octantBox.min.z);

      // if they do overlap, add to that child's list
      if (overlap) {
        childPrimIndices[octant].push_back(primIndex);
      }
    }
  }

  // Check that primative distribution is good enough to subdivide in c ase of weird edge stuff
  int totalPrimativesAssigned = 0;
  int largestChildCount = 0;
  for (int octant = 0; octant < 8; octant++) {
    int count = childPrimIndices[octant].size();
    totalPrimativesAssigned += count;
    if (count > largestChildCount) {
      largestChildCount = count;
    }
  }

  bool notSplit = (largestChildCount >= primIndices.size());
  bool tooDupe = (totalPrimativesAssigned > 3 * primIndices.size());

  if (notSplit || tooDupe) {
    // make leaf instead
    newNode.firstPrim = this->octreePrimIndices.size();
    newNode.primCount = primIndices.size();
    // add primatives to primative array
    for (int primIndex : primIndices) {
      this->octreePrimIndices.push_back(primIndex);
    }
    return nodeIndex;
  }

  // Create a child node for each non empty primative list
  for (int octant = 0; octant < 8; ++octant) {
    if (!childPrimIndices[octant].empty()) {
      octreeNodes[nodeIndex].children[octant] = recursiveBuild(geoms, childPrimIndices[octant], childBBoxes[octant], depth + 1);
    }
  }
  return nodeIndex;
}

// GPU functions
void Octree::sendToDevice() {
  // copy nodes
  cudaMalloc(&dev_nodes, sizeof(OctreeNode) * octreeNodes.size());
  cudaMemcpy(dev_nodes, octreeNodes.data(), sizeof(OctreeNode) * octreeNodes.size(), cudaMemcpyHostToDevice);

  // copy primative indices
  cudaMalloc(&dev_primIndices, sizeof(int) * octreePrimIndices.size());
  cudaMemcpy(dev_primIndices, octreePrimIndices.data(), sizeof(int) * octreePrimIndices.size(), cudaMemcpyHostToDevice);
}

// free device memory
void Octree::freeDevice() {
  if (dev_nodes) {
    cudaFree(dev_nodes);
    dev_nodes = nullptr;
  }
  if (dev_primIndices) {
    cudaFree(dev_primIndices);
    dev_primIndices = nullptr;
  }
}
