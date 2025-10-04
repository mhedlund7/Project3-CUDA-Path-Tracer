#include "gltfLoader.h"
#include "tiny_gltf.h"
#include <iostream>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <utility>


// helper to convert gltf matrix to glm::mat4
glm::mat4 convertGltfMatrixToGlm(const std::vector<double>& gltfMatrix) 
{
    glm::mat4 mat(1.0f);
    if (gltfMatrix.size() == 16) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                mat[i][j] = static_cast<float>(gltfMatrix[i * 4 + j]);
            }
        }
    }
    return mat;
}

// Used to construct world space (by multiplying with parent's world space) at each node
glm::mat4 NodeMatrix(const tinygltf::Node& n) 
{
  // Either use provided matrix or creae from translation, rotation, scale fields
  if (n.matrix.size() == 16) {
      return convertGltfMatrixToGlm(n.matrix);
  }

  glm::mat4 mat(1.0f);
  if (n.translation.size() == 3) {
    mat = glm::translate(mat, glm::vec3(
      static_cast<float>(n.translation[0]),
      static_cast<float>(n.translation[1]),
      static_cast<float>(n.translation[2])));
  }
  if (n.rotation.size() == 4) {
    // reorder rotation quaternion from (x, y, z, w) to (w, x, y, z)
    glm::quat q(
      static_cast<float>(n.rotation[3]),
      static_cast<float>(n.rotation[0]),
      static_cast<float>(n.rotation[1]),
      static_cast<float>(n.rotation[2]));
    mat *= glm::mat4_cast(q);
  }
  if (n.scale.size() == 3) {
    mat = glm::scale(mat, glm::vec3(
      static_cast<float>(n.scale[0]),
      static_cast<float>(n.scale[1]),
      static_cast<float>(n.scale[2])));
  }
  return mat;
}

// Creates a typed pointer into the data in a gltf accessor so we can read it easily in more familiar types
template<typename T>
const T* AccessorPtr(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
  const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
  return reinterpret_cast<const T*>(
    &(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
}


// HELPER CODE
// Used to try and fetch more material info if they exist
// reading a float from an extension object with a defaulVal
static float ReadExtFloat(const tinygltf::Value* extObj, const char* key, float defaultVal) {
  if (!extObj || !extObj->IsObject()) {
    return defaultVal;
  }
  const auto& obj = extObj->Get<tinygltf::Value::Object>();
  auto it = obj.find(key);
  if (it == obj.end()) {
    return defaultVal;
  }
  if (it->second.IsNumber()) {
    return (it->second.Get<double>());
  }
  return defaultVal;
}

// reading a vec3 from an extension object with a defaulVal
static glm::vec3 ReadExtVec3(const tinygltf::Value* extObj, const char* key, glm::vec3 defaultVal) {
  if (!extObj || !extObj->IsObject()) {
    return defaultVal;
  }
  const auto& obj = extObj->Get<tinygltf::Value::Object>();
  auto it = obj.find(key);
  if (it == obj.end()) {
    return defaultVal;
  }
  const tinygltf::Value& v = it->second;
  if (!v.IsArray()) {
    return defaultVal;
  }
  const auto& a = v.Get<tinygltf::Value::Array>();
  if (a.size() < 3) {
    return defaultVal;
  }
  return glm::vec3(
    (a[0].Get<double>()),
    (a[1].Get<double>()),
    (a[2].Get<double>())
  );
}

// reading a vec2 from an extension object with a defaulVal
static glm::vec2 ReadExtVec2(const tinygltf::Value* extObj,
  const char* key,
  glm::vec2 defaultVal) {
  if (!extObj || !extObj->IsObject()) {
    return defaultVal;
  }
  const auto& obj = extObj->Get<tinygltf::Value::Object>();
  auto it = obj.find(key);
  if (it == obj.end()) {
    return defaultVal;
  }
  const tinygltf::Value& v = it->second;
  if (!v.IsArray()) {
    return defaultVal;
  }
  const auto& a = v.Get<tinygltf::Value::Array>();
  if (a.size() < 2) {
    return defaultVal;
  }
  return glm::vec2(
   (a[0].Get<double>()),
    (a[1].Get<double>()));
}

// try and convert from material roughness to an appropraite specular component that looks good
static float RoughnessToPhongExponent(float r) {
  r = std::clamp(r, 0.001f, 0.999f);
  float n = 2.0f / (r * r) - 2.0f;
  return std::max(1.0f, n);
}

// Try and get a pointer to an extension object for gltf if it exists, defaults to nullptr
static const tinygltf::Value* GetExt(const tinygltf::Material& m, const char* extName) {
  auto it = m.extensions.find(extName);
  if (it == m.extensions.end()) {
    return nullptr;
  }
  return &it->second;
}

// gets max component of a vector
static float Max3(glm::vec3 v) { 
  return std::max(v.x, std::max(v.y, v.z)); 
}

// tries to read a vec3 array to out from accessor even when strided
static bool ReadVec3Array(const tinygltf::Model& model,
  const tinygltf::Accessor& acc,
  std::vector<glm::vec3>& out)
{
  if (acc.bufferView < 0 ||
    acc.type != TINYGLTF_TYPE_VEC3 ||
    acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
    return false;

  const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
  if (bv.buffer < 0) {
    return false;
  }
  const tinygltf::Buffer& buf = model.buffers[bv.buffer];

  const size_t stride = (bv.byteStride != 0) ? bv.byteStride : sizeof(float) * 3;
  const size_t offset = bv.byteOffset + acc.byteOffset;

  // bounds check (last element + 3 floats)
  const size_t last = offset + (acc.count ? (acc.count - 1) * stride : 0) + sizeof(float) * 3;
  if (last > buf.data.size()) {
    return false;
  }

  out.resize(acc.count);

  const uint8_t* base = buf.data.data() + offset;

  if (stride == sizeof(float) * 3) {
    // packed :D yippie
    const float* f = reinterpret_cast<const float*>(base);
    for (size_t i = 0; i < acc.count; ++i) {
      out[i] = glm::vec3(f[i * 3 + 0], f[i * 3 + 1], f[i * 3 + 2]);
    }
  }
  else {
    // interleaved
    for (size_t i = 0; i < acc.count; ++i) {
      const float* f = reinterpret_cast<const float*>(base + i * stride);
      out[i] = glm::vec3(f[0], f[1], f[2]);
    }
  }
  return true;
}

// trying to read indices information from the model no matter what size they're stored as
static bool ReadIndicesU32(const tinygltf::Model& model,
  const tinygltf::Accessor& acc,
  std::vector<uint32_t>& out)
{
  if (acc.bufferView < 0) {
    return false;
  }

  const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
  if (bv.buffer < 0) {
    return false;
  }
  const tinygltf::Buffer& buf = model.buffers[bv.buffer];

  const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;

  // get size of index components
  size_t compSize = 0;
  switch (acc.componentType) {
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  compSize = 1; break;
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: compSize = 2; break;
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   compSize = 4; break;
  default: return false;
  }

  const size_t stride = (bv.byteStride != 0) ? bv.byteStride : compSize;

  const size_t last = (acc.count ? (acc.count - 1) * stride : 0) + compSize;
  if (bv.byteOffset + acc.byteOffset + last > buf.data.size()) return false;

  // properly populate indices into out
  out.resize(acc.count);
  for (size_t i = 0; i < acc.count; ++i) {
    const uint8_t* p = base + i * stride;
    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      out[i] = *reinterpret_cast<const uint8_t*>(p);  break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      out[i] = *reinterpret_cast<const uint16_t*>(p); break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      out[i] = *reinterpret_cast<const uint32_t*>(p); break;
    }
  }
  return true;
}

// tries to read a vec2 array from the gltf accessor even if strided, used for stuff like uvs/other vec2s
static bool ReadVec2Array(const tinygltf::Model& model,
  const tinygltf::Accessor& acc,
  std::vector<glm::vec2>& out) {
  if (acc.bufferView < 0 ||
    acc.type != TINYGLTF_TYPE_VEC2 ||
    acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
    return false;

  const auto& bv = model.bufferViews[acc.bufferView];
  const auto& buf = model.buffers[bv.buffer];

  const size_t stride = (bv.byteStride != 0) ? bv.byteStride : sizeof(float) * 2;
  const size_t offset = bv.byteOffset + acc.byteOffset;
  const size_t last = offset + (acc.count ? (acc.count - 1) * stride : 0) + sizeof(float) * 2;
  if (last > buf.data.size()) {
    return false;
  }

  out.resize(acc.count);
  const uint8_t* base = buf.data.data() + offset;
  if (stride == sizeof(float) * 2) {
    const float* f = reinterpret_cast<const float*>(base);
    for (size_t i = 0; i < acc.count; i++) out[i] = glm::vec2(f[i * 2 + 0], f[i * 2 + 1]);
  }
  else {
    for (size_t i = 0; i < acc.count; i++) {
      const float* f = reinterpret_cast<const float*>(base + i * stride);
      out[i] = glm::vec2(f[0], f[1]);
    }
  }
  return true;
}

// tries to convert glTF texture image to RGBA8 soit can be put in as CPUTexture
static int LoadTextureFromGltfTextureIndex(const tinygltf::Model& model,
  int gltfTexIndex,
  // check valid
  std::vector<CPUTexture>& textures) {
  if (gltfTexIndex < 0 || gltfTexIndex >= (int)model.textures.size()) {
    return -1;
  }
  const auto& tex = model.textures[gltfTexIndex];
  int imageIndex = tex.source;
  if (imageIndex < 0 || imageIndex >= (int)model.images.size()) {
    return -1;
  }
  // get image
  const auto& img = model.images[imageIndex];

  CPUTexture out;
  out.width = img.width;
  out.height = img.height;
  // make rgba for proper cpu texture
  const int srcC = std::max(1, img.component);
  out.channels = 4;
  out.rgba.resize(out.width * out.height * 4);

  // populate pixel data
  const uint8_t* src = img.image.data();
  for (int i = 0; i < out.width * out.height; ++i) {
    uint8_t r = 0, g = 0, b = 0, a = 255;
    if (srcC >= 1) r = src[i * srcC + 0];
    if (srcC >= 2) g = src[i * srcC + 1]; else g = r;
    if (srcC >= 3) b = src[i * srcC + 2]; else b = r;
    if (srcC >= 4) a = src[i * srcC + 3];
    out.rgba[i * 4 + 0] = r;
    out.rgba[i * 4 + 1] = g;
    out.rgba[i * 4 + 2] = b;
    out.rgba[i * 4 + 3] = a;
  }

  // actually add texture to textures
  int idx = textures.size();
  textures.push_back(std::move(out));
  return idx;
}

// END HELPER CODE

// Recursively add node's mesh as triangles in the scene
void addNode(int nodeIndex, const glm::mat4& parentWorldMatrix, const tinygltf::Model& model, const int& defaultMaterialIndex, const std::vector<int>& materialMap, std::vector<Geom>& geoms, std::vector<CPUTexture>& textures) {
  // construct curr node's world matrix
  const auto& currNode = model.nodes[nodeIndex];
  glm::mat4 worldMatrix = parentWorldMatrix * NodeMatrix(currNode);

  if (currNode.mesh >= 0) {
    // if current node has a mesh, process it
    const auto& mesh = model.meshes[currNode.mesh];
    // process each primative
    for (const auto& primative : mesh.primitives) {

      // if primative ever fails, continue to next primitive
      // only process triangles
      if (primative.mode != TINYGLTF_MODE_TRIANGLES) {
        continue;
      }
      // skip if no indices
      if (primative.indices < 0) {
        continue;
      }

      auto posIterator = primative.attributes.find("POSITION");
      if (posIterator == primative.attributes.end()) {
        // primative doesn't have position so skip
        continue;
      }

      // try to get positions
      const tinygltf::Accessor& posAcc = model.accessors[posIterator->second];
      std::vector<glm::vec3> positions;
      if (!ReadVec3Array(model, posAcc, positions)) continue;

      // try to get uvs if it has them
      std::vector<glm::vec2> uvs0;
      auto uvIt = primative.attributes.find("TEXCOORD_0");
      bool hasUV = false;
      if (uvIt != primative.attributes.end()) {
        const tinygltf::Accessor& uvAcc = model.accessors[uvIt->second];
        hasUV = ReadVec2Array(model, uvAcc, uvs0);
      }

      // try to get indices
      const tinygltf::Accessor& idxAcc = model.accessors[primative.indices];
      std::vector<uint32_t> indices;
      if (!ReadIndicesU32(model, idxAcc, indices)) continue;
      if (indices.size() < 3) continue;


      // get the material of the primative
      int materialIndex = defaultMaterialIndex;
      if (primative.material >= 0 && primative.material < materialMap.size() && materialMap[primative.material] >= 0) {
        materialIndex = materialMap[primative.material];
      }

      // Create a triangle for each set of 3 indices
      for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i + 0];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];
        Geom triangle = {};
        triangle.type = GeomType::TRIANGLE;
        triangle.materialid = materialIndex;

        // identities cause triangle is stored in world space
        triangle.translation = glm::vec3(0);
        triangle.rotation = glm::vec3(0);
        triangle.scale = glm::vec3(1);
        triangle.transform = glm::mat4(1.0f);
        triangle.inverseTransform = glm::mat4(1.0f);
        triangle.invTranspose = glm::mat4(1.0f);
        
        glm::mat3 M3(worldMatrix);
        bool flipsHandedness = glm::determinant(M3) < 0.0f;

        if (flipsHandedness) {
          int tmp = i1;
          i1 = i2;
          i2 = tmp;
        }

        // store vertices in world space
        glm::vec3 p0 = glm::vec3(worldMatrix * glm::vec4(positions[i0], 1.0f));
        glm::vec3 p1 = glm::vec3(worldMatrix * glm::vec4(positions[i1], 1.0f));
        glm::vec3 p2 = glm::vec3(worldMatrix * glm::vec4(positions[i2], 1.0f));
        triangle.triangleVertices[0] = p0;
        triangle.triangleVertices[1] = p1;
        triangle.triangleVertices[2] = p2;

        // UVS
        if (hasUV) {
          triangle.triangleUVs[0] = uvs0[i0];
          triangle.triangleUVs[1] = uvs0[i1];
          triangle.triangleUVs[2] = uvs0[i2];
        }
        else {
          triangle.triangleUVs[0] = triangle.triangleUVs[1] = triangle.triangleUVs[2] = glm::vec2(0.0f);
        }

        // add triangle to geometry
        geoms.push_back(triangle);
      }
    }
  }
  // recurse to children
  for (const auto& childIndex : currNode.children) {
    addNode(childIndex, worldMatrix, model, defaultMaterialIndex, materialMap, geoms, textures);
  }
}

bool loadGltfAsTriangles(
    const std::string& gltfFile,
    std::vector<Geom>& geoms,
    std::vector<Material>& materials,
    std::vector<CPUTexture>& textures)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfFile.c_str());
    if (!warn.empty()) {
        std::cout << "Warn: " << warn << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to parse glTF: " << gltfFile << ". " << err << std::endl;
        return false;
    }

    // Create materials from gltf materials

    // Maps from gltf material index to our material index
    std::vector<int> materialMap = std::vector<int>(model.materials.size(), -1);

    for (size_t i = 0; i < model.materials.size(); i++) {
      // gltf material
      const auto& gltfMat = model.materials[i];
      // our material
      Material material = {};
      material.baseColorTexture = -1;

      // try and get as many material components as possible to use, otherwise default to just not using them
      // Base Color
      const auto& baseColor = gltfMat.pbrMetallicRoughness.baseColorFactor;
      material.color = glm::vec3(baseColor[0], baseColor[1], baseColor[2]);

      // metals and roughness
      float metallic = gltfMat.pbrMetallicRoughness.metallicFactor;
      float roughness = gltfMat.pbrMetallicRoughness.roughnessFactor;
      roughness = std::clamp(roughness, 0.0f, 1.0f);
      metallic = std::clamp(metallic, 0.0f, 1.0f);

      // specular
      const auto extSpecular = GetExt(gltfMat, "KHR_materials_specular");
      float specularFact = ReadExtFloat(extSpecular, "specularFactor", 1.0f);
      glm::vec3 specularColFact = ReadExtVec3(extSpecular, "specularColorFactor", glm::vec3(1.0f));

      // our material specular exponent
      material.specular.exponent = RoughnessToPhongExponent(roughness);

      // specularColor
      glm::vec3 channelWeights(0.04f);
      // specular color based on metalicness
      glm::vec3 specColor = glm::mix(channelWeights, material.color, metallic);
      specColor *= specularFact;
      specColor *= specularColFact;
      material.specular.color = specColor;

      // IOR
      const auto extIor = GetExt(gltfMat, "KHR_materials_ior");
      float ior = ReadExtFloat(extIor, "ior", 1.5f);
      material.indexOfRefraction = ior;

      // Transmission
      const auto extTrans = GetExt(gltfMat, "KHR_materials_transmission");
      float transmissionFactor = ReadExtFloat(extTrans, "transmissionFactor", 0.0f);
      transmissionFactor = std::clamp(transmissionFactor, 0.0f, 1.0f);

      // Emissive
      const auto& emissive = gltfMat.emissiveFactor;
      // average the emmisive values
      material.emittance = (emissive[0] + emissive[1] + emissive[2]) / 3.0f;
      
      material.hasReflective = (Max3(material.specular.color) > 0.05f || metallic > 0.2f) ? 1.0f : 0.0f;
      material.hasRefractive = (transmissionFactor > 0.05f) ? 1.0f : 0.0f;

      // lobe probabilities based on how metallic/transmissive/specular - try to make look good
      float kd = (1.0f - metallic) * (1.0f - transmissionFactor);
      float ks = std::clamp(Max3(material.specular.color), 0.0f, 1.0f);
      float kt = transmissionFactor;


      // texture stuff
      // bind texture lambda
      auto bindTex = [&](const tinygltf::TextureInfo& ti, int& outIndex) {
        outIndex = LoadTextureFromGltfTextureIndex(model, ti.index, textures);
       };

      // Base Color from texture
      if (gltfMat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
        bindTex(gltfMat.pbrMetallicRoughness.baseColorTexture,
          material.baseColorTexture);
      }

      // map the material indices
      materialMap[i] = materials.size();
      materials.push_back(material);
    }

    // Make a default material in case a mesh has no material
    const int defaultMaterialIndex = materials.size();
    Material defaultMaterial = {};
    defaultMaterial.color = glm::vec3(0.5f, 0.5f, 0.5f);
    defaultMaterial.emittance = 0.0f;
    materials.push_back(defaultMaterial);

    // Create geoms (triangles) from gltf meshes
    const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;

    if (sceneIndex < 0 || sceneIndex >= model.scenes.size()) {
      std::cerr << "Invalid scene index in glTF: " << gltfFile << std::endl;
      return false;
    }

    const auto& scene = model.scenes[sceneIndex];
    for (const auto& nodeIndex : scene.nodes) {
      addNode(nodeIndex, glm::mat4(1.0f), model, defaultMaterialIndex, materialMap, geoms, textures);
    }

    return true;
}


