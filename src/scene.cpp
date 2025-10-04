#include "scene.h"

#include "utilities.h"
#include "gltfLoader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "octree.h"

using namespace std;
using json = nlohmann::json;

string sceneDir = "../scenes/";
vector<string> scenesToLoad = {"cornellRoom.json", "duck/duck.gltf"};

Scene::Scene(string filename)
{

  for (string scene : scenesToLoad) {
    filename = sceneDir + scene;
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
      loadFromJSON(filename);
    }
    else if (ext == ".gltf") {
      loadFromGLTF(filename);
    }
    else
    {
      cout << "Couldn't read from " << filename << endl;
      exit(-1);
    }
  }
  octree.buildOctree(geoms, 8, 10);

}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = true;
            newMaterial.hasRefractive = true;
            newMaterial.diffuseP = 0.0f;
            newMaterial.specularP = 1.0f;
            newMaterial.indexOfRefraction = 1.5;
            newMaterial.specular.exponent = 0;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


void Scene::loadFromGLTF(const std::string& gltfName)
{
  // Set up default camera and state
  Camera& camera = state.camera;
  RenderState& state = this->state;
  camera.resolution.x = 800;
  camera.resolution.y = 800;
  float fovy = 45.0;
  state.iterations = 5000;
  state.traceDepth = 8;
  state.imageName = "gltfImg";
  camera.position = glm::vec3(0.0, 1.0, 3.0);
  camera.lookAt = glm::vec3(0.0, 1.0, 0.0);
  camera.up = glm::vec3(.0, 1.0, 0.0);

  //calculate fov based on resolution
  float yscaled = tan(fovy * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, fovy);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
    2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // actually load the gltf file
  if (!loadGltfAsTriangles(gltfName, geoms, materials, textures)) {
    std::cerr << "load gltf as traingles failes" << std::endl;
    exit(-1);
  }

  // add a light if none exists
  bool hasLight = false;
  for (const auto& mat : materials) {
    if (mat.emittance > 0.0f) {
      hasLight = true;
      break;
    }
  }

  if (!hasLight) {
    Material lightMat = {};
    lightMat.color = glm::vec3(1.0f, 1.0f, 1.0f);
    lightMat.emittance = 5.0f;
    materials.push_back(lightMat);
    // add a big sphere light
    Geom lightGeom = {};
    lightGeom.type = SPHERE;
    lightGeom.materialid = materials.size() - 1;
    lightGeom.translation = glm::vec3(0.0f, 5.0f, 0.0f);
    lightGeom.scale = glm::vec3(3.0f, 3.0f, 3.0f);
    lightGeom.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    lightGeom.transform = utilityCore::buildTransformationMatrix(
      lightGeom.translation, lightGeom.rotation, lightGeom.scale);
    lightGeom.inverseTransform = glm::inverse(lightGeom.transform);
    lightGeom.invTranspose = glm::inverseTranspose(lightGeom.transform);
    geoms.push_back(lightGeom);
  }

  // print geometry and material counts
  std::cout << "Loaded glTF scene with " << geoms.size() << " geometries and "
    << materials.size() << " materials." << std::endl;


  //set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
