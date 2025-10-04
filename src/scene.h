#pragma once

#include "sceneStructs.h"
#include <vector>
#include "octree.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);

    void loadFromGLTF(const std::string& gltfName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<CPUTexture> textures;
    RenderState state;
    Octree octree;
};
