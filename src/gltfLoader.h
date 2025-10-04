#pragma once

#include <string>
#include <vector>
#include "sceneStructs.h"

bool loadGltfAsTriangles(
  const std::string& gltfFile,
  std::vector<Geom>& geoms,
  std::vector<Material>& materials, std::vector<CPUTexture>& textures);