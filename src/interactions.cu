#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

// perfect specular reflect helper
__host__ __device__ glm::vec3 reflectVec(glm::vec3 i, glm::vec3 normal) {
  return i - 2 * glm::dot(i, normal) * normal;
}

// Schlick approximation
__host__ __device__ float schlickFresnel(float cosTheta, float etaIn, float etaOut) {
  float r0 = powf(((etaIn - etaOut) / (etaIn + etaOut)), 2.0f);
  return r0 + (1 - r0) * powf(1.0f - cosTheta, 5.0f);
}

// Imperfect specular ray
__host__ __device__ glm::vec3 randomSpecularRay(const glm::vec3& ray, float exponent, thrust::default_random_engine& rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);
  float r1 = u01(rng);
  float r2 = u01(rng);
  float phi = 2 * PI * r1;
  float theta = acosf(powf(r2, 1.0 / (exponent + 1.0f)));

  glm::vec3 directionNotRay;
  if (abs(ray.x) < SQRT_OF_ONE_THIRD)
  {
    directionNotRay = glm::vec3(1, 0, 0);
  }
  else if (abs(ray.y) < SQRT_OF_ONE_THIRD)
  {
    directionNotRay = glm::vec3(0, 1, 0);
  }
  else
  {
    directionNotRay = glm::vec3(0, 0, 1);
  }

  // Use not-ray direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
    glm::normalize(glm::cross(ray, directionNotRay));
  glm::vec3 perpendicularDirection2 =
    glm::normalize(glm::cross(ray, perpendicularDirection1));

  return glm::normalize(perpendicularDirection1 * (cosf(phi) * sinf(theta)) + perpendicularDirection2 * (sinf(phi) * sinf(theta)) + glm::normalize(ray) * cosf(theta));
}


__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
  glm::vec3 normal,
  thrust::default_random_engine& rng)
{
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng)); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD)
  {
    directionNotNormal = glm::vec3(1, 0, 0);
  }
  else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
  {
    directionNotNormal = glm::vec3(0, 1, 0);
  }
  else
  {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
    glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
    glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal
    + cos(around) * over * perpendicularDirection1
    + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
  PathSegment& pathSegment,
  glm::vec3 intersect,
  glm::vec3 normal,
  const Material& m,
  thrust::default_random_engine& rng)
{
  // TODO: implement this.
  // A basic implementation of pure-diffuse shading will just call the
  // calculateRandomDirectionInHemisphere defined above.

  thrust::uniform_real_distribution<float> u01(0, 1);
  glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);

  // makesure have normal pointing against incoming ray
  bool entering = glm::dot(normal, wo) > 0.0f;
  glm::vec3 cNormal = entering ? normal : -normal;

  // get proper effect probabilities
  float sum = m.diffuseP + m.specularP;
  float pd = 1.0f;
  float ps = 0.0f;
  if (sum > 0.0f) {
    pd = m.diffuseP / sum;
    ps = m.specularP / sum;
  }
  else {
    if (m.hasRefractive || m.hasReflective) {
      pd = 0.f; ps = 1.f;
    }
    else {
      pd = 1.f; ps = 0.f;
    }
  }
  if (!m.hasReflective && !m.hasRefractive) {
    ps = 0;
    pd = 1;
  }

  float r = u01(rng);

  // get probability of the event that is actually happening
  float eP = (r < pd) ? pd : ps;
  // don't spike
  eP = fmaxf(1e-5, eP);

  if (r < pd) {
    // Diffuse
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(cNormal, rng);
    pathSegment.color *= m.color;
    // 1 / eP comp for dif probs
    pathSegment.color *= (1.f / eP);
  }
  else if (!m.hasRefractive) {
    // Only Reflect
    glm::vec3 reflect = reflectVec(-wo, cNormal);

    // imperfectSpecular (if exponent not set just take perfect reflection)
    if (m.specular.exponent >= 1e-5) {
      glm::vec3 wi = randomSpecularRay(reflect, m.specular.exponent, rng);
      // check to make sure getting correct sides
      if (glm::dot(wi, cNormal) < 0.0f) {
        wi = reflectVec(-wi, cNormal);
      }
      pathSegment.ray.direction = glm::normalize(wi);
    }
    else {
      pathSegment.ray.direction = glm::normalize(reflect);
    }

    pathSegment.color *= m.specular.color;
    
    // 1 / eP comp
    pathSegment.color *= (1.f / (eP));
  }
  else {
    // Reflect and Refract
    // get iors
    float etaIn = entering ? 1.0f : m.indexOfRefraction;
    float etaOut = entering ? m.indexOfRefraction : 1.0f;

    // get cosTheta value
    float cosTheta = fminf(1.0f, glm::dot(wo, cNormal));

    float f = schlickFresnel(cosTheta, etaIn, etaOut);
    f = fminf(fmaxf(f, 1e-6f), 1.f - 1e-6f);

    // reflect or refract based on fresnel
    float r2 = u01(rng);
    if (r2 < f) {
      // reflect
      glm::vec3 reflect = reflectVec(-wo, cNormal);

      // imperfectSpecular (if exponent not set just take perfect reflection)
      if (m.specular.exponent >= 1e-5) {
        glm::vec3 wi = randomSpecularRay(reflect, m.specular.exponent, rng);
        // check to make sure getting correct sides
        if (glm::dot(wi, cNormal) < 0.0f) {
          wi = reflectVec(-wi, cNormal);
        }
        pathSegment.ray.direction = glm::normalize(wi);
      }
      else {
        pathSegment.ray.direction = glm::normalize(reflect);
      }
      //pathSegment.color *= m.specular.color;
      pathSegment.color *= glm::vec3(0.1, 0.0, 0.0);

      pathSegment.color *= (1.f / (eP * f));
    }
    else {
      // refract
      glm::vec3 wi = glm::refract(-wo, cNormal, etaIn / etaOut);

      // total internal reflection check -- maybe scuffed
      if (powf(glm::length(wi), 2.0f) < 1e-15f) {
        wi = reflectVec(-wo, cNormal);
        // compensate for prob
        pathSegment.color *= (1.f / (eP));
      }
      else {
        pathSegment.color *= (1.f / (eP * (1 - f)));
      }

      pathSegment.ray.direction = glm::normalize(wi);
    }
  }

}
