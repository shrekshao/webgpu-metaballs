// Copyright 2020 Brandon Jones
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import { wgsl } from '../wgsl-debug-helper.js';
import { ProjectionUniforms, ViewUniforms, ModelUniforms, LightUniforms, MaterialUniforms, ColorConversions, ATTRIB_MAP } from '../shaders/common.js';
import { ClusterLightsStructs, TileFunctions } from '../shaders/clustered-compute.js';

function PBR_VARYINGS(defines) { return wgsl`
  [[location(0)]] worldPos : vec3<f32>;
  [[location(1)]] view : vec3<f32>; // Vector from vertex to camera.
  [[location(2)]] texCoord : vec2<f32>;
  [[location(3)]] color : vec4<f32>;

#if ${defines.USE_NORMAL_MAP}
  [[location(4)]] tbn : mat3x3<f32>;
#else
  [[location(4)]] normal : vec3<f32>;
#endif
`;
}

export function PBRVertexSource(defines) { return wgsl`
  ${ProjectionUniforms}
  ${ViewUniforms}
  ${ModelUniforms}

  struct VertexInputs {
    [[location(${ATTRIB_MAP.POSITION})]] position : vec3<f32>;
    [[location(${ATTRIB_MAP.NORMAL})]] normal : vec3<f32>;
    [[location(${ATTRIB_MAP.TEXCOORD_0})]] texCoord : vec2<f32>;
#if ${defines.USE_NORMAL_MAP}
    [[location(${ATTRIB_MAP.TANGENT})]] tangent : vec4<f32>;
#endif
#if ${defines.USE_VERTEX_COLOR}
    [[location(${ATTRIB_MAP.COLOR_0})]] color : vec4<f32>;
#endif
  };

  struct VertexOutput {
    ${PBR_VARYINGS(defines, 'out')}
    [[builtin(position)]] position : vec4<f32>;
  };

  [[stage(vertex)]]
  fn main(input : VertexInputs) -> VertexOutput {
    var output : VertexOutput;
    let n : vec3<f32> = normalize((model.matrix * vec4<f32>(input.normal, 0.0)).xyz);

#if ${defines.USE_NORMAL_MAP}
    let t : vec3<f32> = normalize((model.matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
    let b : vec3<f32> = cross(n, t) * input.tangent.w;
    output.tbn = mat3x3<f32>(t, b, n);
#else
    output.normal = n;
#endif

#if ${defines.USE_VERTEX_COLOR}
    output.color = input.color;
#else
    output.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
#endif

    output.texCoord = input.texCoord;
    let modelPos : vec4<f32> = model.matrix * vec4<f32>(input.position, 1.0);
    output.worldPos = modelPos.xyz;
    output.view = view.position - modelPos.xyz;
    output.position = projection.matrix * view.matrix * modelPos;
    return output;
  }`;
}

function PBRSurfaceInfo(defines) { return wgsl`
  struct FragmentInput {
    [[builtin(frag_coord)]] fragCoord : vec4<f32>;
    ${PBR_VARYINGS(defines)}
  };

  struct SurfaceInfo {
    baseColor : vec4<f32>;
    albedo : vec3<f32>;
    metallic : f32;
    roughness : f32;
    normal : vec3<f32>;
    f0 : vec3<f32>;
    ao : f32;
    emissive : vec3<f32>;
    v : vec3<f32>;
  };

  fn GetSurfaceInfo(input : FragmentInput) -> SurfaceInfo {
    var surface : SurfaceInfo;
    surface.v = normalize(input.view);

    surface.baseColor = material.baseColorFactor * input.color;
#if ${defines.USE_BASE_COLOR_MAP}
    let baseColorMap : vec4<f32> = textureSample(baseColorTexture, defaultSampler, input.texCoord);
    if (baseColorMap.a < 0.05) {
      discard;
    }
    surface.baseColor = surface.baseColor * baseColorMap;
#endif

    surface.albedo = surface.baseColor.rgb;

    surface.metallic = material.metallicRoughnessFactor.x;
    surface.roughness = material.metallicRoughnessFactor.y;

#if ${defines.USE_METAL_ROUGH_MAP}
    let metallicRoughness : vec4<f32> = textureSample(metallicRoughnessTexture, defaultSampler, input.texCoord);
    surface.metallic = surface.metallic * metallicRoughness.b;
    surface.roughness = surface.roughness * metallicRoughness.g;
#endif

#if ${defines.USE_NORMAL_MAP}
    let N : vec3<f32> = textureSample(normalTexture, defaultSampler, input.texCoord).rgb;
    surface.normal = normalize(input.tbn * (2.0 * N - vec3<f32>(1.0, 1.0, 1.0)));
#else
    surface.normal = normalize(input.normal);
#endif

    let dielectricSpec : vec3<f32> = vec3<f32>(0.04, 0.04, 0.04);
    surface.f0 = mix(dielectricSpec, surface.albedo, vec3<f32>(surface.metallic, surface.metallic, surface.metallic));

#if ${defines.USE_OCCLUSION}
    surface.ao = textureSample(occlusionTexture, defaultSampler, input.texCoord).r * material.occlusionStrength;
#else
    surface.ao = 1.0;
#endif

    surface.emissive = material.emissiveFactor;
#if ${defines.USE_EMISSIVE_TEXTURE}
    surface.emissive = surface.emissive * textureSample(emissiveTexture, defaultSampler, input.texCoord).rgb;
#endif

    return surface;
  }
`; }

// Much of the shader used here was pulled from https://learnopengl.com/PBR/Lighting
// Thanks!
const PBRFunctions = `
let PI : f32 = 3.14159265359;

let LightType_Point : u32 = 0u;
let LightType_Spot : u32 = 1u;
let LightType_Directional : u32 = 2u;

struct PuctualLight {
  lightType : u32;
  pointToLight : vec3<f32>;
  range : f32;
  color : vec3<f32>;
  intensity : f32;
};

fn FresnelSchlick(cosTheta : f32, F0 : vec3<f32>) -> vec3<f32> {
  return F0 + (vec3<f32>(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

fn DistributionGGX(N : vec3<f32>, H : vec3<f32>, roughness : f32) -> f32 {
  let a : f32      = roughness*roughness;
  let a2 : f32     = a*a;
  let NdotH : f32  = max(dot(N, H), 0.0);
  let NdotH2 : f32 = NdotH*NdotH;

  let num : f32    = a2;
  let denom : f32  = (NdotH2 * (a2 - 1.0) + 1.0);

  return num / (PI * denom * denom);
}

fn GeometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let r : f32 = (roughness + 1.0);
  let k : f32 = (r*r) / 8.0;

  let num : f32   = NdotV;
  let denom : f32 = NdotV * (1.0 - k) + k;

  return num / denom;
}

fn GeometrySmith(N : vec3<f32>, V : vec3<f32>, L : vec3<f32>, roughness : f32) -> f32 {
  let NdotV : f32 = max(dot(N, V), 0.0);
  let NdotL : f32 = max(dot(N, L), 0.0);
  let ggx2 : f32  = GeometrySchlickGGX(NdotV, roughness);
  let ggx1 : f32  = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

fn rangeAttenuation(range : f32, distance : f32) -> f32 {
  if (range <= 0.0) {
      // Negative range means no cutoff
      return 1.0 / pow(distance, 2.0);
  }
  return clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0) / pow(distance, 2.0);
}

fn lightRadiance(light : PuctualLight, surface : SurfaceInfo) -> vec3<f32> {
  let L : vec3<f32> = normalize(light.pointToLight);
  let H : vec3<f32> = normalize(surface.v + L);
  let distance : f32 = length(light.pointToLight);

  // cook-torrance brdf
  let NDF : f32 = DistributionGGX(surface.normal, H, surface.roughness);
  let G : f32 = GeometrySmith(surface.normal, surface.v, L, surface.roughness);
  let F : vec3<f32> = FresnelSchlick(max(dot(H, surface.v), 0.0), surface.f0);

  let kD : vec3<f32> = (vec3<f32>(1.0, 1.0, 1.0) - F) * (1.0 - surface.metallic);

  let NdotL : f32 = max(dot(surface.normal, L), 0.0);

  let numerator : vec3<f32> = NDF * G * F;
  let denominator : f32 = max(4.0 * max(dot(surface.normal, surface.v), 0.0) * NdotL, 0.001);
  let specular : vec3<f32> = numerator / vec3<f32>(denominator, denominator, denominator);

  // add to outgoing radiance Lo
  let attenuation : f32 = rangeAttenuation(light.range, distance);
  let radiance : vec3<f32> = light.color * light.intensity * attenuation;
  return (kD * surface.albedo / vec3<f32>(PI, PI, PI) + specular) * radiance * NdotL;
}`;

export function PBRClusteredFragmentSource(defines) { return `
  ${ColorConversions}
  ${ProjectionUniforms}
  ${ClusterLightsStructs}
  ${MaterialUniforms}
  ${LightUniforms}
  ${TileFunctions}

  ${PBRSurfaceInfo(defines)}
  ${PBRFunctions}

  [[stage(fragment)]]
  fn main(input : FragmentInput) -> [[location(0)]] vec4<f32> {
    let surface : SurfaceInfo = GetSurfaceInfo(input);

    // reflectance equation
    var Lo : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let clusterIndex : u32 = getClusterIndex(input.fragCoord);
    let lightCount : u32 = clusterLights.lights[clusterIndex].count;

    for (var lightIndex : u32 = 0u; lightIndex < lightCount; lightIndex = lightIndex + 1u) {
      let i : u32 = clusterLights.lights[clusterIndex].indices[lightIndex];

      var light : PuctualLight;
      light.lightType = LightType_Point;
      light.pointToLight = globalLights.lights[i].position.xyz - input.worldPos;
      light.range = globalLights.lights[i].range;
      light.color = globalLights.lights[i].color;
      light.intensity = globalLights.lights[i].intensity;

      // calculate per-light radiance and add to outgoing radiance Lo
      Lo = Lo + lightRadiance(light, surface);
    }

    let ambient : vec3<f32> = globalLights.ambient * surface.albedo * surface.ao;
    let color : vec3<f32> = linearTosRGB(Lo + ambient + surface.emissive);
    return vec4<f32>(color, surface.baseColor.a);
  }`;
};