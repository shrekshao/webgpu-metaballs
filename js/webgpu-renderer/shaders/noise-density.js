import {
    WORKGROUP_SIZE,
    IsosurfaceVolume,
} from './metaball.js'

import {NoiseComputeSource} from './noise.js';

const Density = `
let numThreads = 8u;

// RWStructuredBuffer<float4> points;
// int numPointsPerAxis;
// float boundsSize;
// float3 centre;
// float3 offset;
// float spacing;
// float3 worldSize;

// int indexFromCoord(uint x, uint y, uint z) {
//     return z * numPointsPerAxis * numPointsPerAxis + y * numPointsPerAxis + x;
// }
`;



// testing snoise function
export const NoiseDensityComputeSource = `

${NoiseComputeSource}



  ${IsosurfaceVolume}
  [[group(0), binding(0)]] var<storage, read_write> volume : IsosurfaceVolume;

  fn positionAt(index : vec3<u32>) -> vec3<f32> {
    return volume.min + (volume.step * vec3<f32>(index.xyz));
  }

  [[stage(compute), workgroup_size(${WORKGROUP_SIZE[0]}, ${WORKGROUP_SIZE[1]}, ${WORKGROUP_SIZE[2]})]]
  fn computeMain([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    let position = positionAt(global_id);
    let valueIndex = global_id.x +
                    (global_id.y * volume.size.x) +
                    (global_id.z * volume.size.x * volume.size.y);

    volume.values[valueIndex] = snoise(position);
  }
`;

export const NoiseDensityTerrainComputeSource = `

${NoiseComputeSource}

  ${IsosurfaceVolume}
  [[group(0), binding(0)]] var<storage, read_write> volume : IsosurfaceVolume;

// Noise settings

[[block]] struct TerrainConfig {
    octaves: i32;

    lacunarity: f32;
    persistence: f32;
    noiseScale: f32;
    noiseWeight: f32;

    floorOffset: f32;
    weightMultiplier: f32;
    hardFloor: f32;
    hardFloorWeight: f32;

    
    // closeEdges: bool;
};
[[group(0), binding(1)]] var<uniform> config : TerrainConfig;

[[block]] struct WorldConfig {
    // From density.compute
    offset: vec3<f32>;
    spacing: f32;

    // 4, 5, 6
    offsets: vec3<f32>;
    pad1: f32;

    // 8, 9, 10
    worldSize: vec3<f32>;
    pad2: f32;
};
[[group(0), binding(2)]] var<uniform> worldConfig : WorldConfig;

  fn positionAt(index : vec3<u32>) -> vec3<f32> {
    return volume.min + (volume.step * vec3<f32>(index.xyz));
  }

  [[stage(compute), workgroup_size(${WORKGROUP_SIZE[0]}, ${WORKGROUP_SIZE[1]}, ${WORKGROUP_SIZE[2]})]]
  fn computeMain([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    let pos = positionAt(global_id);
    let valueIndex = global_id.x +
                    (global_id.y * volume.size.x) +
                    (global_id.z * volume.size.x * volume.size.y);

    var offsetNoise = 0.0;
    var noise = 0.0;

    // var frequency = config.noiseScale / 100.0;
    var frequency = config.noiseScale;
    var amplitude = 1.0;
    var weight = 1.0;
    for (var j = 0; j < config.octaves; j = j + 1) {
        let n = snoise((pos+offsetNoise) * frequency + worldConfig.offsets[j] + worldConfig.offset);
        var v = 1.0 - abs(n);
        v = v*v;
        v = v*weight;
        weight = max(min(v * config.weightMultiplier, 1.0), 0.0);
        noise = noise + v * amplitude;
        amplitude = amplitude * config.persistence;
        frequency = frequency * config.lacunarity;
    }

    // var finalVal = -(pos.y + config.floorOffset) + noise * config.noiseWeight + (pos.y%params.x) * params.y;
    var finalVal = -(pos.y + config.floorOffset) + noise * config.noiseWeight;

    if (pos.y < config.hardFloor) {
        finalVal = finalVal + config.hardFloorWeight;
    }

    // if (config.closeEdges) {
        let edgeOffset = abs(pos * 2.0) - worldConfig.worldSize + worldConfig.spacing * 0.5;
        let edgeWeight = clamp(sign(max(max(edgeOffset.x,edgeOffset.y),edgeOffset.z)), 0.0, 1.0);
        finalVal = finalVal * (1.0 - edgeWeight) - 100.0 * edgeWeight;
    // }

    // volume.values[valueIndex] = finalVal;
    volume.values[valueIndex] = -finalVal;
  }
`;



export const NewTerrainComputeSource = `

${NoiseComputeSource}

  ${IsosurfaceVolume}
  [[group(0), binding(0)]] var<storage, read_write> volume : IsosurfaceVolume;

// Noise settings

[[block]] struct TerrainConfig {
    octaves: i32;

    lacunarity: f32;
    persistence: f32;
    noiseScale: f32;
    noiseWeight: f32;

    floorOffset: f32;
    weightMultiplier: f32;
    hardFloor: f32;
    hardFloorWeight: f32;

    
    // closeEdges: bool;
};
[[group(0), binding(1)]] var<uniform> config : TerrainConfig;

[[block]] struct MeshConfig {
    // From density.compute
    offset: vec3<f32>;
    spacing: f32;

    // 4, 5, 6
    offsets: vec3<f32>;
    pad1: f32;

    // 8, 9, 10
    worldSize: vec3<f32>;
    pad2: f32;
};
[[group(0), binding(2)]] var<uniform> meshConfig : MeshConfig;



  fn positionAt(index : vec3<u32>) -> vec3<f32> {
    return volume.min + (volume.step * vec3<f32>(index.xyz));
  }

  [[stage(compute), workgroup_size(${WORKGROUP_SIZE[0]}, ${WORKGROUP_SIZE[1]}, ${WORKGROUP_SIZE[2]})]]
  fn computeMain([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // let pos = positionAt(global_id);
    let pos = positionAt(global_id) + meshConfig.offset;
    let valueIndex = global_id.x +
                    (global_id.y * volume.size.x) +
                    (global_id.z * volume.size.x * volume.size.y);
    var finalVal = 0.0;

    finalVal = config.floorOffset - pos.y;

    var amplitude = 1.0;
    var weight = 1.0;
    var frequency = config.noiseScale;
    var noise = 0.0;
    for (var j = 0; j < config.octaves; j = j + 1) {

      noise = noise + amplitude * snoise(pos * frequency);
      amplitude = amplitude * config.persistence;
      frequency = frequency * config.lacunarity;
    }

    finalVal = finalVal + noise * config.noiseWeight;
    if (pos.y < config.hardFloor) {
      // used for deepest canyon
      finalVal = finalVal + config.hardFloorWeight;
    }
    

    // volume.values[valueIndex] = finalVal;
    volume.values[valueIndex] = -finalVal;
  }
`;