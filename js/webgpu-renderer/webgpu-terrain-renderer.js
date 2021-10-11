// Copyright 2021 Brandon Jones
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import { BIND_GROUP, ATTRIB_MAP } from './shaders/common.js';
import {
  MetaballVertexSource,
  MetaballFragmentSource,
  MetaballFieldComputeSource,
  MarchingCubesComputeSource,
  MetaballVertexPointSource,
  MetaballFragmentPointSource,
  WORKGROUP_SIZE
} from './shaders/metaball.js';

import {
  NoiseDensityComputeSource,
  NoiseDensityTerrainComputeSource,
  NewTerrainComputeSource,
} from './shaders/noise-density.js';

import {
  MarchingCubesEdgeTable,
  MarchingCubesTriTable,
} from "../marching-cubes-tables.js";

const MAX_METABALLS = 32;

// Common assets used by every variant of the Metaball renderer
class WebGPUTerrainRendererBase {
  constructor(renderer, volume, createBuffers=true) {
    this.renderer = renderer;
    this.device = renderer.device;
    this.volume = volume;

    // Computes buffer sizes large enough for the maximum possible number of triangles in that volume
    this.marchingCubeCells = (volume.width-1) * (volume.height-1) * (volume.depth-1);
    this.vertexBufferSize = (Float32Array.BYTES_PER_ELEMENT * 3) * 12 * this.marchingCubeCells;
    this.indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * 15 * this.marchingCubeCells;

    //this.vertexBufferSize = METABALLS_VERTEX_BUFFER_SIZE;
    //this.indexBufferSize = METABALLS_INDEX_BUFFER_SIZE;

    this.indexCount = 0;

    // Metaball resources
    if (createBuffers) {
      this.vertexBuffer = this.device.createBuffer({
        size: this.vertexBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      });

      this.normalBuffer = this.device.createBuffer({
        size: this.vertexBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      });

      this.indexBuffer = this.device.createBuffer({
        size: this.indexBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
      });
    }

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.renderer.bindGroupLayouts.frame,
          this.renderer.bindGroupLayouts.metaball
        ]
      }),
      vertex: {
        module: this.device.createShaderModule({ code: MetaballVertexSource }),
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: 12,
          attributes: [{
            shaderLocation: ATTRIB_MAP.POSITION,
            format: 'float32x3',
            offset: 0
          }],
        }, {
          arrayStride: 12,
          attributes: [{
            shaderLocation: ATTRIB_MAP.NORMAL,
            format: 'float32x3',
            offset: 0
          }],
        }]
      },
      fragment: {
        module: this.device.createShaderModule({ code: MetaballFragmentSource }),
        entryPoint: "fragmentMain",
        targets: [{
          format: this.renderer.contextFormat,
        }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: this.renderer.renderBundleDescriptor.depthStencilFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: {
        count: this.renderer.renderBundleDescriptor.sampleCount
      }
    });
  }

  updateMetaballs(metaballs, marchingCubes) {
    marchingCubes.updateVolume(metaballs);
  }

  update(marchingCubes) {
    throw new Error('update must be implemented in a class that extends WebGPUMetaballRendererBase');
  }

  draw(passEncoder) {
    if (this.indexCount) {
      passEncoder.setPipeline(this.pipeline);
      passEncoder.setBindGroup(BIND_GROUP.Frame, this.renderer.bindGroups.frame);
      passEncoder.setBindGroup(1, this.renderer.bindGroups.metaball);
      passEncoder.setVertexBuffer(0, this.vertexBuffer);
      passEncoder.setVertexBuffer(1, this.normalBuffer);
      passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
      passEncoder.drawIndexed(this.indexCount, 1, 0, 0, 0);
    }
  }
}

/**
 * For certain types of algorithmically generated data, it may be possible to generate the data in
 * a compute shader. This allows the data to be directly populated into the GPU-side buffer with
 * no copies, and as a result can be the most efficent route. Not every data set is well suited for
 * generation within a compute shader, however, and as such this method is only practical for data
 * which is algorithmically generated (for example: particle effects).
 * 
 * Advantages:
 *  - Does not require staging buffers.
 *  - No CPU or GPU-side copies.
 *  - Takes advantage of GPU hardware, parallelism.
 *
 * Disadvantages:
 *  - Potentially high complexity.
 *  - Not all algorithms are well suited for implementation as a compute shader.
 *  - May still require copy of external data for use in the shader.
 */

export class TerrainComputeRenderer extends WebGPUTerrainRendererBase {
  constructor(renderer, volume) {
    super(renderer, volume, false);

    // Fill a buffer with the lookup tables we need for the marching cubes algorithm.
    this.tablesBuffer = this.device.createBuffer({
      size: (MarchingCubesEdgeTable.length + MarchingCubesTriTable.length) * 4,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });

    const tablesArray = new Int32Array(this.tablesBuffer.getMappedRange());
    tablesArray.set(MarchingCubesEdgeTable);
    tablesArray.set(MarchingCubesTriTable, MarchingCubesEdgeTable.length);
    this.tablesBuffer.unmap();

    this.volumeElements = volume.width * volume.height * volume.depth;
    this.volumeBufferSize = (Float32Array.BYTES_PER_ELEMENT * 12) +
                            (Uint32Array.BYTES_PER_ELEMENT * 4) +
                            (Float32Array.BYTES_PER_ELEMENT * this.volumeElements);

    this.volumeBuffer = this.device.createBuffer({
      size: this.volumeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    // Fill the buffer with information about the isosurface volume.
    const volumeMappedArray = this.volumeBuffer.getMappedRange();
    const volumeFloat32 = new Float32Array(volumeMappedArray);
    const volumeSize = new Uint32Array(volumeMappedArray, 48, 3);

    volumeFloat32[0] = volume.xMin;
    volumeFloat32[1] = volume.yMin;
    volumeFloat32[2] = volume.zMin;

    volumeFloat32[4] = volume.xMax;
    volumeFloat32[5] = volume.yMax;
    volumeFloat32[6] = volume.zMax;

    volumeFloat32[8] = volume.xStep;
    volumeFloat32[9] = volume.yStep;
    volumeFloat32[10] = volume.zStep;

    volumeSize[0] = volume.width;
    volumeSize[1] = volume.height;
    volumeSize[2] = volume.depth;

    // this is threshold
    // volumeFloat32[15] = 40; // Threshold. TODO: Should be dynamic.
    volumeFloat32[15] = 0; // Threshold. TODO: Should be dynamic.


    // const yMid = 0.5 * (volume.yMin + volume.yMax);
    // const yScale = volume.yMax - volume.yMin;

    // // Init random values for volume.values
    // const volumeValueArray = new Float32Array(volumeMappedArray, 64, this.volumeElements);
    // for (let z = 0; z < volume.depth; z++) {
    //   // for (let y = 0; y < volume.yStep; y++) {
    //   for (let x = 0; x < volume.width; x++) {
    //     const heightValue = Math.random() * yScale * 0.1 + yMid;
    //     for (let y = 0; y < volume.height; y++) {
    //       let i = x + y * volume.width + z * volume.width * volume.height;
    //       volumeValueArray[i] = y * volume.yStep + volume.yMin - heightValue;
    //       // console.log(volumeValueArray[i]);
    //       // volumeValueArray[i] = Math.random() * 2.0 - 1.0;
    //     }
    //   }
    // }
    // // for (let i = 0; i < this.volumeElements; i++) {
    // //   volumeValueArray[i] = Math.random() * 2.0 - 1.0;
    // // }

    this.volumeBuffer.unmap();

    // Mesh resources
    this.marchingCubeCells = (volume.width-1) * (volume.height-1) * (volume.depth-1);
    this.vertexBufferSize = (Float32Array.BYTES_PER_ELEMENT * 3) * 12 * this.marchingCubeCells;
    this.indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * 15 * this.marchingCubeCells;

    this.vertexBuffer = this.device.createBuffer({
      size: this.vertexBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });

    this.normalBuffer = this.device.createBuffer({
      size: this.vertexBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });

    this.indexBuffer = this.device.createBuffer({
      size: this.indexBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
    });

    this.indirectArray = new Uint32Array(9);
    this.indirectArray[0] = 4;
    this.indirectBuffer = this.device.createBuffer({
      size: this.indirectArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // Create compute pipeline that handles the metaball isosurface.
    const terrainModule = this.device.createShaderModule({
      label: 'Metaball Isosurface Compute Shader',
      // code: MetaballFieldComputeSource
      // code: NoiseDensityComputeSource
      // code: NoiseDensityTerrainComputeSource
      code: NewTerrainComputeSource
    });

    this.noiseSettings = {
      numOctaves: 6,

      lacunarity: 2.0,

      persistence: 0.52,
      noiseScale: 0.5,
      noiseWeight: 1.0,
      floorOffset: 1.0,

      weightMultiplier: 3.61,
      hardFloor: 0.5,
      hardFloorWeight: 3.05,
    };

    this.configUniformBuffer = (() => {
      const buffer = this.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12,
        mappedAtCreation: true,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      // new Float32Array(buffer.getMappedRange())[0] = settings.numLights;

      const configBufferMappedArray = buffer.getMappedRange();
      this.fillConfigBuffer(this.noiseSettings, configBufferMappedArray);

      
      buffer.unmap();
      return buffer;
    })();

    this.offsetsBuffer = (() => {
      const buffer = this.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12,
        mappedAtCreation: true,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const configF32 = new Float32Array(buffer.getMappedRange());

      // offset
      configF32[0] = 0;
      configF32[1] = 0;
      configF32[2] = 0;

      // spacing
      configF32[3] = volume.xStep;     // assume xStep, yStep, zStep is same

      // offsets x, y, z
      const offsetRange = 1000;
      // // const offsetRange = 1;
      // configF32[4] = offsetRange * (Math.random() * 2 - 1);
      // configF32[5] = offsetRange * (Math.random() * 2 - 1);
      // configF32[6] = offsetRange * (Math.random() * 2 - 1);
      configF32[4] = 0;
      configF32[5] = 0;
      configF32[6] = 0;


      // worldSize x, y, z
      configF32[8] = volume.xMax - volume.xMin;
      configF32[9] = volume.yMax - volume.yMin;
      configF32[10] = volume.zMax - volume.zMin;

      // console.log(configF32);
      console.log(volume);

      buffer.unmap();
      return buffer;
    })();

    this.device.createComputePipelineAsync({
      label: 'Metaball Isosurface Compute Pipeline',
      compute: { module: terrainModule, entryPoint: 'computeMain' }
    }).then((pipeline) => {
      this.terrainComputePipeline = pipeline;
      this.terrainComputeBindGroup = this.device.createBindGroup({
        layout: this.terrainComputePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.volumeBuffer,
            },
          },
          {
            binding: 1,
            resource: {
              buffer: this.configUniformBuffer,
            },
          },
          {
            binding: 2,
            resource: {
              buffer: this.offsetsBuffer,
            },
          },
        ],
      });
    });

    // Create compute pipeline that handles the marching cubes triangulation.
    const marchingCubesModule = this.device.createShaderModule({
      label: 'Marching Cubes Compute Shader',
      code: MarchingCubesComputeSource
    });

    this.device.createComputePipelineAsync({
      label: 'Marching Cubes Compute Pipeline',
      compute: { module: marchingCubesModule, entryPoint: 'computeMain' }
    }).then((pipeline) => {;
      this.marchingCubesComputePipeline = pipeline;
      this.marchingCubesComputeBindGroup = this.device.createBindGroup({
        layout: this.marchingCubesComputePipeline.getBindGroupLayout(0),
        entries: [{
          binding: 0,
          resource: {
            buffer: this.tablesBuffer,
          },
        }, {
          binding: 1,
          resource: {
            buffer: this.volumeBuffer,
          },
        }, {
          binding: 2,
          resource: {
            buffer: this.vertexBuffer,
          },
        }, {
          binding: 3,
          resource: {
            buffer: this.normalBuffer,
          },
        }, {
          binding: 4,
          resource: {
            buffer: this.indexBuffer,
          },
        }, {
          binding: 5,
          resource: {
            buffer: this.indirectBuffer,
          },
        }],
      });
    });
  }

  updateMetaballs(metaballs, marchingCubes) {

    // Zero out the indirect buffer every time.
    this.device.queue.writeBuffer(this.indirectBuffer, 0, this.indirectArray);
  }

  fillConfigBuffer(settings, arrayBuffer) {
    const configF32 = new Float32Array(arrayBuffer, 4);
    const configOctaves = new Int32Array(arrayBuffer, 0, 1);

    configOctaves[0] = settings.numOctaves;

    configF32[0] = settings.lacunarity;
    configF32[1] = settings.persistence;
    configF32[2] = settings.noiseScale;
    configF32[3] = settings.noiseWeight;

    configF32[4] = settings.floorOffset;
    configF32[5] = settings.weightMultiplier;
    configF32[6] = settings.hardFloor;
    configF32[7] = settings.hardFloorWeight;
  }

  updateNoiseSettings(settings) {
    const b = new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT * 12);
    this.noiseSettings = settings;  // temp hack
    this.fillConfigBuffer(settings, b);

    this.device.queue.writeBuffer(
      this.configUniformBuffer,
      0,
      b
    );
  }

  update(marchingCubes) {
    // Update the volume buffer with the latest isosurface values.
    //this.device.queue.writeBuffer(this.volumeBuffer, 64, marchingCubes.volume.values, 0, this.volumeElements);

    // Run the compute shader to fill the position/normal/index buffers.
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    const dispatchSize = [
      this.volume.width / WORKGROUP_SIZE[0],
      this.volume.height / WORKGROUP_SIZE[1],
      this.volume.depth / WORKGROUP_SIZE[2]
    ];

    if (this.terrainComputePipeline) {
      passEncoder.setPipeline(this.terrainComputePipeline);
      passEncoder.setBindGroup(0, this.terrainComputeBindGroup);
      passEncoder.dispatch(...dispatchSize);
    }

    if (this.marchingCubesComputePipeline) {
      passEncoder.setPipeline(this.marchingCubesComputePipeline);
      passEncoder.setBindGroup(0, this.marchingCubesComputeBindGroup);
      passEncoder.dispatch(...dispatchSize);
    }

    passEncoder.endPass();
    this.device.queue.submit([commandEncoder.finish()]);

    this.indexCount = this.indexBufferSize / Uint32Array.BYTES_PER_ELEMENT;
  }

  /*draw(passEncoder) {
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(BIND_GROUP.Frame, this.renderer.bindGroups.frame);
    passEncoder.setBindGroup(1, this.renderer.bindGroups.metaball);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setVertexBuffer(1, this.normalBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexedIndirect(this.indirectBuffer, 4);
  }*/

  // TODO: DrawIndirect once the buffers are dynamically packed.
}

export class TerrainComputePointRenderer extends TerrainComputeRenderer {
  constructor(renderer, volume) {
    super(renderer, volume);

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.renderer.bindGroupLayouts.frame,
          this.renderer.bindGroupLayouts.metaball
        ]
      }),
      vertex: {
        module: this.device.createShaderModule({ code: MetaballVertexPointSource }),
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: 12,
          stepMode: 'instance',
          attributes: [{
            shaderLocation: ATTRIB_MAP.POSITION,
            format: 'float32x3',
            offset: 0,
            
          }],
        }, {
          arrayStride: 12,
          stepMode: 'instance',
          attributes: [{
            shaderLocation: ATTRIB_MAP.NORMAL,
            format: 'float32x3',
            offset: 0,
          }],
        }]
      },
      fragment: {
        module: this.device.createShaderModule({ code: MetaballFragmentPointSource }),
        entryPoint: "fragmentMain",
        targets: [{
          format: this.renderer.contextFormat,
        }]
      },
      primitive: {
        topology: 'triangle-strip',
        stripIndexFormat: 'uint32',
        cullMode: 'none',
      },
      depthStencil: {
        format: this.renderer.renderBundleDescriptor.depthStencilFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: {
        count: this.renderer.renderBundleDescriptor.sampleCount
      }
    });
  }

  draw(passEncoder) {
    if (this.indexCount) {
      passEncoder.setPipeline(this.pipeline);
      passEncoder.setBindGroup(BIND_GROUP.Frame, this.renderer.bindGroups.frame);
      passEncoder.setBindGroup(1, this.renderer.bindGroups.metaball);
      passEncoder.setVertexBuffer(0, this.vertexBuffer);
      passEncoder.setVertexBuffer(1, this.normalBuffer);
      passEncoder.drawIndirect(this.indirectBuffer, 0);
    }
  }
}