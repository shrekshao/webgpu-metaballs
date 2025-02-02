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

// This import installs hooks that help us output better formatted shader errors
import './wgsl-debug-helper.js';

import { Renderer } from '../renderer.js';
import { ProjectionUniformsSize, ViewUniformsSize, BIND_GROUP, ATTRIB_MAP } from './shaders/common.js';
import { WebGPUTextureLoader } from 'webgpu-texture-loader';
import { ClusteredLightManager } from './clustered-lights.js';
import { WebGPULightSprites } from './webgpu-light-sprites.js';
import { WebGPUglTF } from './webgpu-gltf.js';

import { GPUStats } from './gpu-stats.js';

import {
  MetaballWriteBuffer,
  MetaballNewBuffer,
  MetaballNewStagingBuffer,
  MetaballSingleStagingBuffer,
  MetaballStagingBufferRing,
  MetaballComputeRenderer,
  MetaballComputePointRenderer,
} from './webgpu-metaball-renderer.js';

import {
  TerrainComputeRenderer,
  TerrainComputePointRenderer,
} from './webgpu-terrain-renderer.js';

// import Gradient from 'javascript-color-gradient';

// function hexToRgb(hex) {
//   var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
//   return result ? {
//     r: parseInt(result[1], 16),
//     g: parseInt(result[2], 16),
//     b: parseInt(result[3], 16)
//   } : null;
// }

const MetaballMethods = {
  writeBuffer: MetaballWriteBuffer,
  newBuffer: MetaballNewBuffer,
  newStaging: MetaballNewStagingBuffer,
  singleStaging: MetaballSingleStagingBuffer,
  stagingRing: MetaballStagingBufferRing,
  gpuGenerated: MetaballComputeRenderer,
  pointCloud: MetaballComputePointRenderer,
};

const SAMPLE_COUNT = 4;
const DEPTH_FORMAT = "depth24plus";

export class WebGPURenderer extends Renderer {
  constructor() {
    super();

    this.sampleCount = SAMPLE_COUNT;
    this.depthFormat = DEPTH_FORMAT;

    this.context = this.canvas.getContext('webgpu');

    this.gpuStats = new GPUStats();
    this.metaballMethod = null;
  }

  async init() {
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    // Enable compressed textures if available
    const requiredFeatures = [];
    if (this.adapter.features.has('texture-compression-bc') != -1) {
      requiredFeatures.push('texture-compression-bc');
    }

    // Enable timestamp queries if available
    /*if (this.adapter.features.has('timestamp-query') != -1) {
      requiredFeatures.push('timestamp-query');
    }*/

    // This is just for testing. Ignore.
    const requiredLimits = { maxBindGroups: 4 };

    this.device = await this.adapter.requestDevice({requiredFeatures, requiredLimits});

    this.contextFormat = this.context.getPreferredFormat(this.adapter);
    this.renderBundleDescriptor = {
      colorFormats: [ this.contextFormat ],
      depthStencilFormat: DEPTH_FORMAT,
      sampleCount: SAMPLE_COUNT
    };

    this.textureLoader = new WebGPUTextureLoader(this.device);

    this.colorAttachment = {
      // view is acquired and set in onResize.
      view: undefined,
      // attachment is acquired and set in onFrame.
      resolveTarget: undefined,
      loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
      storeOp: 'discard', // Discards the multisampled view, not the resolveTarget
    };

    this.depthAttachment = {
      // view is acquired and set in onResize.
      view: undefined,
      depthLoadValue: 1.0,
      depthStoreOp: 'discard',
      stencilLoadValue: 0,
      stencilStoreOp: 'discard',
    };

    this.renderPassDescriptor = {
      colorAttachments: [this.colorAttachment],
      depthStencilAttachment: this.depthAttachment
    };

    this.bindGroupLayouts = {
      frame: this.device.createBindGroupLayout({
        label: `frame-bgl`,
        entries: [{
          binding: 0, // Projection uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: {},
        }, {
          binding: 1, // View uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          buffer: {}
        }, {
          binding: 2, // Light uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        }, {
          binding: 3, // Cluster Lights storage
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        }]
      }),

      material: this.device.createBindGroupLayout({
        label: `material-bgl`,
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {}
        },
        {
          binding: 1, // defaultSampler
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {}
        },
        {
          binding: 2, // baseColorTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 3, // normalTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 4, // metallicRoughnessTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 5, // occlusionTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 6, // emissiveTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        }]
      }),

      primitive: this.device.createBindGroupLayout({
        label: `primitive-bgl`,
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: {}
        }]
      }),

      metaball: this.device.createBindGroupLayout({
        label: `lava-bgl`,
        entries: [{
          binding: 0, // sampler
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {}
        },
        {
          binding: 1, // texture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        }]
      })
    };

    this.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [
        this.bindGroupLayouts.frame, // set 0
        this.bindGroupLayouts.material, // set 1
        this.bindGroupLayouts.primitive, // set 2
      ]
    });

    this.projectionBuffer = this.device.createBuffer({
      size: ProjectionUniformsSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    this.viewBuffer = this.device.createBuffer({
      size: ViewUniformsSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    this.lightsBuffer = this.device.createBuffer({
      size: this.lightManager.uniformArray.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });

    this.clusteredLights = new ClusteredLightManager(this);

    this.bindGroups = {
      frame: this.device.createBindGroup({
        layout: this.bindGroupLayouts.frame,
        entries: [{
          binding: 0,
          resource: {
            buffer: this.projectionBuffer,
          },
        }, {
          binding: 1,
          resource: {
            buffer: this.viewBuffer,
          },
        }, {
          binding: 2,
          resource: {
            buffer: this.lightsBuffer,
          },
        }, {
          binding: 3,
          resource: {
            buffer: this.clusteredLights.clusterLightsBuffer,
          }
        }],
      })
    }

    this.defaultSampler = this.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      // addressModeU: 'repeat',
      // addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    this.lightSprites = new WebGPULightSprites(this);

    this.metaballRenderer = null;
    this.metaballsNeedUpdate = true;





    // const h = this.colorGradientTextureSize = 256;
    // this.colorGradientBuffer = this.device.createBuffer({
    //   size: h * 1 * 4,
    //   usage: GPUBufferUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    //   // mappedAtCreation: true
    // });
    // this.colorGradientTexture = this.device.createTexture({
    //   size: [h, 1, 1],
    //   format: 'rgba8unorm',
    //   usage:
    //     GPUTextureUsage.TEXTURE_BINDING |
    //     GPUTextureUsage.COPY_DST
    // });

  }

  onResize(width, height) {
    if (!this.device) return;

    this.context.configure({
      device: this.device,
      format: this.contextFormat,
      size: {width, height}
    });

    const msaaColorTexture = this.device.createTexture({
      size: { width, height },
      sampleCount: SAMPLE_COUNT,
      format: this.contextFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.colorAttachment.view = msaaColorTexture.createView();

    const depthTexture = this.device.createTexture({
      size: { width, height },
      sampleCount: SAMPLE_COUNT,
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    this.depthAttachment.view = depthTexture.createView();

    // Update the Projection uniforms. These only need to be updated on resize.
    this.device.queue.writeBuffer(this.projectionBuffer, 0, this.frameUniforms.buffer, 0, ProjectionUniformsSize);

    // On every size change we need to re-compute the cluster grid.
    this.clusteredLights.updateClusterBounds();
  }

  async setScene(gltf) {
    super.setScene(gltf);
    this.scene = new WebGPUglTF(this, gltf);
    this.updateMetaballs(0);
  }

  setMetaballMethod(method) {
    // const rendererConstructor = MetaballMethods[method];
    // if (!rendererConstructor) {
    //   this.metaballRenderer = null;
    //   return;
    // }
    const rendererConstructor = TerrainComputeRenderer;
    // const rendererConstructor = TerrainComputePointRenderer;

    this.metaballRenderer = new rendererConstructor(this, this.marchingCubes.volume);
    this.metaballsNeedUpdate = true;
    this.metaballMethod = method;
  }

  async setMetaballStyle(style, colorSettings) {
    super.setMetaballStyle(style);

    // let texture;

    // if (this.metaballTexturePath == 'gradient') {
    //   // texture;
    //   // const t = await this.textureLoader.fromUrl('./media/textures/water.jpg', {colorSpace: 'sRGB'});
    //   // texture = t.texture;

    //   const data = new Uint8Array(this.colorGradientTextureSize * 4);

    //   this.device.queue.writeBuffer(this.colorGradientBuffer, 0, data);
    //   this.device.copyBufferToTexture(this.colorGradientBuffer, this.colorGradientTexture);

    // } else {
    //   const t = await this.textureLoader.fromUrl(this.metaballTexturePath, {colorSpace: 'sRGB'});
    //   texture = t.texture;
    // }

    const t = await this.textureLoader.fromUrl(this.metaballTexturePath, {colorSpace: 'sRGB'});
    const texture = t.texture;
    

    this.bindGroups.metaball = this.device.createBindGroup({
      layout: this.bindGroupLayouts.metaball,
      entries: [{
        binding: 0,
        resource: this.defaultSampler,
      }, {
        binding: 1,
        resource: texture.createView(),
      }],
    });
  }

  setMetaballStep(step) {
    super.setMetaballStep(step);
    this.setMetaballMethod(this.metaballMethod);
  }

  async updateMetaballs(timestamp) {
    if (this.metaballsNeedUpdate && this.metaballRenderer) {
      this.metaballsNeedUpdate = false;

      super.updateMetaballs(timestamp);

      this.metaballRenderer.updateMetaballs(this.metaballs, this.marchingCubes);

      /*if (!this.scene) {
        return;
      }*/

      await this.metaballRenderer.update(this.marchingCubes);
      this.metaballsNeedUpdate = true;
    }
  }

  onFrame(timestamp) {
    this.gpuStats.begin();

    // TODO: If we want multisampling this should attach to the resolveTarget,
    // but there seems to be a bug with that right now?
    this.colorAttachment.resolveTarget = this.context.getCurrentTexture().createView();

    // Update the View uniforms buffer with the values. These are used by most shader programs
    // and don't change for the duration of the frame.
    this.device.queue.writeBuffer(this.viewBuffer, 0, this.frameUniforms.buffer, ProjectionUniformsSize, ViewUniformsSize);

    // Update the light unform buffer with the latest values as well.
    this.device.queue.writeBuffer(this.lightsBuffer, 0, this.lightManager.uniformArray);

    const commandEncoder = this.device.createCommandEncoder({});
    this.clusteredLights.updateClusterLights(commandEncoder);

    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    if (this.scene && this.renderEnvironment) {
      this.scene.draw(passEncoder);
    }

    if (this.drawMetaballs && this.metaballRenderer && this.bindGroups.metaball) {
      // Draw metaballs.
      this.metaballRenderer.draw(passEncoder);
    }

    if (this.lightManager.render) {
      // Last, render a sprite for all of the lights. This is done using instancing so it's a single
      // call for every light.
      this.lightSprites.draw(passEncoder);
    }

    passEncoder.endPass();
    const commandBuffer = commandEncoder.finish();
    this.device.queue.submit([commandBuffer]);

    this.gpuStats.end();
  }
}