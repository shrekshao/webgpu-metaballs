// Copyright 2021 Brandon Jones
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

import { ProjectionUniforms, ViewUniforms, ColorConversions, ATTRIB_MAP } from './common.js';

export const MetaballVertexSource = `
  ${ProjectionUniforms}
  ${ViewUniforms}

  struct VertexInput {
    [[location(${ATTRIB_MAP.POSITION})]] position : vec3<f32>;
  };

  struct VertexOutput {
    [[location(0)]] worldPosition : vec3<f32>;
    [[builtin(position)]] position : vec4<f32>;
  };

  [[stage(vertex)]]
  fn vertexMain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.worldPosition = input.position;
    output.position = projection.matrix * view.matrix * vec4<f32>(input.position, 1.0);
    return output;
  }
`;

export const MetaballFragmentSource = `
  struct VertexOutput {
    [[location(0)]] worldPosition : vec3<f32>;
  };

  [[stage(fragment)]]
  fn fragmentMain(input : VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>((2.5 - input.worldPosition.y) / 1.0, input.worldPosition.y / 10.0, 0.0, 1.0);
  }
`;