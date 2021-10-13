export const NoiseComputeSource = `
// WGSL noise shader based on
// Noise Shader Library for Unity - https://github.com/keijiro/NoiseShader
//
// Original work (webgl-noise) Copyright (C) 2011 Ashima Arts.
// Translation and modification was made by Keijiro Takahashi.
//
// This shader is based on the webgl-noise GLSL shader. For further details
// of the original shader, please see the following description from the
// original source code.
//

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

fn mod289v3(x : vec3<f32>) -> vec3<f32>
{
    return x - floor(x / 289.0) * 289.0;
}

fn mod289v4(x : vec4<f32>) -> vec4<f32> 
{
    return x - floor(x / 289.0) * 289.0;
}

fn permute(x : vec4<f32>) -> vec4<f32> 
{
    return mod289v4((x * 34.0 + 1.0) * x);
}

fn taylorInvSqrt(r : vec4<f32>) -> vec4<f32> 
{
    return 1.79284291400159 - r * 0.85373472095314;
}

fn snoise(v : vec3<f32>) -> f32
{
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    var i  = floor(v + dot(v, C.yyy));
    let x0 = v   - i + dot(i, C.xxx);

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    // x1 = x0 - i1  + 1.0 * C.xxx;
    // x2 = x0 - i2  + 2.0 * C.xxx;
    // x3 = x0 - 1.0 + 3.0 * C.xxx;
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - 0.5;

    // Permutations
    i = mod289v3(i); // Avoid truncation effects in permutation
    let p =
      permute(permute(permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
                            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
                            + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    let j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    let x_ = floor(j / 7.0);
    let y_ = floor(j - 7.0 * x_);  // mod(j,N)

    let x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    let y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    //let s0 = vec4<f32>(lessThan(b0, 0.0)) * 2.0 - 1.0;
    //let s1 = vec4<f32>(lessThan(b1, 0.0)) * 2.0 - 1.0;
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var g0 = vec3<f32>(a0.xy, h.x);
    var g1 = vec3<f32>(a0.zw, h.y);
    var g2 = vec3<f32>(a1.xy, h.z);
    var g3 = vec3<f32>(a1.zw, h.w);

    // Normalise gradients
    let norm = taylorInvSqrt(vec4<f32>(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 = g0 * norm.x;
    g1 = g1 * norm.y;
    g2 = g2 * norm.z;
    g3 = g3 * norm.w;

    // Mix final noise value
    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    m = m * m;

    let px = vec4<f32>(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * dot(m, px);
}

fn snoise_grad(v : vec3<f32>) -> vec4<f32>
{
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    var i  = floor(v + dot(v, C.yyy));
    let x0 = v   - i + dot(i, C.xxx);

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    // x1 = x0 - i1  + 1.0 * C.xxx;
    // x2 = x0 - i2  + 2.0 * C.xxx;
    // x3 = x0 - 1.0 + 3.0 * C.xxx;
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - 0.5;

    // Permutations
    i = mod289v3(i); // Avoid truncation effects in permutation
    let p =
      permute(permute(permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
                            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
                            + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    let j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    let x_ = floor(j / 7.0);
    let y_ = floor(j - 7.0 * x_);  // mod(j,N)

    let x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    let y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    //let s0 = vec4<f32>(lessThan(b0, 0.0)) * 2.0 - 1.0;
    //let s1 = vec4<f32>(lessThan(b1, 0.0)) * 2.0 - 1.0;
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var g0 = vec3<f32>(a0.xy, h.x);
    var g1 = vec3<f32>(a0.zw, h.y);
    var g2 = vec3<f32>(a1.xy, h.z);
    var g3 = vec3<f32>(a1.zw, h.w);

    // Normalise gradients
    let norm = taylorInvSqrt(vec4<f32>(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 = g0 * norm.x;
    g1 = g1 * norm.y;
    g2 = g2 * norm.z;
    g3 = g3 * norm.w;

    // Compute noise and gradient at P
    let m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    let m2 = m * m;
    let m3 = m2 * m;
    let m4 = m2 * m2;
    let grad =
        -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
        -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
        -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
        -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    let px = vec4<f32>(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * vec4<f32>(grad, dot(m4, px));
}
`;