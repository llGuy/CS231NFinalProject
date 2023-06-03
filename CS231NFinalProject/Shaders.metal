//
//  Shaders.metal
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/1/23.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

struct RasterizerData
{
    // The [[position]] attribute qualifier of this member indicates this value is the clip space
    //   position of the vertex wen this structure is returned from the vertex shader
    float4 clipSpacePosition [[position]];

    // Since this member does not have a special attribute qualifier, the rasterizer will
    //   interpolate its value with values of other vertices making up the triangle and
    //   pass that interpolated value to the fragment shader for each fragment in that triangle;
    float2 textureCoordinate;
};

// Vertex Function
vertex RasterizerData
renderVertex(uint                   vertexID             [[ vertex_id ]])
{
    RasterizerData out;
    
    float2 vertexArray[6] = {
        { 1.f, 1.f },
        { 0.f, 1.f },
        { 0.f, 0.f },
        
        { 1.f, 1.f },
        { 0.f, 0.f },
        { 1.f, 0.f },
    };

    out.clipSpacePosition.xy = vertexArray[vertexID].xy * 2.0f - float2(1.0f);
    out.clipSpacePosition.y *= -1.0f;
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;

    out.textureCoordinate.x = vertexArray[vertexID].y;
    out.textureCoordinate.y = vertexArray[vertexID].x;
    out.textureCoordinate.y = 1.0 - out.textureCoordinate.y;

    return out;
}

fragment float4 renderFragment(RasterizerData  in           [[stage_in]],
                               texture2d<half> colorTexture [[ texture(0) ]])
{
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);

    // Sample the texture and return the color to colorSample
    half4 colorSample = colorTexture.sample (textureSampler, in.textureCoordinate);
    
    colorSample = (half4)pow(colorSample, half4(2.2));
    
    return float4(colorSample);
}
