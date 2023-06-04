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

    out.textureCoordinate.x = vertexArray[vertexID].x;
    out.textureCoordinate.y = vertexArray[vertexID].y;

    return out;
}

fragment float4 renderFragment(RasterizerData  in           [[stage_in]],
                               texture2d<half> colorTexture [[texture(0)]])
{
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);
    
    // Calculate adjusted texture coordinates based on the viewport
    // float2 adjustedTexCoord = (in.textureCoordinate * viewportSize) + viewportOrigin;
    
    // Sample the texture and return the color to colorSample
    half4 colorSample = colorTexture.sample(textureSampler, in.textureCoordinate);
    
    colorSample = (half4)pow(colorSample, half4(2.2));
    
    return float4(colorSample);
}

struct CropInfo {
    uint2 croppedSourceOffset;
    uint2 croppedSourceExtent;
};

kernel void cropAndRotateKernel(texture2d<float> inTexture [[texture(0)]],
                                texture2d<float, access::write> outTexture [[texture(1)]],
                                constant CropInfo &cropInfo [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if ((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    
    float2 uvOutput = (float2)gid / float2(outTexture.get_width(), outTexture.get_height());
    
    float2 croppedSourceOffset = (float2)cropInfo.croppedSourceOffset;
    float2 croppedSourceExtent = (float2)cropInfo.croppedSourceExtent;
    
    float2 rotatedUvOutput = float2(uvOutput.y, 1.0-uvOutput.x);
    float2 uvInput = (croppedSourceOffset + croppedSourceExtent * rotatedUvOutput) / float2(inTexture.get_width(), inTexture.get_height());
    
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);
    
    float4 inColor = inTexture.sample(textureSampler, uvInput);
    
    outTexture.write(inColor, gid);
}

typedef struct
{
    float2 position [[attribute(0)]];
    float2 uvs [[attribute(1)]];
} FontVertexIn;

typedef struct
{
    float4 clipPosition [[position]];
    float2 uvs;
} FontVertexOut;

vertex FontVertexOut
renderFontVertex(FontVertexIn in [[stage_in]],
                 uint vertexID [[vertex_id]])
{
    FontVertexOut out;
    
    out.clipPosition = float4(in.position, 0.0, 1.0);
    
    out.uvs = in.uvs;
    
    return out;
}

fragment float4 renderFontFragment(FontVertexOut in [[stage_in]],
                                   texture2d<float> fontAtlas [[texture(0)]])
{
    constexpr sampler textureSampler (mag_filter::nearest,
                                      min_filter::nearest);
    
    float4 inColor = fontAtlas.sample(textureSampler, in.uvs);
    if (inColor.a < 0.5)
        discard_fragment();
    
    return inColor;
}

typedef struct
{
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
} BoxVertexIn;

typedef struct
{
    float4 clipPosition [[position]];
    float4 color;
} BoxVertexOut;

vertex BoxVertexOut
renderBoxVertex(BoxVertexIn in [[stage_in]],
                 uint vertexID [[vertex_id]])
{
    BoxVertexOut out;
    
    out.clipPosition = float4(in.position.xy, 0.0, 1.0);
    
    out.color = in.color;
    
    return out;
}

fragment float4 renderBoxFragment(BoxVertexOut in [[stage_in]])
{
    return in.color;
}
