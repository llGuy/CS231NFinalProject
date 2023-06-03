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

kernel void
renderKernel(texture2d<half, access::read>  inTexture  [[texture(0)]],
             texture2d<half, access::write> outTexture [[texture(1)]],
             uint2                          gid [[thread_position_in_grid]])
{
    // Check if the pixel is within the bounds of the output texture
    if ((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }

    half4 inColor  = inTexture.read(gid);
    outTexture.write(inColor, gid);
}

