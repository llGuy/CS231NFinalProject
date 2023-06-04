//
//  FontRenderer.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef FontRenderer_h
#define FontRenderer_h

#if 1

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

struct FontRenderInfo
{
    id<MTLBuffer> vertexBuffer;
    uint32_t vertexCount;
};

struct BoxRenderInfo
{
    id<MTLBuffer> vertexBuffer;
    uint32_t vertexCount;
};

@interface FontRenderer : NSObject

-(nonnull instancetype)initWithDevice:(nonnull id<MTLDevice>)device defaultLibrary:(nonnull id<MTLLibrary>)defaultLib view:(nonnull MTKView *)view;

-(void)pushText:(nonnull struct FontRenderInfo *)info text:(nonnull char *)text position:(vector_float2)pos viewport:(vector_uint2)viewport;
-(struct FontRenderInfo)makeFontRenderInfo:(nonnull id<MTLDevice>)device;
-(void)flushFonts:(nonnull id<MTLRenderCommandEncoder>)cmdbuf fontRenderInfo:(nonnull struct FontRenderInfo *)renderInfo;

-(struct BoxRenderInfo)makeBoxRenderInfo:(nonnull id<MTLDevice>)device;
-(void)pushBoxPixelCoords:(nonnull struct BoxRenderInfo *)info position:(vector_int2)pos size:(vector_int2)size color:(vector_float4)color viewport:(vector_uint2)viewport;
-(void)flushBoxes:(nonnull id<MTLRenderCommandEncoder>)cmdbuf boxRenderInfo:(nonnull struct BoxRenderInfo *)renderInfo;

@end

#endif

#endif /* FontRenderer_h */
