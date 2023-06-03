//
//  Renderer.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/1/23.
//

#import <simd/simd.h>
#import <ModelIO/ModelIO.h>

#import "Camera.h"
#import "Renderer.h"
#import "Def.h"

// Include header shared between C code here, which executes Metal API commands, and .metal files
#import "ShaderTypes.h"

#include "Time.h"

struct FrameData
{
    id<MTLTexture> cameraOutput;
    id<MTLTexture> croppedCameraOutput;
};

@implementation Renderer
{
    // Waits for a free camera slot, initialized to MAX_FRAMES_IN_FLIGHT
    dispatch_semaphore_t mInFlightSemaphore;
    
    // The index of the target, in [0, MAX_FRAMES_IN_FLIGHT)
    uint8_t mCurrentFrame;
    
    // Metal library instance.
    id <MTLDevice> mDevice;
    
    // We submit processing commands to this queue.
    id <MTLCommandQueue> mCommandQueue;

    //
    id <MTLRenderPipelineState> mRenderOutputPipeline;
    
    //
    id <MTLComputePipelineState> mCropPipeline;
    
    // Camera outputs.
    struct FrameData mFrames[MAX_FRAMES_IN_FLIGHT];
    
    // Encapsulates camera capture.
    Camera *mCamera;
    
    struct TimeStamp *mPrevTimeStamp;
    struct TimeStamp *mCurrentTimeStamp;
    
    vector_uint2 mViewportSize;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if (self)
    {
        mDevice = view.device;
        mInFlightSemaphore = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);
        [self loadMetalWithView:view];
        
        mCamera = [[Camera alloc] initWithDevice:mDevice];
        mCurrentFrame = 0;
        
        mPrevTimeStamp = allocTimeStamp();
        mCurrentTimeStamp = allocTimeStamp();
        
        getCurrentTime(mPrevTimeStamp);
    }

    return self;
}

/// Load Metal state objects and initialize renderer dependent view properties
- (void)loadMetalWithView:(nonnull MTKView *)view;
{
    // Configures color format for the view.
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    
    mViewportSize.x = view.drawableSize.width;
    mViewportSize.y = view.drawableSize.height;
    
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
        textureDescriptor.textureType = MTLTextureType2D;
        textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
        textureDescriptor.width = mViewportSize.y;
        textureDescriptor.height = mViewportSize.x;
        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        mFrames[i].croppedCameraOutput = [mDevice newTextureWithDescriptor:textureDescriptor];
    }

    
    // Boilerplate.
    id<MTLLibrary> defaultLibrary = [mDevice newDefaultLibrary];
    
    // Create handle to shader-contained compute kernel.
    id<MTLFunction> renderFunctionVertex = [defaultLibrary newFunctionWithName:@"renderVertex"];
    id<MTLFunction> renderFunctionFragment = [defaultLibrary newFunctionWithName:@"renderFragment"];
    
    NSError *error = NULL;
    
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"Render Camera Output";
    pipelineStateDescriptor.vertexFunction = renderFunctionVertex;
    pipelineStateDescriptor.fragmentFunction = renderFunctionFragment;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    
    mRenderOutputPipeline = [mDevice newRenderPipelineStateWithDescriptor:pipelineStateDescriptor
                                                                    error:&error];
    
    // !!
    id<MTLFunction> cropFunction = [defaultLibrary newFunctionWithName:@"cropKernel"];
    mCropPipeline = [mDevice newComputePipelineStateWithFunction:cropFunction error: &error];
    
    
    // Initialize the command queue.
    mCommandQueue = [mDevice newCommandQueue];
}

- (void)renderFinalOutput:(nonnull MTKView *)view commandBuffer:(id<MTLCommandBuffer>)cmdbuf
{
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil)
    {
        // Create the encoder for the render pass.
        id<MTLRenderCommandEncoder> renderEncoder =
        [cmdbuf renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"MyRenderEncoder";

        // Set the region of the drawable to draw into.
        [renderEncoder setViewport:(MTLViewport){0.0, 0.0, mViewportSize.x, mViewportSize.y, 0.0, 1.0 }];

        [renderEncoder setRenderPipelineState:mRenderOutputPipeline];

        [renderEncoder setFragmentTexture:mFrames[mCurrentFrame].croppedCameraOutput
                                  atIndex:0];

        // Draw the quad.
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                          vertexStart:0
                          vertexCount:6];

        [renderEncoder endEncoding];
    }
}

/// Per frame updates here
- (void)drawInMTKView:(nonnull MTKView *)view
{
    // Wait for a free slot in mFrames.
    dispatch_semaphore_wait(mInFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    // ...
    id<MTLCommandBuffer> commandBuffer = [mCommandQueue commandBuffer];
    commandBuffer.label = @"FrameCommandBuffer";
    
    // Tell metal to call a certain function when the command buffer finishes.
    __block dispatch_semaphore_t blockSema = mInFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(blockSema);
     }];
    
    // Dequeue image from camera.
    id<MTLTexture> newImg = [mCamera dequeueTexture];
    if (newImg != nil)
    {
        mFrames[mCurrentFrame].cameraOutput = newImg;
        
        struct {
            vector_uint2 croppedSourceOffset;
            vector_uint2 croppedSourceExtent;
        } cropInfo;
        
        id<MTLTexture> inputTexture = mFrames[mCurrentFrame].cameraOutput;
        id<MTLTexture> outputTexture = mFrames[mCurrentFrame].croppedCameraOutput;
        
        float aspectRatio = (float)outputTexture.height / (float)outputTexture.width;
        cropInfo.croppedSourceExtent.x = (int)inputTexture.width;
        cropInfo.croppedSourceExtent.y = (int)((float)inputTexture.width * aspectRatio);
        
        int inputTextureHeight = (int)inputTexture.height;
        
        cropInfo.croppedSourceOffset.x = 0;
        cropInfo.croppedSourceOffset.y = (int)(inputTextureHeight - cropInfo.croppedSourceExtent.y) / 2;
        
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        
        MTLSize numThreadGroups;
        numThreadGroups.width  = (outputTexture.width  + threadGroupSize.width -  1) / threadGroupSize.width;
        numThreadGroups.height = (outputTexture.height + threadGroupSize.height - 1) / threadGroupSize.height;
        numThreadGroups.depth = 1;
        
        id<MTLComputeCommandEncoder> cropEncoder = [commandBuffer computeCommandEncoder];
        [cropEncoder setComputePipelineState:mCropPipeline];
        [cropEncoder setTexture:inputTexture atIndex:0];
        [cropEncoder setTexture:outputTexture atIndex:1];
        [cropEncoder setBytes:&cropInfo length:sizeof(cropInfo) atIndex:2];
        [cropEncoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadGroupSize];
        [cropEncoder endEncoding];
    }
    
    [self renderFinalOutput:view commandBuffer:commandBuffer];

    [commandBuffer presentDrawable:view.currentDrawable];

    [commandBuffer commit];
    
    // We are guaranteed that compute passes finish one after the other.
    mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/// Respond to drawable size or orientation changes here
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    mViewportSize.x = size.width;
    mViewportSize.y = size.height;
    
    printf("%d %d\n", mViewportSize.x, mViewportSize.y);
    
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
        textureDescriptor.textureType = MTLTextureType2D;
        textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
        textureDescriptor.width = mViewportSize.y;
        textureDescriptor.height = mViewportSize.x;
        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        mFrames[i].croppedCameraOutput = [mDevice newTextureWithDescriptor:textureDescriptor];
    }
}

@end
