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

    // Compute pipeline.
    id <MTLComputePipelineState> mRenderOutputPipeline;
    
    // Camera outputs.
    id <MTLTexture> mCameraOutputs[MAX_FRAMES_IN_FLIGHT];
    
    // Encapsulates camera capture.
    Camera *mCamera;
}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if (self)
    {
        mDevice = view.device;
        mInFlightSemaphore = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);
        [self loadMetalWithView:view];
        
        mCamera = [[Camera alloc] initWithDevice:mDevice];
        mCurrentFrame = 0;
    }

    return self;
}

/// Load Metal state objects and initialize renderer dependent view properties
- (void)loadMetalWithView:(nonnull MTKView *)view;
{
    // Configures color format for the view.
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;

    // Boilerplate.
    id<MTLLibrary> defaultLibrary = [mDevice newDefaultLibrary];
    
    // Create handle to shader-contained compute kernel.
    id<MTLFunction> renderFunction = [defaultLibrary newFunctionWithName:@"renderKernel"];
    
    // Initialize compute pipeline with our compute kernel,
    // no error handling.
    NSError *error = NULL;
    mRenderOutputPipeline = [mDevice newComputePipelineStateWithFunction:renderFunction error:&error];
    
    // Initialize the command queue.
    mCommandQueue = [mDevice newCommandQueue];
}

/// Per frame updates here
- (void)drawInMTKView:(nonnull MTKView *)view
{

    // Wait for a free slot in mCameraOutputs.
    dispatch_semaphore_wait(mInFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    // ...
    id<MTLCommandBuffer> commandBuffer = [mCommandQueue commandBuffer];
    commandBuffer.label = @"FrameCommandBuffer";
    
    // Tell metal to call a certain function when the command buffer
    // finishes.
    __block dispatch_semaphore_t blockSema = mInFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(blockSema);
     }];
    
    // Dequeue image from camera.
    id<MTLTexture> newImg = [mCamera dequeueTexture];
    if (newImg != nil)
    {
        mCameraOutputs[mCurrentFrame] = newImg;
    }
    
    // ..
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    id<MTLTexture> targetTexture = view.currentDrawable.texture;
    [computeEncoder setComputePipelineState:mRenderOutputPipeline];
    [computeEncoder setTexture:mCameraOutputs[mCurrentFrame] atIndex:0];
    [computeEncoder setTexture:targetTexture atIndex:1];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroupCount;
    
    threadgroupCount.width  = (targetTexture.width  + threadgroupSize.width -  1) / threadgroupSize.width;
    
    threadgroupCount.height = (targetTexture.height + threadgroupSize.height - 1) / threadgroupSize.height;
    
    // The image data is 2D, so set depth to 1.
    threadgroupCount.depth = 1;

    [computeEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    
    [commandBuffer presentDrawable:view.currentDrawable];

    [commandBuffer commit];
    
    // We are guaranteed that compute passes finish one after the other.
    mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/// Respond to drawable size or orientation changes here
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
}

@end
