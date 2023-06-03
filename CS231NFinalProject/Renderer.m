//
//  Renderer.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/1/23.
//

#import <simd/simd.h>
#import <ModelIO/ModelIO.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
    dispatch_semaphore_t mInFlightSemaphores[MAX_FRAMES_IN_FLIGHT];
    
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
    
    MPSImageGaussianBlur *mGaussianKernel;
    
    // Camera outputs.
    struct FrameData mFrames[MAX_FRAMES_IN_FLIGHT];
    
    // Encapsulates camera capture.
    Camera *mCamera;
    
    struct TimeStamp *mPrevTimeStamp;
    struct TimeStamp *mCurrentTimeStamp;
    float mDt;
    
    vector_uint2 mViewportSize;
    
    // Weirdest bug - if we don't compile this in, app crashes.
    MTKMeshBufferAllocator *mMeshAllocator;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if (self)
    {
        mDevice = view.device;
        
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            mInFlightSemaphores[i] = dispatch_semaphore_create(1);
        
        [self loadMetalWithView:view];
        
        mCamera = [[Camera alloc] initWithDevice:mDevice];
        mCurrentFrame = 0;
        
        mPrevTimeStamp = allocTimeStamp();
        mCurrentTimeStamp = allocTimeStamp();
        
        getCurrentTime(mPrevTimeStamp);
        
        mMeshAllocator = [[MTKMeshBufferAllocator alloc]
                          initWithDevice: mDevice];
    }

    return self;
}

- (void)createImageProcessingPipeline:(id<MTLLibrary>)defaultLibrary;
{
    NSError *error = nil;
    
    // !!
    id<MTLFunction> cropFunction = [defaultLibrary newFunctionWithName:@"cropAndRotateKernel"];
    mCropPipeline = [mDevice newComputePipelineStateWithFunction:cropFunction error: &error];
    
    mGaussianKernel = [[MPSImageGaussianBlur alloc] initWithDevice:mDevice sigma:10.0f];
}

- (void)createCroppedCameraImages:(nonnull MTKView *)view;
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
        textureDescriptor.textureType = MTLTextureType2D;
        textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
        textureDescriptor.width = mViewportSize.x;
        textureDescriptor.height = mViewportSize.y;
        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        mFrames[i].croppedCameraOutput = [mDevice newTextureWithDescriptor:textureDescriptor];
    }
}

/// Load Metal state objects and initialize renderer dependent view properties
- (void)loadMetalWithView:(nonnull MTKView *)view;
{
    // Configures color format for the view.
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    view.sampleCount = 1;

    
    mViewportSize.x = view.drawableSize.width;
    mViewportSize.y = view.drawableSize.height;
    
    [self createCroppedCameraImages:view];
    
    // Boilerplate.
    id<MTLLibrary> defaultLibrary = [mDevice newDefaultLibrary];
    
    // Create handle to shader-contained compute kernel.
    id<MTLFunction> renderFunctionVertex = [defaultLibrary newFunctionWithName:@"renderVertex"];
    id<MTLFunction> renderFunctionFragment = [defaultLibrary newFunctionWithName:@"renderFragment"];
    
    NSError *error = NULL;
    
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"Render Camera Output";
    pipelineStateDescriptor.rasterSampleCount = view.sampleCount;
    pipelineStateDescriptor.vertexFunction = renderFunctionVertex;
    pipelineStateDescriptor.fragmentFunction = renderFunctionFragment;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    pipelineStateDescriptor.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;

    
    mRenderOutputPipeline = [mDevice newRenderPipelineStateWithDescriptor:pipelineStateDescriptor
                                                                    error:&error];
    
    [self createImageProcessingPipeline:defaultLibrary];
    
    // Initialize the command queue.
    mCommandQueue = [mDevice newCommandQueue];
}

- (void)encodeFinalRender:(nonnull MTKView *)view commandBuffer:(id<MTLCommandBuffer>)cmdbuf
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

- (void)encodeCropAndRotate:(id<MTLCommandBuffer>)cmdbuf
{
    struct {
        vector_uint2 croppedSourceOffset;
        vector_uint2 croppedSourceExtent;
    } cropInfo;
    
    id<MTLTexture> inputTexture = mFrames[mCurrentFrame].cameraOutput;
    id<MTLTexture> outputTexture = mFrames[mCurrentFrame].croppedCameraOutput;
    
    float aspectRatio = (float)outputTexture.width / (float)outputTexture.height;
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
    
    id<MTLComputeCommandEncoder> cropEncoder = [cmdbuf computeCommandEncoder];
    [cropEncoder setComputePipelineState:mCropPipeline];
    [cropEncoder setTexture:inputTexture atIndex:0];
    [cropEncoder setTexture:outputTexture atIndex:1];
    [cropEncoder setBytes:&cropInfo length:sizeof(cropInfo) atIndex:2];
    [cropEncoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadGroupSize];
    [cropEncoder endEncoding];
}

- (void)encodeImageProcessing:(id<MTLCommandBuffer>)cmdbuf
{
    [self encodeCropAndRotate:cmdbuf];
    
    [mGaussianKernel encodeToCommandBuffer:cmdbuf inPlaceTexture:&mFrames[mCurrentFrame].croppedCameraOutput fallbackCopyAllocator:nil];
}

- (void)calculateFramerate
{
    getCurrentTime(mCurrentTimeStamp);
    mDt = getTimeDifference(mCurrentTimeStamp, mPrevTimeStamp);
    getCurrentTime(mPrevTimeStamp);
}

- (void)printFramerate;
{
    printf("Framerate: %f\n", 1.0f/mDt);
}

/// Per frame updates here
- (void)drawInMTKView:(nonnull MTKView *)view
{
    // Wait for a free slot in mFrames.
    dispatch_semaphore_wait(mInFlightSemaphores[mCurrentFrame], DISPATCH_TIME_FOREVER);
    
    // ...
    id<MTLCommandBuffer> commandBuffer = [mCommandQueue commandBuffer];
    commandBuffer.label = @"FrameCommandBuffer";
    
    // Tell metal to call a certain function when the command buffer finishes.
    __block dispatch_semaphore_t blockSema = mInFlightSemaphores[mCurrentFrame];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(blockSema);
     }];
    
#if 1
    /* Render output from previous submission. */
    [self encodeFinalRender:view commandBuffer:commandBuffer];
    
    // Dequeue image from camera.
    id<MTLTexture> newImg = [mCamera dequeueTexture];
    if (newImg != nil)
    {
        mFrames[mCurrentFrame].cameraOutput = newImg;
        
        // Perform all image processing operations
        [self encodeImageProcessing:commandBuffer];
    }
    else
    {
        /* Did not get new image from camera */
        printf("Did not get new image from camera\n");
    }
#endif
    
    // We are guaranteed that compute passes finish one after the other.
    mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    [commandBuffer presentDrawable:view.currentDrawable];

    [commandBuffer commit];
    
    [self calculateFramerate];
    // [self printFramerate];
}

/// Respond to drawable size or orientation changes here
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    mViewportSize.x = size.width;
    mViewportSize.y = size.height;
    
    printf("%d %d\n", mViewportSize.x, mViewportSize.y);
    
    [self createCroppedCameraImages:view];
}

@end
