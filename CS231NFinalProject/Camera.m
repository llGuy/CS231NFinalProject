//
//  Camera.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#import <AVKit/AVKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#import "Def.h"
#import "Camera.h"
#import "NSMutableArray+QueueAdditions.h"

@implementation Camera
{
    CVMetalTextureCacheRef mTextureCache;
    AVCaptureDevice *mVideoDevice;
    AVCaptureSession *mCaptureSession;
    NSMutableArray *mTextures;
    NSLock *mQueueLock;
    dispatch_queue_t mDispatchQueue;
}

- (nonnull instancetype)initWithDevice:(id<MTLDevice>)device;
{
    self = [super init];
    
    mTextures = [[NSMutableArray alloc] init];
    mQueueLock = [[NSLock alloc] init];
    
    /* Setup some preliminary objects */
    CVMetalTextureCacheCreate(NULL, NULL, device, NULL, &mTextureCache);
    
    /* Make the camera session and get device (back camera). */
    @try
    {
        NSError *error;
        
        mCaptureSession = [[AVCaptureSession alloc] init];
        [mCaptureSession setSessionPreset:AVCaptureSessionPreset640x480];
        
        mVideoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        if (mVideoDevice == nil)
        {
            exit(-1);
            return self;
        }
        
        AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:mVideoDevice error:&error];
        [mCaptureSession addInput:input];
        
        for(AVCaptureDeviceFormat *format in [mVideoDevice formats])
        {
            // Check for 60 FPS
            CMFormatDescriptionRef description = format.formatDescription;
            float maxFrameRate = ((AVFrameRateRange*)[format.videoSupportedFrameRateRanges objectAtIndex:0]).maxFrameRate;
            
            BOOL containsColorSpace = false;
            for (NSNumber *colorSpace in format.supportedColorSpaces)
            {
                if (colorSpace.intValue == AVCaptureColorSpace_sRGB)
                {
                    containsColorSpace = true;
                    break;
                }
            }
            
            if (!containsColorSpace)
                continue;
            
            if (maxFrameRate < 60.0f)
                continue;
            
            CMVideoDimensions dim = CMVideoFormatDescriptionGetDimensions(description);
            if (dim.width < 2000 && dim.height < 1500)
                continue;
            
            // We can proceed to set to this format.
            if ([mVideoDevice lockForConfiguration:NULL] == YES)
            {
                NSLog(@"formats  %@ %@ %@", format.mediaType, format.formatDescription, format.videoSupportedFrameRateRanges);
                
                mVideoDevice.activeFormat = format;
                [mVideoDevice setActiveVideoMinFrameDuration:CMTimeMake(1,60)];
                [mVideoDevice setActiveVideoMaxFrameDuration:CMTimeMake(1,60)];
                [mVideoDevice setExposureMode:AVCaptureExposureModeAutoExpose];
                
                [mVideoDevice unlockForConfiguration];
                break;
            }
        }
        
        mDispatchQueue = dispatch_queue_create("CameraSessionQueue", DISPATCH_QUEUE_SERIAL);
        
        AVCaptureVideoDataOutput *dataOutput = [[AVCaptureVideoDataOutput alloc] init];
        [dataOutput setAlwaysDiscardsLateVideoFrames:YES];
        [dataOutput setVideoSettings:@{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}];
        [dataOutput setSampleBufferDelegate:self queue:mDispatchQueue];
        
        [mCaptureSession addOutput:dataOutput];
        [mCaptureSession commitConfiguration];
    }
    @catch (NSException *exception)
    {
        NSLog(@"%s - %@", __PRETTY_FUNCTION__, exception.description);
        exit(-1);
    }
    @finally
    {
    }
    
    [mCaptureSession startRunning];
    
    return self;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection;
{
    /* Make sure that we don't overload the texture queue in case the renderer cannot catch up. */
    [mQueueLock lock];
    while ([mTextures count] > MAX_FRAMES_IN_FLIGHT-1)
    {
        [mTextures dequeue];
    }
    [mQueueLock unlock];
    
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    
    CVMetalTextureRef texture = NULL;
    
    CVReturn status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, mTextureCache, pixelBuffer, NULL, MTLPixelFormatBGRA8Unorm, width, height, 0, &texture);
    
    if (status == kCVReturnSuccess)
    {
        id<MTLTexture> newTexture = CVMetalTextureGetTexture(texture);
        
        [mQueueLock lock];
        [mTextures enqueue:newTexture];
        [mQueueLock unlock];
        
        // NSLog(@"Got picture from camera!");
        
        CFRelease(texture);
    }
}

- (id<MTLTexture>)dequeueTexture;
{
    id<MTLTexture> ret = nil;
    
    [mQueueLock lock];
    if ([mTextures count] > 0)
        ret = [mTextures dequeue];
    [mQueueLock unlock];
    
    return ret;
}

@end
