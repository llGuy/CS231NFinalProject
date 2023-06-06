//
//  Camera.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#ifndef Camera_h
#define Camera_h

#import <Metal/Metal.h>
#import <AVKit/AVKit.h>

struct CameraImage
{
    id<MTLTexture> image;
    CFTimeInterval timeStamp;
};

@interface Camera : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>

-(nonnull instancetype)initWithDevice:(nonnull id<MTLDevice>)device;
-(nullable id<MTLTexture>)dequeueTexture;

@end

#endif /* Camera_h */
