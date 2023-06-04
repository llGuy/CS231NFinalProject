//
//  YOLONet.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef YOLONet_h
#define YOLONet_h

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface YOLONet : NSObject

-(instancetype)initWithDevice:(nonnull id<MTLDevice>)device;
-(nonnull MPSImage *)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(nonnull id<MTLCommandBuffer>)cmdbuf;
-(void)makeBoundingBoxes:(nonnull MPSImage *)inputImage;

@end

#endif /* YOLONet_h */
