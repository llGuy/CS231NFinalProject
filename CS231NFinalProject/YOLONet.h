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

struct Prediction
{
    int classIndex;
    vector_int2 offset;
    vector_int2 extent;
    float score;
};

@interface YOLONet : NSObject

-(nonnull instancetype)initWithDevice:(nonnull id<MTLDevice>)device;
-(nonnull MPSImage *)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(nonnull id<MTLCommandBuffer>)cmdbuf;
-(void)makeBoundingBoxes:(nonnull MPSImage *)inputImage predictions:(struct Prediction *)dst predictionCount:(int *)count;
-(nonnull const char *)getLabel:(int)classIndex;

@end

#endif /* YOLONet_h */
