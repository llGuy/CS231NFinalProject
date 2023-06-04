//
//  YOLONet.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef YOLONet_h
#define YOLONet_h

#import <Metal/Metal.h>

@interface YOLONet : NSObject

-(nonnull instancetype)initWithDevice:(nonnull id<MTLDevice>)device;
-(id<MTLTexture>)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(id<MTLCommandBuffer>)cmdbuf;

@end

#endif /* YOLONet_h */
