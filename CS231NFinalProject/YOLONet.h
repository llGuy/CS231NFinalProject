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

@end

#endif /* YOLONet_h */
