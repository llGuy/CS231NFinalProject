//
//  YOLONet.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#import <Foundation/Foundation.h>

#import "YOLONet.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@implementation YOLONet
{
    MPSNNGraph *mNetGraph;
    
    // Input to the network
    int mInputWidth;
    int mInputHeight;
}

- (void)createGraph:(id<MTLDevice>)device
{
    MPSNNImageNode *inputImageNode = [[MPSNNImageNode alloc] initWithHandle:nil];
    MPSNNLanczosScaleNode *scaleNode = [[MPSNNLanczosScaleNode alloc] initWithSource:inputImageNode outputSize:MTLSizeMake(mInputWidth, mInputHeight, 1)];
    
}

- (nonnull instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    
    if (self)
    {
        [self createGraph:device];
    }
    
    return self;
}

@end
