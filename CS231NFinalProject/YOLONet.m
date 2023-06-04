//
//  YOLONet.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#import <Foundation/Foundation.h>

#import "YOLONet.h"
#import "NetParams.h"
#import "PaddingPolicy.h"

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
    mInputWidth = 416;
    mInputHeight = 416;
    
    // Some preprocessing.
    MPSNNImageNode *inputImageNode = [[MPSNNImageNode alloc] initWithHandle:nil];
    MPSNNLanczosScaleNode *scaleNode = [[MPSNNLanczosScaleNode alloc] initWithSource:inputImageNode outputSize:MTLSizeMake(mInputWidth, mInputHeight, 1)];
    
    // Convolutions.
    MPSCNNConvolutionNode *convNode1 = [[MPSCNNConvolutionNode alloc] initWithSource:scaleNode.resultImage weights:[[NetParams alloc] init:@"conv1" kernelSize:3 inputFeatureChannels:3 outputFeatureChannels:16]];
    MPSCNNPoolingMaxNode *poolNode1 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode1.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode2 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode1.resultImage weights:[[NetParams alloc] init:@"conv2" kernelSize:3 inputFeatureChannels:16 outputFeatureChannels:32]];
    MPSCNNPoolingMaxNode *poolNode2 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode2.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode3 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode2.resultImage weights:[[NetParams alloc] init:@"conv3" kernelSize:3 inputFeatureChannels:32 outputFeatureChannels:64]];
    MPSCNNPoolingMaxNode *poolNode3 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode3.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode4 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode3.resultImage weights:[[NetParams alloc] init:@"conv4" kernelSize:3 inputFeatureChannels:64 outputFeatureChannels:128]];
    MPSCNNPoolingMaxNode *poolNode4 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode4.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode5 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode4.resultImage weights:[[NetParams alloc] init:@"conv5" kernelSize:3 inputFeatureChannels:128 outputFeatureChannels:256]];
    MPSCNNPoolingMaxNode *poolNode5 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode5.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode6 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode5.resultImage weights:[[NetParams alloc] init:@"conv6" kernelSize:3 inputFeatureChannels:256 outputFeatureChannels:512]];
    MPSCNNPoolingMaxNode *poolNode6 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode6.resultImage filterSize:2 stride:1];
    poolNode6.paddingPolicy = [[Pool6PaddingPolicy alloc] init];
    
    MPSCNNConvolutionNode *convNode7 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode6.resultImage weights:[[NetParams alloc] init:@"conv7" kernelSize:3 inputFeatureChannels:512 outputFeatureChannels:1024]];
    
    MPSCNNConvolutionNode *convNode8 = [[MPSCNNConvolutionNode alloc] initWithSource:convNode7.resultImage weights:[[NetParams alloc] init:@"conv8" kernelSize:3 inputFeatureChannels:1024 outputFeatureChannels:1024]];
    
    MPSCNNConvolutionNode *convNode9 = [[MPSCNNConvolutionNode alloc] initWithSource:convNode8.resultImage weights:[[NetParams alloc] init:@"conv9" kernelSize:1 inputFeatureChannels:1024 outputFeatureChannels:125]];
    
    NSArray *result = @[convNode9.resultImage];
    BOOL needed = true;
    mNetGraph = [[MPSNNGraph alloc] initWithDevice:device resultImages:result resultsAreNeeded:&needed];
    
    NSLog(@"%@", mNetGraph.debugDescription);
}

-(void)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(id<MTLCommandBuffer>)cmdbuf
{
    MPSImage *mpsImage = [[MPSImage alloc] initWithTexture:inputTexture featureChannels:3];
    
    NSArray *inputImages = @[ mpsImage ];
    [mNetGraph encodeToCommandBuffer:cmdbuf sourceImages:inputImages];
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
