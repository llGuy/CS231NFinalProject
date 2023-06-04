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
#import "math.h"
#import <Accelerate/Accelerate.h>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@implementation YOLONet
{
    MPSNNGraph *mNetGraph;
    
    // Input to the network
    int mInputWidth;
    int mInputHeight;
}

float blockSize = 32;
int gridHeight = 13;
int gridWidth = 13;
int boxesPerCell = 5;
int numClasses = 20;

float const anchors[] = {1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f};

char const *labels[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

float keepThreshold = 0.3;

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

int float_offset(int channel, int x, int y) {
    
    int slice = channel / 4;
    int indexInSlice = channel - slice*4;
    int offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice;
    return offset;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

-(void)makeBoundingBoxes:(MPSImage *) image
{
    void *buffer = malloc(sizeof(float) * 13 * 13 * 128);
    [image readBytes:buffer dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth imageIndex:0];
    float *features = (float *) buffer;
    float tx, ty, tw, th, tc, x, y, w, h, confidence;
    
    for (int cy = 0; cy < gridHeight; cy++) {
        for (int cx = 0; cx < gridWidth; cx++) {
            for (int b = 0; b < boxesPerCell; b++){
                
                // Operate on bounding boxes, determine which to keep.
                int channel = b * (numClasses + 5);
                tx = features[float_offset(channel, cx, cy)];
                ty = features[float_offset(channel + 1, cx, cy)];
                tw = features[float_offset(channel + 2, cx, cy)];
                th = features[float_offset(channel + 3, cx, cy)];
                tc = features[float_offset(channel + 4, cx, cy)];
                
                // Location of bounding box in input image.
                x = (float) cx + sigmoid(tx) * blockSize;
                y = (float) cy + sigmoid(ty) * blockSize;

                //
                w = exp(tw) * anchors[2 * b] * blockSize;
                h = exp(th) * anchors[2 * b + 1] * blockSize;

                confidence = sigmoid(tc);
                
                int numClasses = sizeof(labels) / sizeof(labels[0]);
                float classes[numClasses];
                for (int i = 0; i < numClasses; ++i)
                {
                    classes[i] = features[float_offset(channel + 5 + i, cx, cy)];
                }
                
                // Compute softmax.
                int max = 0;
                for (int i = 0; i < numClasses; ++i)
                {
                    if (classes[i] > classes[max]) max = i;
                }
                float sum = 1e-4;
                for (int i = 0; i < numClasses; ++i)
                {
                    sum += exp(classes[i] - classes[max]);
                }
                for (int i = 0; i < numClasses; ++i)
                {
                    classes[i] = exp(classes[i] - classes[max]) / sum;
                }
                
                // Argmax.
                int detectedClass = max;
                float detectedClassScore = classes[max];
                
                //
                float confidenceInClass = detectedClassScore * confidence;
                
                //
                if (confidenceInClass >= keepThreshold)
                {
                    printf("found %s\n", labels[detectedClass]);
                }
            }
        }
    }
    
    free(buffer);
}

-(id<MTLTexture>)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(id<MTLCommandBuffer>)cmdbuf
{
    MPSImage *mpsImage = [[MPSImage alloc] initWithTexture:inputTexture featureChannels:3];
    
    NSArray *inputImages = @[ mpsImage ];
    
    MPSImage *result = [mNetGraph encodeToCommandBuffer:cmdbuf sourceImages:inputImages];
    
    printf("%d %d %d %d\n", (int)result.numberOfImages, (int)result.width, (int)result.height, (int)result.featureChannels);
    
    return result.texture;
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
