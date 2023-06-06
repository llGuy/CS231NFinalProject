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
#import <MetalKit/MetalKit.h>

#include <vector>
#include <algorithm>
#include <numeric>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

float decode(uint16_t float16_value)
{
    // MSB -> LSB
    // float16=1bit: sign, 5bit: exponent, 10bit: fraction
    // float32=1bit: sign, 8bit: exponent, 23bit: fraction
    // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
    // for denormalized exponent(0): value=2**-14*(0.fraction)
    uint32_t sign = float16_value >> 15;
    uint32_t exponent = (float16_value >> 10) & 0x1F;
    uint32_t fraction = (float16_value & 0x3FF);
    uint32_t float32_value;
    if (exponent == 0)
    {
        if (fraction == 0)
        {
            // zero
            float32_value = (sign << 31);
        }
        else
        {
            // can be represented as ordinary value in float32
            // 2 ** -14 * 0.0101
            // => 2 ** -16 * 1.0100
            // int int_exponent = -14;
            exponent = 127 - 14;
            while ((fraction & (1 << 10)) == 0)
            {
                //int_exponent--;
                exponent--;
                fraction <<= 1;
            }
            fraction &= 0x3FF;
            // int_exponent += 127;
            float32_value = (sign << 31) | (exponent << 23) | (fraction << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        /* Inf or NaN */
        float32_value = (sign << 31) | (0xFF << 23) | (fraction << 13);
    }
    else
    {
        /* ordinary number */
        float32_value = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
    }
    
    return *((float*)&float32_value);
}

static inline float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline int floatOffset(int channel, int x, int y, int gridHeight, int gridWidth)
{
    int slice = channel / 4;
    int indexInSlice = channel - slice*4;
    int offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice;
    
    return offset;
}

static inline int maxOfFloatArray(float *inArr, int floatCount)
{
    int max = 0;
    for (int i = 0; i < floatCount; ++i)
        if (inArr[i] > inArr[max])
            max = i;
    
    return max;
}

static inline void softmax(float *inArr, int floatCount)
{
    // Compute softmax.
    int max = maxOfFloatArray(inArr, floatCount);
    
    float sum = 1e-4;
    for (int i = 0; i < floatCount; ++i)
        sum += exp(inArr[i] - inArr[max]);
    
    float maxFloatValue = inArr[max];
    
    for (int i = 0; i < floatCount; ++i)
        inArr[i] = exp(inArr[i] - maxFloatValue) / sum;
}

static inline float predictionIOU(const Prediction &a, const Prediction &b)
{
    float areaA = a.extent.x * a.extent.y;
    if (areaA <= 0)
        return 0.0;
    
    float areaB = b.extent.x * b.extent.y;
    if (areaB <= 0)
        return 0.0;
    
    float intersectionMinX = MAX(a.offset.x, b.offset.x);
    float intersectionMinY = MAX(a.offset.y, b.offset.y);
    float intersectionMaxX = MIN(a.offset.x + a.extent.x, b.offset.x + b.extent.x);
    float intersectionMaxY = MIN(a.offset.y + a.extent.y, b.offset.y + b.extent.y);
    float intersectionArea = MAX(intersectionMaxY - intersectionMinY, 0) * MAX(intersectionMaxX - intersectionMinX, 0);
    
    return intersectionArea / (areaA + areaB - intersectionArea);
}

std::vector<Prediction> nonMaxSuppression(const std::vector<Prediction> &predictions, int limit, float threshold)
{
    // Do an argsort on the confidence scores, from high to low.
    std::vector<size_t> sortedIndices;
    sortedIndices.resize(predictions.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&](size_t i, size_t j) {
        return predictions[i].score > predictions[j].score;
    });
    
    std::vector<Prediction> selected = {};
    
    std::vector<bool> active(predictions.size(), true);
    int numActive = (int)active.size();
    
    for (size_t i = 0; i < predictions.size(); i++) {
        if (active[i]) {
            const auto boxA = predictions[sortedIndices[i]];
            selected.push_back(boxA);
            if (selected.size() >= limit) {
                break;
            }
            
            for (size_t j = i + 1; j < predictions.size(); j++) {
                if (active[j]) {
                    const auto boxB = predictions[sortedIndices[j]];
                    if (predictionIOU(boxA, boxB) > threshold) {
                        active[j] = false;
                        numActive--;
                        if (numActive <= 0) {
                            goto end;
                        }
                    }
                }
            }
        }
    }
    
end:
    return selected;
}

@implementation YOLONet
{
    MPSNNGraph *mNetGraph;
    
    // Input to the network
    int mInputWidth;
    int mInputHeight;
    
    float mBlockSize;
    int mGridHeight;
    int mGridWidth;
    int mBoxesPerCell;
    int mNumClasses;
    
    float mAnchors[10];
    const char *mLabels[20];
    float mKeepThreshold;
    
    id<MTLTexture> mDebugTexture;
}

- (void)createGraph:(id<MTLDevice>)device
{
    NSError *error = nil;
    
    // Load the texture file
    MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice: device];
    
    NSString* path = [[NSBundle mainBundle] pathForResource:@"dog416" ofType:@"png" inDirectory:@""];
    NSURL *url = [NSURL fileURLWithPath:path];
    mDebugTexture = [loader newTextureWithContentsOfURL:url options:nil error:&error];
    
    if (error)
        NSLog(@"%@", error.description);
    
    mBlockSize = 32;
    mGridHeight = 13;
    mGridWidth = 13;
    mBoxesPerCell = 5;
    mNumClasses = 20;
    mKeepThreshold = 0.5;
    
    float anchors[] = {1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f};
    memcpy(mAnchors, anchors, sizeof(anchors));
    
    const char *labels[] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
    memcpy(mLabels, labels, sizeof(labels));
    
    mInputWidth = 416;
    mInputHeight = 416;
    
    // Some preprocessing.
    MPSNNImageNode *inputImageNode = [[MPSNNImageNode alloc] initWithHandle:nil];
    MPSNNLanczosScaleNode *scaleNode = [[MPSNNLanczosScaleNode alloc] initWithSource:inputImageNode outputSize:MTLSizeMake(mInputWidth, mInputHeight, 3)];
    
    // Convolutions.
    MPSCNNConvolutionNode *convNode1 = [[MPSCNNConvolutionNode alloc] initWithSource:scaleNode.resultImage weights:[[NetParams alloc] init:@"conv1" kernelSize:3 inputFeatureChannels:3 outputFeatureChannels:16]];
    convNode1.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode1 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode1.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode2 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode1.resultImage weights:[[NetParams alloc] init:@"conv2" kernelSize:3 inputFeatureChannels:16 outputFeatureChannels:32]];
    convNode2.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode2 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode2.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode3 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode2.resultImage weights:[[NetParams alloc] init:@"conv3" kernelSize:3 inputFeatureChannels:32 outputFeatureChannels:64]];
    convNode3.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode3 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode3.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode4 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode3.resultImage weights:[[NetParams alloc] init:@"conv4" kernelSize:3 inputFeatureChannels:64 outputFeatureChannels:128]];
    convNode4.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode4 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode4.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode5 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode4.resultImage weights:[[NetParams alloc] init:@"conv5" kernelSize:3 inputFeatureChannels:128 outputFeatureChannels:256]];
    convNode5.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode5 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode5.resultImage filterSize:2];
    
    MPSCNNConvolutionNode *convNode6 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode5.resultImage weights:[[NetParams alloc] init:@"conv6" kernelSize:3 inputFeatureChannels:256 outputFeatureChannels:512]];
    convNode6.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    MPSCNNPoolingMaxNode *poolNode6 = [[MPSCNNPoolingMaxNode alloc] initWithSource:convNode6.resultImage filterSize:2 stride:1];
    poolNode6.paddingPolicy = [[Pool6PaddingPolicy alloc] init];
    
    MPSCNNConvolutionNode *convNode7 = [[MPSCNNConvolutionNode alloc] initWithSource:poolNode6.resultImage weights:[[NetParams alloc] init:@"conv7" kernelSize:3 inputFeatureChannels:512 outputFeatureChannels:1024]];
    convNode7.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    
    MPSCNNConvolutionNode *convNode8 = [[MPSCNNConvolutionNode alloc] initWithSource:convNode7.resultImage weights:[[NetParams alloc] init:@"conv8" kernelSize:3 inputFeatureChannels:1024 outputFeatureChannels:1024]];
    convNode8.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    
    MPSCNNConvolutionNode *convNode9 = [[MPSCNNConvolutionNode alloc] initWithSource:convNode8.resultImage weights:[[NetParams alloc] initNoLeaky:@"conv9" kernelSize:1 inputFeatureChannels:1024 outputFeatureChannels:125]];
    convNode9.accumulatorPrecision = MPSNNConvolutionAccumulatorPrecisionOptionHalf;
    
    NSArray *result = @[convNode9.resultImage];
    BOOL needed = true;
    mNetGraph = [[MPSNNGraph alloc] initWithDevice:device resultImages:result resultsAreNeeded:&needed];
    
    NSLog(@"%@", mNetGraph.debugDescription);
}

-(void)makeBoundingBoxes:(nonnull MPSImage *)inputImage predictions:(struct Prediction *)dst predictionCount:(int *)count
{
    void *buffer = malloc(sizeof(uint16_t) * 13 * 13 * 128);
    [inputImage readBytes:buffer dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth imageIndex:0];
    
    void *buffer0 = malloc(sizeof(uint16_t) * 13 * 13 * 4);
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, 13, 13, 1);
    [inputImage.texture getBytes:buffer0 bytesPerRow:13*4*2 bytesPerImage:0 fromRegion:region mipmapLevel:0 slice:0];
    
    uint16_t *features0 = (uint16_t *)buffer0;
    
    float16_t *features = (float16_t *) buffer;
    float tx, ty, tw, th, tc, x, y, w, h, confidence;
   
    std::vector<Prediction> predictions;
    
    // printf("%f vs %d\n", (float)features[0], (int)features0[0]);
    
    for (int cy = 0; cy < mGridHeight; cy++) {
        for (int cx = 0; cx < mGridWidth; cx++) {
            for (int b = 0; b < mBoxesPerCell; b++){
                
                // Operate on bounding boxes, determine which to keep.
                int channel = b * (mNumClasses + 5);
                tx = (float)(features[floatOffset(channel, cx, cy, mGridHeight, mGridWidth)]);
                ty = (float)(features[floatOffset(channel + 1, cx, cy, mGridHeight, mGridWidth)]);
                tw = (float)(features[floatOffset(channel + 2, cx, cy, mGridHeight, mGridWidth)]);
                th = (float)(features[floatOffset(channel + 3, cx, cy, mGridHeight, mGridWidth)]);
                tc = (float)(features[floatOffset(channel + 4, cx, cy, mGridHeight, mGridWidth)]);
                
                // Location of bounding box in input image.
                x = ((float) cx + sigmoid(tx)) * mBlockSize;
                y = ((float) cy + sigmoid(ty)) * mBlockSize;

                //
                w = exp(tw) * mAnchors[2 * b] * mBlockSize;
                h = exp(th) * mAnchors[2 * b + 1] * mBlockSize;

                confidence = sigmoid(tc);
                
                int numClasses = sizeof(mLabels) / sizeof(mLabels[0]);
                float classes[numClasses];
                
                bool allZero = true;
                
                for (int i = 0; i < numClasses; ++i)
                {
                    classes[i] = (float)(features[floatOffset(channel + 5 + i, cx, cy, mGridHeight, mGridWidth)]);
                    
                    if (classes[i] != 0.0f)
                        allZero = false;
                }
                
                if (allZero)
                    continue;
                
#if 0
                float copy[numClasses];
                memcpy(copy, classes, sizeof(classes));
                
                if (cx == 0 && cy == 0 && b == 0)
                {
                    for (int i= 0 ; i < numClasses; ++i)
                        printf("%f\n", copy[i]);
                }
#endif
                
                softmax(classes, numClasses);
                
                int max = maxOfFloatArray(classes, numClasses);
                
                // Argmax.
                int detectedClass = max;
                float detectedClassScore = classes[max];
                
                float confidenceInClass = detectedClassScore * confidence;
                
                if (confidenceInClass > mKeepThreshold)
                {
#if 0
                    if (x-w/2.0f < 0.0f)
                    {
                        printf("Bug\n");
                    }
#endif
                    
                    Prediction prediction = {
                        detectedClass,
                        simd_make_int2((int)x - (int)w/2, (int)y - (int)h/2),
                        simd_make_int2((int)w, (int)h),
                        confidenceInClass
                    };
                    
                    predictions.push_back(prediction);
                }
            }
        }
    }
    
    auto selectedPredictions = nonMaxSuppression(predictions, 10, 0.5);
    
    memcpy(dst, selectedPredictions.data(), sizeof(struct Prediction) * selectedPredictions.size());
    *count = (int)selectedPredictions.size();
    
#if 0
    printf("%d\n", (int)selectedPredictions.size());
    
    for (int i = 0; i < selectedPredictions.size(); ++i)
    {
        printf("%s: %f\n", mLabels[selectedPredictions[i].classIndex], selectedPredictions[i].score);
    }
    
    printf("\n");
#endif
    
    free(buffer);
    free(buffer0);
}

-(nonnull MPSImage *)encodeGraph:(nonnull id<MTLTexture>)inputTexture commandBuffer:(id<MTLCommandBuffer>)cmdbuf
{
    // MPSImage *mpsImage = [[MPSImage alloc] initWithTexture:inputTexture featureChannels:3];
    MPSImage *mpsImage = [[MPSImage alloc] initWithTexture:mDebugTexture featureChannels:3];
    
    NSArray *inputImages = @[ mpsImage ];
    
    MPSImage *result = [mNetGraph encodeToCommandBuffer:cmdbuf sourceImages:inputImages];
    
    return result;
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

-(nonnull const char *)getLabel:(int)classIndex
{
    return mLabels[classIndex];
}

@end
