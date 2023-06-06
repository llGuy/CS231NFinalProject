//
//  NetParams.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#include <fcntl.h>
#include <unistd.h>

#include "NetParams.h"

@implementation NetParams
{
    NSData *mData;

    NSString *mName;
    
    int mKernelSize;
    int mInChannels;
    int mOutChannels;
    
    BOOL mLeaky;
}

- (nonnull instancetype)init:(nonnull NSString *)name kernelSize:(int)size inputFeatureChannels:(int)inChannels outputFeatureChannels:(int)outChannels;
{
    self = [super init];
    
    if (self)
    {
        mName = name;
        mKernelSize = size;
        mInChannels = inChannels;
        mOutChannels = outChannels;
        
        mLeaky = true;
    }
    
    return self;
}


- (nonnull instancetype)initNoLeaky:(nonnull NSString *)name kernelSize:(int)size inputFeatureChannels:(int)inChannels outputFeatureChannels:(int)outChannels;
{
    self = [super init];
    
    if (self)
    {
        mName = name;
        mKernelSize = size;
        mInChannels = inChannels;
        mOutChannels = outChannels;
        
        mLeaky = false;
    }
    
    return self;
}


- (nullable float *)biasTerms;
{
    return NULL;
}

- (MPSDataType)dataType
{
    return MPSDataTypeFloat32;
}

- (nonnull MPSCNNConvolutionDescriptor *)descriptor
{
    MPSCNNConvolutionDescriptor *desc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:mKernelSize kernelHeight:mKernelSize inputFeatureChannels:mInChannels outputFeatureChannels:mOutChannels];
    
    if (mLeaky)
    {
        // Set activation to leaky relu
        MPSNNNeuronDescriptor *neuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.1 b:0];
        [desc setFusedNeuronDescriptor:neuronDescriptor];
        
        // Set the batch normalization parameters
        float *weightsPtr = (float *)mData.bytes;
        float *meanPtr = weightsPtr + (mInChannels * mKernelSize * mKernelSize * mOutChannels);
        float *variancePtr = meanPtr + mOutChannels;
        float *gammaPtr = variancePtr + mOutChannels;
        float *betaPtr = gammaPtr + mOutChannels;
        [desc setBatchNormalizationParametersForInferenceWithMean:meanPtr variance:variancePtr gamma:gammaPtr beta:betaPtr epsilon:1e-3];
    }
    else
    {
        MPSNNNeuronDescriptor *neuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone a:0 b:0];
        [desc setFusedNeuronDescriptor:neuronDescriptor];
    }
    
    return desc;
}

- (nullable NSString *)label
{
    return mName;
}

- (BOOL)load;
{
    NSString* path = [[NSBundle mainBundle] pathForResource:mName ofType:@"bin" inDirectory:@""];
    mData = [NSData dataWithContentsOfFile:path options:NSDataReadingUncached error:NULL];
    
    if (mData.length > 0)
        return YES;
    else
        return NO;
}

- (void)purge;
{
    mData = nil;
}

- (nonnull void *)weights
{
    printf("%f\n", *(float *)mData.bytes);
    
    return (void *)mData.bytes;
}

- (nonnull id)copyWithZone:(nullable NSZone *)zone
{
    NSLog(@"Copy not defined");
    exit(-1);
}

@end
