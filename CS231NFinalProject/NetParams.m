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
    NSString *mPath;
    float *mData;
    
    NSString *mName;
    
    int mKernelSize;
    int mInChannels;
    int mOutChannels;
}

- (nonnull instancetype)init:(nonnull NSString *)name kernelSize:(int)size inputFeatureChannels:(int)inChannels outputFeatureChannels:(int)outChannels;
{
    mName = name;
    mKernelSize = size;
    mInChannels = inChannels;
    mOutChannels = outChannels;
}


- (nullable float *)biasTerms;
- (MPSDataType)dataType;
- (nonnull MPSCNNConvolutionDescriptor *)descriptor; {}
- (nullable NSString *)label; {}

- (BOOL)load;
{
    mPath = [[NSBundle mainBundle] pathForResource:@"conv1" ofType:@"bin" inDirectory:@""];
    mData = [NSData dataWithContentsOfFile:mPath options:NSDataReadingUncached error:NULL];
    
    return YES;
}

- (nonnull float *)lookupTableForUInt8Kernel; {}
- (void)purge; {}
- (nonnull vector_float2 *)rangesForUInt8Kernel; {}
- (nonnull void *)weights;


@end
