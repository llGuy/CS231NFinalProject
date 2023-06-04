//
//  NetParams.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef NetParams_h
#define NetParams_h

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface NetParams : NSObject <MPSCNNConvolutionDataSource>

- (nonnull instancetype)init:(nonnull NSString *)name kernelSize:(int)size inputFeatureChannels:(int)inChannels outputFeatureChannels:(int)outChannels;
- (nullable float *)biasTerms;
- (MPSDataType)dataType;
- (nonnull MPSCNNConvolutionDescriptor *)descriptor;
- (nullable NSString *)label;
- (BOOL)load;
- (nonnull float *)lookupTableForUInt8Kernel;
- (void)purge;
- (nonnull vector_float2 *)rangesForUInt8Kernel;
- (nonnull void *)weights;
- (nonnull id)copyWithZone:(nullable NSZone *)zone;

@end

#endif /* NetParams_h */
