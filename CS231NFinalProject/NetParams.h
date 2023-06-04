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

- (float *)biasTerms;
- (MPSDataType)dataType;
- (MPSCNNConvolutionDescriptor *)descriptor;
- (NSString *)label;
- (BOOL)load;
- (float *)lookupTableForUInt8Kernel;
- (void)purge;
- (vector_float2 *)rangesForUInt8Kernel;
- (void *)weights;

@end

#endif /* NetParams_h */
