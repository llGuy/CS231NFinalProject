//
//  PaddingPolicy.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef PaddingPolicy_h
#define PaddingPolicy_h

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface Pool6PaddingPolicy : NSObject <MPSNNPadding>

- (nonnull instancetype)init;
- (MPSNNPaddingMethod)paddingMethod;
- (MPSImageDescriptor *)destinationImageDescriptorForSourceImages:(NSArray<MPSImage *> *)sourceImages
                                                     sourceStates:(NSArray<MPSState *> *)sourceStates
                                                        forKernel:(MPSKernel *)kernel
                                              suggestedDescriptor:(MPSImageDescriptor *)inDescriptor;

@end

#endif /* PaddingPolicy_h */
