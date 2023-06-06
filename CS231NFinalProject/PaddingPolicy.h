//
//  PaddingPolicy.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#ifndef PaddingPolicy_h
#define PaddingPolicy_h

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface Pool6PaddingPolicy : NSObject <MPSNNPadding, NSSecureCoding>

- (nonnull instancetype)init;
- (MPSNNPaddingMethod)paddingMethod;
- (MPSImageDescriptor *)destinationImageDescriptorForSourceImages:(nonnull NSArray<MPSImage *> *)sourceImages
                                                     sourceStates:(nonnull NSArray<MPSState *> *)sourceStates
                                                        forKernel:(nonnull MPSKernel *)kernel
                                              suggestedDescriptor:(nonnull MPSImageDescriptor *)inDescriptor;

@end

#endif /* PaddingPolicy_h */
