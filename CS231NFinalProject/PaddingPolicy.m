//
//  PaddingPolicy.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#import <Foundation/Foundation.h>

#import "PaddingPolicy.h"

@implementation Pool6PaddingPolicy

- (nonnull instancetype)init
{
    self = [super init];
    
    return self;
}

- (MPSNNPaddingMethod)paddingMethod
{
    return MPSNNPaddingMethodCustom | MPSNNPaddingMethodSizeSame;
}

- (MPSImageDescriptor *)destinationImageDescriptorForSourceImages:(NSArray<MPSImage *> *)sourceImages
                                                     sourceStates:(NSArray<MPSState *> *)sourceStates
                                                        forKernel:(MPSKernel *)kernel
                                              suggestedDescriptor:(MPSImageDescriptor *)inDescriptor
{
    if ([kernel isKindOfClass:[MPSCNNPooling class]])
    {
        MPSCNNPooling *poolingKernel = (MPSCNNPooling *)kernel;
        MPSOffset offset = {1, 1, 0};
        poolingKernel.offset = offset;
        poolingKernel.edgeMode = MPSImageEdgeModeClamp;
    }
    
    return inDescriptor;
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder
{
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder
{
    return self;
}

- (NSString *)label
{
    return @"Pool6Padding";
}

@end
