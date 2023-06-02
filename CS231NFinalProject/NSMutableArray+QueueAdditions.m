//
//  NSMutableArray+QueueAdditions.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#import <Foundation/Foundation.h>

@implementation NSMutableArray (QueueAdditions)

-(id)dequeue;
{
    id headObject = [self objectAtIndex:0];
    if (headObject != nil) {
        [self removeObjectAtIndex:0];
    }
    return headObject;
}

- (void) enqueue:(id)obj;
{
    [self addObject:obj];
}

@end
