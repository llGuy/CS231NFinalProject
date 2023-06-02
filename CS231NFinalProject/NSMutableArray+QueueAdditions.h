//
//  NSMutableArray+QueueAdditions.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#ifndef NSMutableArray_QueueAdditions_h
#define NSMutableArray_QueueAdditions_h

@interface NSMutableArray (QueueAdditions)

-(id)dequeue;
-(void)enqueue:(id)obj;

@end

#endif /* NSMutableArray_QueueAdditions_h */
