//
//  Time.h
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#ifndef Time_h
#define Time_h

struct TimeStamp;

struct TimeStamp *allocTimeStamp(void);
void freeTimeStamp(struct TimeStamp *);
void getCurrentTime(struct TimeStamp *);
float getTimeDifference(struct TimeStamp *end, struct TimeStamp *start);

#endif /* Time_h */
