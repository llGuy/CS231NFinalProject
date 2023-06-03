//
//  Time.cpp
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/2/23.
//

#include <chrono>

struct TimeStamp
{
   std::chrono::high_resolution_clock::time_point now;
};

extern "C" struct TimeStamp *allocTimeStamp(void);
extern "C" void freeTimeStamp(struct TimeStamp *);
extern "C" void getCurrentTime(struct TimeStamp *);
extern "C" float getTimeDifference(struct TimeStamp *end, struct TimeStamp *start);

struct TimeStamp *allocTimeStamp(void)
{
    return new TimeStamp;
}

void freeTimeStamp(struct TimeStamp *ts)
{
    delete ts;
}

void getCurrentTime(struct TimeStamp *ts)
{
    ts->now = std::chrono::high_resolution_clock::now();
}

float getTimeDifference(struct TimeStamp *end, struct TimeStamp *start)
{
  std::chrono::duration<float> seconds = end->now - start->now;
  float delta = seconds.count();
  return (float)delta;
}
