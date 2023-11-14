#ifndef TIMER_H
#define TIMER_H

void timer_init(int n);

void timer_finalize();

void timer_start(int idx);

void timer_stop(int idx);

double timer_read(int idx);

void timer_reset(int idx);

#endif
