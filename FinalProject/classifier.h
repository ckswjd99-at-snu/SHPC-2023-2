#pragma once

#include <string>
#include <vector>

#define Assert(expression, message) assert(expression &&message)

using namespace std;

#define OFFSET0 0
#define OFFSET1 OFFSET0 + 256 * 70 * 7
#define OFFSET2 OFFSET1 + 256
#define OFFSET3 OFFSET2 + 256 * 1008
#define OFFSET4 OFFSET3 + 256 * 1008
#define OFFSET5 OFFSET4 + 256 * 256 * 7
#define OFFSET6 OFFSET5 + 256
#define OFFSET7 OFFSET6 + 256 * 256 * 3
#define OFFSET8 OFFSET7 + 256
#define OFFSET9 OFFSET8 + 256 * 256 * 3
#define OFFSET10 OFFSET9 + 256
#define OFFSET11 OFFSET10 + 256 * 256 * 3
#define OFFSET12 OFFSET11 + 256
#define OFFSET13 OFFSET12 + 256 * 256 * 3
#define OFFSET14 OFFSET13 + 256
#define OFFSET15 OFFSET14 + 256 * 102
#define OFFSET16 OFFSET15 + 256 * 102
#define OFFSET17 OFFSET16 + 1024 * 8704
#define OFFSET18 OFFSET17 + 1024
#define OFFSET19 OFFSET18 + 1024 * 1024
#define OFFSET20 OFFSET19 + 1024
#define OFFSET21 OFFSET20 + 4 * 1024

#define MAX_LENGTH 1014
#define VOCAB_SIZE 70

void initialize_classifier(float *, int);
void finalize_classifier();
void classifier(float *, float *, int);
