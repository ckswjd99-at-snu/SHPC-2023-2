#pragma once

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "classifier.h"

using namespace std;

void print_help();
void parse_option(int, char **, int *, bool *);
void *read_binary(const char *);
double get_time();
