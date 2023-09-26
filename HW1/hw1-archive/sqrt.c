#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void fallback_print_usage() {
  printf("Usage: ./sqrt number\n");
  printf("Example: ./sqrt 2\n");
  exit(0);
}

void print_sqrt(double number) { printf("%.8lf\n", sqrt(number)); }

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fallback_print_usage();
  }
  print_sqrt(atof(argv[1]));
  return 0;
}
