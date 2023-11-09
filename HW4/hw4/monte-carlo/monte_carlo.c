#include <mpi.h>
#include <stdio.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  int count = 0;

  // TODO: Parallelize the code using mpi_world_size processes (1 process per
  // node.
  // In total, (mpi_world_size * threads_per_process) threads will collaborate
  // to compute pi.

  if (mpi_rank == 0) {
    for (int i = 0; i < num_points; i++) {
      double x = xs[i];
      double y = ys[i];

      if (x*x + y*y <= 1)
        count++;
    }
  }


  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return (double) 4 * count / num_points;
}
