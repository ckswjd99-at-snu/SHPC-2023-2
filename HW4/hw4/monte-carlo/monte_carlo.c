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

  int node_start = num_points * mpi_rank / mpi_world_size;
  int node_end = num_points * (mpi_rank + 1) / mpi_world_size;
  int node_num_points = node_end - node_start;

  int local_count = 0;

  if (mpi_rank == 0) {
    MPI_Scatter(xs, node_num_points, MPI_DOUBLE, MPI_IN_PLACE, node_num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(ys, node_num_points, MPI_DOUBLE, MPI_IN_PLACE, node_num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Scatter(NULL, node_num_points, MPI_DOUBLE, xs, node_num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, node_num_points, MPI_DOUBLE, ys, node_num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  #pragma omp parallel for reduction(+:local_count) num_threads(threads_per_process)
  for (int i = 0; i < node_num_points; i++) {
    double x = xs[i];
    double y = ys[i];

    if (x*x + y*y <= 1)
      local_count++;
  }

  MPI_Reduce(&local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return (double) 4 * count / num_points;
}
