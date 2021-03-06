#include "conditional_algorithm.h"

#include <Eigen/Dense>

#include "src/collectors/base_collector.h"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd ConditionalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                                BaseCollector *const coll) {
  return Eigen::MatrixXd::Zero(1, 1);
}
