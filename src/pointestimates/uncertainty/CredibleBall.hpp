#ifndef CREDIBLE_BALL_HPP
#define CREDIBLE_BALL_HPP

#include "../../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "../lossfunction/BinderLoss.hpp"
#include "../lossfunction/LossFunction.hpp"
#include "../lossfunction/VariationInformation.hpp"

enum LOSS_FUNCTION {
  BINDER_LOSS,
  VARIATION_INFORMATION,
  VARIATION_INFORMATION_NORMALIZED
};

class CredibleBall {
 private:
  LossFunction* loss_function;     // metric to compute the region
  Eigen::MatrixXi mcmc_sample;     // MCMC matrix of clusters
  Eigen::VectorXi point_estimate;  // output of the greedy algorithm
  Eigen::MatrixXd credibleBall;  // set of clusters inside the credible region
  int T;                         // number of clusters, ie mcmc_sample.rows
  int N;                         // dimension of cluster, ie mcmc_sample.cols
  double alpha;                  // level of the credible ball region
  double radius;                 // radius of the credible ball

 public:
  CredibleBall(LOSS_FUNCTION loss_type_, Eigen::MatrixXi& mcmc_sample_,
               double alpha_, Eigen::VectorXi& point_estimate_);
  ~CredibleBall();
  double calculateRegion(double rate);  // calculate the radius
  Eigen::MatrixXd VerticalUpperBound();
  Eigen::MatrixXd VerticalLowerBound();
  Eigen::MatrixXd HorizontalBound();

 private:
  void populateCredibleSet();  // populate the credibleBall
};
#endif  // CREDIBLE_BALL_HPP
