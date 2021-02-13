#include "CredibleBall.hpp"

#include <cstdlib>
#include <iostream>

using namespace std;

CredibleBall::CredibleBall(LOSS_FUNCTION loss_type,
                           Eigen::MatrixXi& mcmc_sample_, double alpha_,
                           Eigen::VectorXi& point_estimate_)
    : loss_function(0) {
  cout << "[CONSTRUCTORS]" << endl;
  cout << "CredibleBall Constructor" << endl;
  switch (loss_type) {
    case BINDER_LOSS:
      loss_function = new BinderLoss(1, 1);
      break;
    case VARIATION_INFORMATION: {
      loss_function = new VariationInformation(false);
      break;
    }
    case VARIATION_INFORMATION_NORMALIZED: {
      loss_function = new VariationInformation(true);
      break;
    }
    default:
      throw std::domain_error("Loss function not recognized");
  }
  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  N = mcmc_sample.cols();
  alpha = alpha_;
  point_estimate = point_estimate_;
}

CredibleBall::~CredibleBall() {
  cout << "[Destructor]" << endl;
  cout << "CredibleBall Destructor" << endl;
  delete loss_function;
}

double CredibleBall::calculateRegion(double rate) {
  double episilon = 0.0;
  double probability = 0.0;
  int steps = 1;
  loss_function->SetFirstCluster(point_estimate);

  while (1) {
    episilon += rate * steps;

    for (int i = 0; i < T; i++) {
      loss_function->SetSecondCluster(mcmc_sample.row(i));
      if (loss_function->Loss() <= episilon) {
        probability += 1;
      }
    }

    probability /= T;

    if (probability >= 1 - alpha) {
      radius = episilon;
      populateCredibleSet();
      break;
    }

    steps++;
    probability = 0;
  }

  return episilon;
}

void CredibleBall::populateCredibleSet() {
  loss_function->SetFirstCluster(point_estimate);

  for (int i = 0; i < T; i++) {
    loss_function->SetSecondCluster(mcmc_sample.row(i));

    if (loss_function->Loss() <= radius) {
      for (int j = 0; j < N; j++) {
        credibleBall(i, j) = mcmc_sample(i, j);
      }
    }
  }
}

Eigen::MatrixXd CredibleBall::VerticalUpperBound() {}
Eigen::MatrixXd CredibleBall::VerticalLowerBound() {}

Eigen::MatrixXd CredibleBall::HorizontalBound() {}