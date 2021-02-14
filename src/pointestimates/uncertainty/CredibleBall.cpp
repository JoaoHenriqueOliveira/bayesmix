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

int CredibleBall::count_cluster_row(int index) {
  int row = credibleBall.rows();
  if (index > row) {
    cout << "Row out of bounds!" << endl;
    return -1;
  }
  int col = credibleBall.cols();
  int count = 0;

  for (int i = 0; i < col; i++) {
    for (int j = i + 1; j < col; j++) {
      if (credibleBall(index, i) != credibleBall(index, j)) {
        count++;
      }
    }
  }

  return count;
}

Eigen::VectorXi transform_row2vec(Eigen::MatrixXi matrix, int row) {
  if (row > matrix.rows()) {
    return -1;
  }
  Eigen::VectorXi vec;
  for (int j = 0; j < matrix.cols(); j++) {
    vec(j) = matrix(row, j);
  }

  return vec;
}

Eigen::VectorXi CredibleBall::VerticalUpperBound() {
  Eigen::VectorXi vec1, vec2;
  Eigen::VectorXi vub;
  int rows = credibleBall.rows();
  int tmp1, tmp2;
  int min = 0;

  // find the least cardinality among the clusters
  for (int i = 0; i < rows; i++) {
    tmp1 = count_cluster_row(i);
    for (int j = i + 1; j < rows; j++) {
      tmp2 = count_cluster_row(j);
      if (tmp1 <= tmp2) {
        min = tmp1;
      }
    }
  }

  if (min == 0) {
    cout << "Something wrong ain't right.";
    return -1;
  }

  // save the index of the cluster with min cardinality
  int aux = 0;
  for (int i = 0; i < rows; i++) {
    if (count_cluster_row(i) == min) {
      vec1(aux) = i;
      aux++;
    }
  }

  // among the clusters with min cardinality find the max distance
  // from the point_estimate
  double var1 = 0.0;

  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    for (int j = i + 1; j < vec1.size(); j++) {
      vec2 = transform_row2vec(credibleBall, vec1(j));
      loss_function->SetSecondCluster(vec2);
      double loss = loss_function->Loss();
      if (loss > var1) {
        var1 = loss;
      }
    }
  }

  // select the index of the clusters with that max distance with least
  // cardinality
  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    for (int j = i + 1; j < vec1.size(); j++) {
      vec2 = transform_row2vec(credibleBall, vec1(j));
      loss_function->SetSecondCluster(vec2);
      double loss = loss_function->Loss();
      if (loss == var1) {
        vub(i) = i;
      }
    }
  }

  return vub;
}

Eigen::MatrixXd CredibleBall::VerticalLowerBound() {
  Eigen::VectorXi vec1, vec2;
  Eigen::VectorXi vlb;
  int rows = credibleBall.rows();
  int tmp1, tmp2;
  int max = 0;

  // find the highest cardinality among the clusters
  for (int i = 0; i < rows; i++) {
    tmp1 = count_cluster_row(i);
    for (int j = i + 1; j < rows; j++) {
      tmp2 = count_cluster_row(j);
      if (tmp1 >= tmp2) {
        max = tmp1;
      }
    }
  }

  if (max == 0) {
    cout << "Something wrong ain't right.";
    return -1;
  }

  // save the index of the cluster with min cardinality
  int aux = 0;
  for (int i = 0; i < rows; i++) {
    if (count_cluster_row(i) == max) {
      vec1(aux) = i;
      aux++;
    }
  }

  // among the clusters with max cardinality find the max distance
  // from the point_estimate
  double var1 = 0.0;

  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    for (int j = i + 1; j < vec1.size(); j++) {
      vec2 = transform_row2vec(credibleBall, vec1(j));
      loss_function->SetSecondCluster(vec2);
      double loss = loss_function->Loss();
      if (loss > var1) {
        var1 = loss;
      }
    }
  }

  // select the index of the clusters with that max distance with max
  // cardinality
  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    for (int j = i + 1; j < vec1.size(); j++) {
      vec2 = transform_row2vec(credibleBall, vec1(j));
      loss_function->SetSecondCluster(vec2);
      double loss = loss_function->Loss();
      if (loss == var1) {
        vlb(i) = i;
      }
    }
  }

  return vlb;
}

Eigen::VectorXi CredibleBall::HorizontalBound() {
  Eigen::VectorXi hb;
  Eigen::VectorXi vec;
  int row = credibleBall.rows();
  int col = credibleBall.cols();
  int aux = 0;
  double max = 0.0;

  // find the max distance among all the clusters in the credible ball
  for (int i = 0; i < row; i++) {
    loss_function->SetFirstCluster(point_estimate);
    vec = transform_row2vec(credibleBall, i);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    if (loss >= max) {
      max = loss;
    }
  }

  // select the clusters with that index
  for (int i = 0; i < row; i++) {
    loss_function->SetFirstCluster(point_estimate);
    vec = transform_row2vec(credibleBall, i);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    if (loss == max) {
      hb(aux) = i;
      aux++;
    }
  }

  return hb;
}
