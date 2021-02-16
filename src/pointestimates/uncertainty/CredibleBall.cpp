#include "CredibleBall.hpp"

using namespace std;

void transform_row2vec(Eigen::MatrixXi matrix, int row, Eigen::VectorXi vec) {
  if (row > matrix.rows()) {
    cout << "Error, row out of bounds!" << endl;
  }
  for (int j = 0; j < matrix.cols(); j++) {
    vec(j) = matrix(row, j);
  }

  return;
}

void add_row2matrix(Eigen::MatrixXi trg, int row, Eigen::MatrixXi src,
                    int index) {
  // we add row index of src to row "row" of trg
  int col = src.cols();

  for (int i = 0; i < col; i++) {
    trg(row, i) = src(index, i);
  }
}

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
  double epsilon = 0.0;
  double probability = 0.0;
  int steps = 1;
  loss_function->SetFirstCluster(point_estimate);
  cout << "Assessing the value of the radius..."
       << "\n";

  while (1) {
    epsilon += rate * steps;

    for (int i = 0; i < T; i++) {
      Eigen::VectorXi vec;
      transform_row2vec(mcmc_sample.cast<int>(), i, vec);
      loss_function->SetSecondCluster(vec);
      if (loss_function->Loss() <= epsilon) {
        probability += 1;
      }
    }

    probability /= T;

    if (probability >= 1 - alpha) {
      cout << "Radius estimated: " << epsilon << endl;
      radius = epsilon;
      populateCredibleSet();
      break;
    }

    steps++;
    probability = 0;
  }

  return epsilon;
}

void CredibleBall::populateCredibleSet() {
  loss_function->SetFirstCluster(point_estimate);

  for (int i = 0; i < T; i++) {
    Eigen::VectorXi vec;
    transform_row2vec(mcmc_sample.cast<int>(), i, vec);
    loss_function->SetSecondCluster(vec);

    if (loss_function->Loss() <= radius) {
      for (int j = 0; j < N; j++) {
        credibleBall(i, j) = mcmc_sample(i, j);
      }
    }
  }
}

int CredibleBall::count_cluster_row(int row) {
  // Returns the number of partitions in a cluster in a row of the credible
  // ball
  int total_row = credibleBall.rows();
  if (row > total_row) {
    cout << "Row out of bounds!" << endl;
    return -1;
  }
  int col = credibleBall.cols();
  set<int, greater<int>> s;

  for (int i = 0; i < col; i++) {
    int tmp = credibleBall(row, i);
    s.insert(tmp);
  }

  return s.size();
}

Eigen::MatrixXi CredibleBall::VerticalUpperBound() {
  Eigen::VectorXi vec1, vec2;
  // vec1 has the indexes of the clusters with minimum cardinality in the ball
  // vec2 is an auxiliary vector
  Eigen::MatrixXi vub;             // the output
  int rows = credibleBall.rows();  // size of the credible ball
  int tmp1;
  int min = INT_MAX;

  // find the least cardinality among the clusters
  for (int i = 0; i < rows; i++) {
    tmp1 = count_cluster_row(i);
    if (tmp1 <= min) {
      min = tmp1;
    }
  }

  // save the indexes of the clusters with min cardinality
  int aux = 0;
  for (int i = 0; i < rows; i++) {
    if (count_cluster_row(i) == min) {
      vec1(aux) = i;
      aux++;
    }
  }

  // among the clusters with min cardinality find the max distance
  // from the point_estimate
  double max_distance = -1.0;

  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    // we save the row vec(j) of the credibleBall in vec2
    transform_row2vec(credibleBall, vec1(i), vec2);
    // compute the distance, ie the loss for the corresponding row
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss > max_distance) {
      max_distance = loss;
    }
  }

  // select the index of the clusters with that max distance with least
  // cardinality
  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    transform_row2vec(credibleBall, vec1(i), vec2);
    loss_function->SetSecondCluster(vec2);

    double loss = loss_function->Loss();
    int aux = 0;
    if (loss == max_distance) {
      add_row2matrix(vub, aux, credibleBall, vec1(i));
      aux++;
    }
  }

  return vub;
}

Eigen::MatrixXi CredibleBall::VerticalLowerBound() {
  Eigen::VectorXi vec1, vec2;
  // vec1 has the indexes of the clusters with maximum cardinality in the ball
  // vec2 is an auxiliary vector
  Eigen::VectorXi vlb;
  int rows = credibleBall.rows();
  int tmp1;
  int max = -1;

  // find the highest cardinality among the clusters
  for (int i = 0; i < rows; i++) {
    tmp1 = count_cluster_row(i);
    if (tmp1 >= max) {
      max = tmp1;
    }
  }

  // save the indexes of the clusters with max cardinality
  int aux = 0;
  for (int i = 0; i < rows; i++) {
    if (count_cluster_row(i) == max) {
      vec1(aux) = i;
      aux++;
    }
  }

  // among the clusters with max cardinality find the max distance
  // from the point_estimate
  double max_distance = 0.0;

  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    transform_row2vec(credibleBall, vec1(i), vec2);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    if (loss > max_distance) {
      max_distance = loss;
    }
  }

  // select the index of the clusters with that max distance with max
  // cardinality
  for (int i = 0; i < vec1.size(); i++) {
    loss_function->SetFirstCluster(point_estimate);
    transform_row2vec(credibleBall, vec1(i), vec2);
    loss_function->SetSecondCluster(vec2);
    double loss = loss_function->Loss();

    int aux = 0;
    if (loss == max_distance) {
      add_row2matrix(vlb, aux, credibleBall, vec1(i));
      aux++;
    }
  }

  return vlb;
}

Eigen::MatrixXi CredibleBall::HorizontalBound() {
  // clusters in the ball that are more distat from the center
  Eigen::MatrixXi hb;
  Eigen::VectorXi vec;
  int row = credibleBall.rows();
  // int col = credibleBall.cols();
  int aux = 0;
  double max = -1.0;

  // find the max distance among all the clusters in the credible ball
  for (int i = 0; i < row; i++) {
    loss_function->SetFirstCluster(point_estimate);
    transform_row2vec(credibleBall, i, vec);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    if (loss >= max) {
      max = loss;
    }
  }

  // select the clusters with that index
  for (int i = 0; i < row; i++) {
    loss_function->SetFirstCluster(point_estimate);
    transform_row2vec(credibleBall, i, vec);
    loss_function->SetSecondCluster(vec);
    double loss = loss_function->Loss();

    int aux = 0;
    if (loss == max) {
      add_row2matrix(hb, aux, credibleBall, vec(i));
      aux++;
    }
  }

  return hb;
}
