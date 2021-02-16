#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "../../utils/cluster_utils.hpp"
#include "../../utils/io_utils.hpp"
#include "CredibleBall.hpp"

using namespace std;
using namespace Eigen;

template <typename M>
M load_csv(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<int> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ' ')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, RowMajor>>(
      values.data(), rows, values.size() / rows);
}

int main(int argc, char const *argv[]) {
  cout << "Credible-balls test" << endl;

  if (argc != 6) {
    throw domain_error(
        "Syntax : ./run_cb filename_mcmc filename_pe filename_out loss rate");
  }

  string filename_mcmc = argv[1];
  string filename_pe = argv[2];
  string filename_out = argv[3];
  int loss_type = std::stoi(argv[4]);
  double learning_rate = stoi(argv[5]);

  Eigen::MatrixXi mcmc, pe_tmp;
  Eigen::VectorXi pe;
  cout << "ok1" << endl;
  pe_tmp = load_csv<MatrixXi>(filename_pe);
  cout << "ok2" << endl;
  pe = pe_tmp.row(0);
  cout << "ok3" << endl;
  mcmc = load_csv<MatrixXi>(filename_mcmc);

  cout << "Matrix with dimensions : " << mcmc.rows() << "*" << mcmc.cols()
       << " found." << endl;

  CredibleBall CB =
      CredibleBall(static_cast<LOSS_FUNCTION>(loss_type), mcmc, 0.05, pe);

  double epsilon = CB.calculateRegion(learning_rate);
  /*
  Eigen::MatrixXi VUB = CB.VerticalUpperBound();
  Eigen::MatrixXi VLB = CB.VerticalLowerBound();
  Eigen::MatrixXi HB = CB.HorizontalBound();
*/
  return 0;
}
