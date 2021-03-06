#include "nnw_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "marginal_state.pb.h"
#include "matrix.pb.h"
#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

//! \param prec_ Value to set to prec
void NNWHierarchy::set_prec_and_utilities(const Eigen::MatrixXd &prec_) {
  state.prec = prec_;

  // Update prec utilities
  prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec_).matrixL().transpose();
  Eigen::VectorXd diag = prec_chol.diagonal();
  prec_logdet = 2 * log(diag.array()).sum();
}

void NNWHierarchy::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
  state.mean = hypers->mean;
  set_prec_and_utilities(hypers->var_scaling *
                         Eigen::MatrixXd::Identity(dim, dim));
  clear_data();
}

void NNWHierarchy::clear_data() {
  data_sum = Eigen::VectorXd::Zero(dim);
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
  card = 0;
  cluster_data_idx = std::set<int>();
}

void NNWHierarchy::update_summary_statistics(const Eigen::VectorXd &datum,
                                             bool add) {
  if (add) {
    data_sum += datum;
    data_sum_squares += datum * datum.transpose();
  } else {
    data_sum -= datum;
    data_sum_squares -= datum * datum.transpose();
  }
}

//! \param data                    Matrix of row-vectorial data points
//! \param mu0, lambda0, tau0, nu0 Original values for hyperparameters
//! \return                        Vector of updated values for hyperparameters
NNWHierarchy::Hyperparams NNWHierarchy::normal_wishart_update() {
  // Initialize relevant objects
  Hyperparams post_params;

  // Compute updated hyperparameters
  post_params.var_scaling = hypers->var_scaling + card;
  post_params.deg_free = hypers->deg_free + 0.5 * card;

  Eigen::VectorXd mubar = data_sum.array() / card;  // sample mean
  post_params.mean = (hypers->var_scaling * hypers->mean + card * mubar) /
                     (hypers->var_scaling + card);
  // Compute tau_n
  Eigen::MatrixXd tau_temp =
      data_sum_squares - card * mubar * mubar.transpose();
  tau_temp += (card * hypers->var_scaling / (card + hypers->var_scaling)) *
              (mubar - hypers->mean) * (mubar - hypers->mean).transpose();
  tau_temp = 0.5 * tau_temp + hypers->scale_inv;
  post_params.scale = stan::math::inverse_spd(tau_temp);
  return post_params;
}

void NNWHierarchy::update_hypers(
    const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else if (prior->has_normal_mean_prior()) {
    // Get hyperparameters
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().mean());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().var());
    double lambda0 = prior->normal_mean_prior().var_scaling();
    // Compute posterior hyperparameters
    Eigen::MatrixXd sigma00inv = stan::math::inverse_spd(sigma00);
    Eigen::MatrixXd prec = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd num = Eigen::MatrixXd::Zero(dim, 1);
    for (auto &st : states) {
      Eigen::MatrixXd prec_i = bayesmix::to_eigen(st.multi_ls_state().prec());
      prec += prec_i;
      num += prec_i * bayesmix::to_eigen(st.multi_ls_state().mean());
    }
    prec = hypers->var_scaling * prec + sigma00inv;
    num = hypers->var_scaling * num + sigma00inv * mu00;
    Eigen::VectorXd mu_n = prec.llt().solve(num);
    // Update hyperparameters with posterior sampling
    hypers->mean = stan::math::multi_normal_prec_rng(mu_n, prec, rng);
  }

  else if (prior->has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().mean());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().var());
    // for lambda0
    double alpha00 = prior->ngiw_prior().var_scaling_prior().shape();
    double beta00 = prior->ngiw_prior().var_scaling_prior().rate();
    // for tau0
    double nu00 = prior->ngiw_prior().scale_prior().deg_free();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior->ngiw_prior().scale_prior().scale());
    // Compute posterior hyperparameters
    Eigen::MatrixXd sigma00inv = stan::math::inverse_spd(sigma00);
    Eigen::MatrixXd tau_n = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd num = Eigen::MatrixXd::Zero(dim, 1);
    double beta_n = 0.0;
    for (auto &st : states) {
      Eigen::VectorXd mean = bayesmix::to_eigen(st.multi_ls_state().mean());
      Eigen::MatrixXd prec = bayesmix::to_eigen(st.multi_ls_state().prec());
      tau_n += prec;
      num += prec * mean;
      beta_n +=
          (hypers->mean - mean).transpose() * prec * (hypers->mean - mean);
    }
    Eigen::MatrixXd prec_n = hypers->var_scaling * tau_n + sigma00inv;
    tau_n += tau00;
    num = hypers->var_scaling * num + sigma00inv * mu00;
    beta_n = beta00 + 0.5 * beta_n;
    Eigen::MatrixXd sig_n = stan::math::inverse_spd(prec_n);
    Eigen::VectorXd mu_n = sig_n * num;
    double alpha_n = alpha00 + 0.5 * states.size();
    double nu_n = nu00 + states.size() * hypers->deg_free;
    // Update hyperparameters with posterior random Gibbs sampling
    hypers->mean = stan::math::multi_normal_rng(mu_n, sig_n, rng);
    hypers->var_scaling = stan::math::gamma_rng(alpha_n, beta_n, rng);
    hypers->scale = stan::math::inv_wishart_rng(nu_n, tau_n, rng);
    hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

//! \param data Matrix of row-vectorial single data point
//! \return     Log-Likehood vector evaluated in data
double NNWHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  // Initialize relevant objects
  return bayesmix::multi_normal_prec_lpdf(datum, state.mean, prec_chol,
                                          prec_logdet);
}

//! \param data Matrix of row-vectorial a single data point
//! \return     Marginal distribution vector evaluated in data
double NNWHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
  // Compute dof and scale of marginal distribution
  double nu_n = 2 * hypers->deg_free - dim + 1;
  Eigen::MatrixXd sigma_n = hypers->scale_inv *
                            (hypers->deg_free - 0.5 * (dim - 1)) *
                            hypers->var_scaling / (hypers->var_scaling + 1);

  // TODO: chec if this is optimized as our bayesmix::multi_normal_prec_lpdf
  return stan::math::multi_student_t_lpdf(datum, nu_n, hypers->mean, sigma_n);
}

void NNWHierarchy::draw() {
  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(hypers->deg_free, hypers->scale, rng);

  // Update state
  state.mean = stan::math::multi_normal_prec_rng(
      hypers->mean, tau_new * hypers->var_scaling, rng);
  set_prec_and_utilities(tau_new);
}

//! \param data Matrix of row-vectorial data points
void NNWHierarchy::sample_given_data() {
  // Update values
  Hyperparams params = normal_wishart_update();

  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(params.deg_free, params.scale, rng);
  state.mean = stan::math::multi_normal_prec_rng(
      params.mean, tau_new * params.var_scaling, rng);

  // Update state
  set_prec_and_utilities(tau_new);
}

void NNWHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  data_sum = Eigen::VectorXd::Zero(data.cols());
  data_sum_squares = Eigen::MatrixXd::Zero(data.cols(), data.cols());

  for (int i = 0; i < data.rows(); i++) {
    data_sum += data.row(i);
    data_sum_squares += data.row(i).transpose() * data.row(i);
  }
  card = data.rows();
  log_card = std::log(card);
  sample_given_data();
}

void NNWHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.mean = to_eigen(statecast.multi_ls_state().mean());
  set_prec_and_utilities(to_eigen(statecast.multi_ls_state().prec()));
  set_card(statecast.cardinality());
}

void NNWHierarchy::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::NNWPrior &>(
          prior_);
  prior = std::make_shared<bayesmix::NNWPrior>(priorcast);
  hypers = std::make_shared<Hyperparams>();
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = bayesmix::to_eigen(prior->fixed_values().mean());
    dim = hypers->mean.size();
    hypers->var_scaling = prior->fixed_values().var_scaling();
    hypers->scale = bayesmix::to_eigen(prior->fixed_values().scale());
    hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
    hypers->deg_free = prior->fixed_values().deg_free();
    // Check validity
    if (hypers->var_scaling <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (dim != hypers->scale.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    if (hypers->deg_free <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
  }

  else if (prior->has_normal_mean_prior()) {
    // Get hyperparameters
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().mean());
    dim = mu00.size();
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().var());
    double lambda0 = prior->normal_mean_prior().var_scaling();
    Eigen::MatrixXd tau0 =
        bayesmix::to_eigen(prior->normal_mean_prior().scale());
    double nu0 = prior->normal_mean_prior().deg_free();
    // Check validity
    unsigned int dim = mu00.size();
    if (sigma00.rows() != dim or tau0.rows() != dim) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    bayesmix::check_spd(sigma00);
    if (lambda0 <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    bayesmix::check_spd(tau0);
    if (nu0 <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = lambda0;
    hypers->scale = tau0;
    hypers->scale_inv = stan::math::inverse_spd(tau0);
    hypers->deg_free = nu0;
  }

  else if (prior->has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().mean());
    dim = mu00.size();
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().var());
    // for lambda0
    double alpha00 = prior->ngiw_prior().var_scaling_prior().shape();
    double beta00 = prior->ngiw_prior().var_scaling_prior().rate();
    // for tau0
    double nu00 = prior->ngiw_prior().scale_prior().deg_free();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior->ngiw_prior().scale_prior().scale());
    // for nu0
    double nu0 = prior->ngiw_prior().deg_free();
    // Check validity:
    // dimensionality
    if (sigma00.rows() != dim or tau00.rows() != dim) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    // for mu0
    bayesmix::check_spd(sigma00);
    // for lambda0
    if (alpha00 <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (beta00 <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    // for tau0
    if (nu00 <= 0) {
      throw std::invalid_argument("Degrees of freedom parameter must be > 0");
    }
    bayesmix::check_spd(tau00);
    // check nu0
    if (nu0 <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = alpha00 / beta00;
    hypers->scale = tau00 / (nu00 + dim + 1);
    hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
    hypers->deg_free = nu0;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void NNWHierarchy::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::MultiLSState state_;
  bayesmix::to_proto(state.mean, state_.mutable_mean());
  bayesmix::to_proto(state.prec, state_.mutable_prec());
  auto *out_cast = google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out);
  out_cast->mutable_multi_ls_state()->CopyFrom(state_);
  out_cast->set_cardinality(card);
}

void NNWHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::NNWPrior hypers_;
  bayesmix::to_proto(hypers->mean,
                     hypers_.mutable_fixed_values()->mutable_mean());
  hypers_.mutable_fixed_values()->set_var_scaling(hypers->var_scaling);
  hypers_.mutable_fixed_values()->set_deg_free(hypers->deg_free);
  bayesmix::to_proto(hypers->scale,
                     hypers_.mutable_fixed_values()->mutable_scale());

  google::protobuf::internal::down_cast<bayesmix::NNWPrior *>(out)
      ->mutable_fixed_values()
      ->CopyFrom(hypers_.fixed_values());
}
