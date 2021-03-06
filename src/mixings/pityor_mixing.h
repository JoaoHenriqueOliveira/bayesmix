#ifndef BAYESMIX_MIXINGS_PITYOR_MIXING_H_
#define BAYESMIX_MIXINGS_PITYOR_MIXING_H_

#include "base_mixing.h"
#include "mixing_prior.pb.h"

//! Class that represents the Pitman-Yor process mixture model.

//! This class represents a particular mixture model for iterative BNP
//! algorithms, called the Pitman-Yor process. It has two parameters, strength
//! and discount. It is a generalized version of the Dirichlet process, which
//! has discount = 0 and strength = total mass. In terms of the algorithms, it
//! translates to a mixture that assigns a weight to the creation of a new
//! cluster proportional to their cardinalities, but reduced by the discount
//! factor, while the weight for a newly created cluster is the remaining
//! one counting the total amount as the sample size increased by the strength.

class PitYorMixing : public BaseMixing {
 public:
  struct State {
    double strength, discount;
  };

 protected:
  State state;
  std::shared_ptr<bayesmix::PYPrior> prior;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~PitYorMixing() = default;
  PitYorMixing() = default;

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster
  double mass_existing_cluster(std::shared_ptr<BaseHierarchy> hier,
                               const unsigned int n, bool log,
                               bool propto) const override;
  //! Mass probability for choosing a newly created cluster
  double mass_new_cluster(const unsigned int n_clust, const unsigned int n,
                          bool log, bool propto) const override;

  void initialize() override;

  void update_state(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  std::string get_id() const override { return "PY"; }
};

#endif  // BAYESMIX_MIXINGS_PITYOR_MIXING_H_
