#ifndef WINDOWHMM_H
#define WINDOWHMM_H

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <random>
#include "../thirdparty/json.hpp"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class WindowHMM {
public:
    WindowHMM(int n_snp, int n_proto);

    void fit(const MatrixXi &X);

    void init_emit_from_X(const MatrixXi &X);
    void init_from_file(const string &path);

    // model parameters
    VectorXd start;
    vector<MatrixXd> trans;
    MatrixXd emit; // Probability of emitting a minor allele `1` (n_snp, n_proto)

    double compute_total_loglkl(const VectorXi &x);
    void compute_alpha_beta(const VectorXi &x, MatrixXd & log_alpha, MatrixXd & log_beta);
    void print_suff_stat();
    void print_param();

private:
    void init_random_params(const MatrixXi & X);
    void compute_obs_loglkl(const VectorXi &x, MatrixXd &log_prob);

    void forward_pass(const MatrixXd &log_obs, MatrixXd &fwd_lattice, VectorXd &fwd_cond_prob);

    void backward_pass(const MatrixXd &log_obs, MatrixXd &bwd_lattice, VectorXd &bwd_cond_prob);

    void compute_posterior(const MatrixXd &fwd_lattice, const MatrixXd &bwd_lattice, MatrixXd &posterior);

    void init_suff_stat();

    void accum_suff_stat(const VectorXi &x, const MatrixXd &log_obs, const MatrixXd &fwd_lattice,
                         const MatrixXd &bwd_lattice, const MatrixXd &posterior);


    void compute_xi(const MatrixXd &log_obs,
                    const MatrixXd &fwd_lattice,
                    const MatrixXd &bwd_lattice);

    void do_mstep();

    default_random_engine random_engine;

    // preset constants
    const double rel_tol = 0.01;
    const int max_iter = 100;
    const double proto_bias = 0.9;
    const double trans_prior = 0.00006;
    const double emit_prior = 0.001;

    const int n_snp;
    const int n_proto;
    int n_hap = -1;

    // sufficient statistics
    VectorXd suff_stat_start;
    vector<MatrixXd> suff_stat_trans;
    vector<MatrixXd> suff_stat_emit;

    // temporary variable
    vector<MatrixXd> tmp_xi; // (n_snp - 1) x n_proto x n_proto
    VectorXd tmp_snp_vec; // n_snp
    VectorXd tmp_proto_vec; // n_proto

};


#endif
