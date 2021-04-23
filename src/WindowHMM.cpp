#include "WindowHMM.h"
#include "Utils.h"

void WindowHMM::init_from_file(const string &path) {
    using json = nlohmann::json;
    ifstream file(path);
    json data;
    file >> data;

    start = from_json(data["pi"]);
    for (int i = 0; i < data["trans"].size(); i++) {
        trans[i] = from_json(data["trans"].at(i));
    }

    emit = from_json(data["emit"]);
}

WindowHMM::WindowHMM(int n_snp, int n_proto) :
        n_snp(n_snp), n_proto(n_proto) {

    // initialize model parameters
    start = VectorXd::Constant(n_proto, 1.0 / n_proto);
    for (int t = 0; t < n_snp - 1; t++) {
        trans.push_back(MatrixXd::Zero(n_proto, n_proto));
    }
    for (int t = 0; t < n_snp - 1; t++) {
        for (int i = 0; i < n_proto; i++) {
            for (int j = 0; j < n_proto; j++) {
                if (i == j) trans[t](i, j) = proto_bias;
                else trans[t](i, j) = (1 - proto_bias) / (n_proto - 1.0);
            }
        }
    }
    emit = MatrixXd::Zero(n_snp, n_proto);


    // allocate sufficient statistics
    suff_stat_start = VectorXd::Zero(n_proto);
    for (int i = 0; i < n_snp - 1; i++) {
        suff_stat_trans.push_back(MatrixXd::Zero(n_proto, n_proto));
    }
    for (int i = 0; i < n_snp; i++) {
        suff_stat_emit.push_back(MatrixXd::Zero(n_proto, 2));
    }

    // allocate temporary variable
    for (int t = 0; t < n_snp - 1; t++) {
        tmp_xi.push_back(MatrixXd::Zero(n_proto, n_proto));
    }
    tmp_snp_vec = VectorXd::Zero(n_snp);
    tmp_proto_vec = VectorXd::Zero(n_proto);
}

void WindowHMM::init_emit_from_X(const MatrixXi &X) {

    // X: (n_indiv, n_snp) integer matrix

    std::uniform_real_distribution<double> unif_dist(-1.0, 1.0);

}
double WindowHMM::compute_total_loglkl(const VectorXi &x) {
    // NOTE: maybe performance bottleneck
    MatrixXd log_prob(n_snp, n_proto);
    MatrixXd fwdlattice(n_snp, n_proto);
    VectorXd fwd_cond_prob(n_snp);
    compute_obs_loglkl(x, log_prob);
    forward_pass(log_prob, fwdlattice, fwd_cond_prob);
    return fwd_cond_prob.sum();
}

void WindowHMM::compute_obs_loglkl(const VectorXi &x, MatrixXd &log_prob) {
    for (int t = 0; t < n_snp; t++) {
        if (x(t) == 1) {
            for (int i = 0; i < n_proto; i++) {
                log_prob(t, i) = log(emit(t, i));
            }
        } else {
            for (int i = 0; i < n_proto; i++) {
                log_prob(t, i) = log(1 - emit(t, i));
            }
        }
    }
}

void WindowHMM::forward_pass(const MatrixXd &log_obs, MatrixXd &fwd_lattice, VectorXd &fwd_cond_prob) {
    // initialize
    int t, i, j;
    double m;

    // clean up
    for (i = 0; i < n_snp; i++) {
        fwd_cond_prob(i) = 0.0;
    }
    for (i = 0; i < fwd_lattice.rows(); i++) {
        for (j = 0; j < fwd_lattice.cols(); j++) {
            fwd_lattice(i, j) = 0.0;
        }
    }

    m = log_obs.row(0).maxCoeff();

    // initialize
    for (j = 0; j < n_proto; j++) {
        fwd_lattice(0, j) = start(j) * exp(log_obs(0, j) - m);
        fwd_cond_prob(0) += fwd_lattice(0, j);
    }

    for (j = 0; j < n_proto; j++) {
        fwd_lattice(0, j) /= fwd_cond_prob(0);
    }
    fwd_cond_prob(0) = log(fwd_cond_prob(0)) + m;
    // induction

    for (t = 0; t < n_snp - 1; t++) {
        m = log_obs.row(t + 1).maxCoeff();
        for (j = 0; j < n_proto; j++) {
            for (i = 0; i < n_proto; i++) {
                fwd_lattice(t + 1, j) += fwd_lattice(t, i) * trans[t](i, j);
            }
            fwd_lattice(t + 1, j) *= exp(log_obs(t + 1, j) - m);
            fwd_cond_prob(t + 1) += fwd_lattice(t + 1, j);
        }
        for (j = 0; j < n_proto; j++) {
            fwd_lattice(t + 1, j) /= fwd_cond_prob(t + 1);
        }
        fwd_cond_prob(t + 1) = log(fwd_cond_prob(t + 1)) + m;
    }
}

void WindowHMM::backward_pass(const MatrixXd &log_obs, MatrixXd &bwd_lattice, VectorXd &bwd_cond_prob) {
    int t, i, j;
    double m;
    VectorXf L(n_snp);
    // clean up
    for (i = 0; i < n_snp; i++) {
        bwd_cond_prob(i) = 0.0;
    }
    for (i = 0; i < bwd_lattice.rows(); i++) {
        for (j = 0; j < bwd_lattice.cols(); j++) {
            bwd_lattice(i, j) = 0.0;
        }
    }
    // initialize
    for (j = 0; j < n_proto; j++) {
        bwd_lattice(n_snp - 1, j) = 1.0;
    }

    for (t = n_snp - 2; t >= 0; t--) {
        m = log_obs.row(t + 1).maxCoeff();
        for (i = 0; i < n_proto; i++) {
            L(i) = exp(log_obs(t + 1, i) - m);
        }
        for (j = 0; j < n_proto; j++) {
            for (i = 0; i < n_proto; i++) {
                bwd_lattice(t, j) += bwd_lattice(t + 1, i) * trans[t](j, i) * L(i);
            }
            bwd_cond_prob(t + 1) += bwd_lattice(t, j);
        }

        for (j = 0; j < n_proto; j++) {
            bwd_lattice(t, j) /= bwd_cond_prob[t + 1];
        }
        bwd_cond_prob[t + 1] = log(bwd_cond_prob[t + 1]) + m;
    }
}


void WindowHMM::compute_xi(const MatrixXd &log_obs, const MatrixXd &fwd_lattice, const MatrixXd &bwd_lattice) {
    int t, i, j;
    double c, m;

    for (t = 0; t < n_snp - 1; t++) {
        c = 0.;
        m = log_obs.row(t + 1).maxCoeff();

        for (j = 0; j < n_proto; j++) {
            tmp_proto_vec[j] = exp(log_obs(t + 1, j) - m);
        }
        for (j = 0; j < n_proto; j++) {
            for (i = 0; i < n_proto; i++) {
                tmp_xi[t](i, j) = fwd_lattice(t, i) * trans[t](i, j) * tmp_proto_vec[j] * bwd_lattice(t + 1, j);
                c += tmp_xi[t](i, j);
            }
        }
        for (i = 0; i < n_proto; i++) {
            for (j = 0; j < n_proto; j++) {
                tmp_xi[t](i, j) /= c;
            }
        }
    }
}

void WindowHMM::compute_posterior(const MatrixXd &fwd_lattice, const MatrixXd &bwd_lattice, MatrixXd &posterior) {
    double s;
    for (int i = 0; i < n_snp; i++) {
        s = 0.;
        for (int j = 0; j < n_proto; j++) {
            posterior(i, j) = fwd_lattice(i, j) * bwd_lattice(i, j);
            s += posterior(i, j);
        }
        for (int j = 0; j < n_proto; j++) {
            posterior(i, j) /= s;
        }
    }
}


void WindowHMM::init_suff_stat() {
    suff_stat_start.setConstant(trans_prior * n_hap / (n_proto - 1.0));
    for (int i = 0; i < n_snp - 1; i++) {
        // TODO: define n_hap
        suff_stat_trans[i].setConstant(trans_prior * n_hap / (n_proto - 1.0));
    }
    for (int i = 0; i < n_snp; i++) {
        suff_stat_emit[i].setConstant(emit_prior * n_hap / n_proto);
    }
}

void WindowHMM::print_suff_stat() {
    cout << "start: " << endl;
    cout << suff_stat_start << endl;
    cout << "trans: " << endl;

    for (int i = 0; i < 3; i++) {
        cout << "snp " << i << ":" << endl;
        cout << suff_stat_trans[i] << endl;
    }
    cout << "emit: " << endl;
    for (int i = 0; i < 3; i++) {
        cout << "snp " << i << ":" << endl;
        cout << suff_stat_emit[i] << endl;
    }
}

void WindowHMM::print_param() {
    cout << "start: " << endl;
    cout << start << endl;
    cout << "trans: " << endl;

    for (int i = 0; i < 3; i++) {
        cout << "snp " << i << ":" << endl;
        cout << trans[i] << endl;
    }
    cout << "emit: " << endl;
    cout << emit(seq(0, 3), all) << endl;
}

void WindowHMM::accum_suff_stat(const VectorXi &x, const MatrixXd &log_obs, const MatrixXd &fwd_lattice,
                                const MatrixXd &bwd_lattice, const MatrixXd &posterior) {
    // accumulate start
    suff_stat_start += posterior.row(0);

    // accumulate transition
    compute_xi(log_obs, fwd_lattice, bwd_lattice);

    for (int t = 0; t < n_snp - 1; t++) {
        suff_stat_trans[t] += tmp_xi[t];
    }

    // accumulate emission
    for (int t = 0; t < n_snp; t++) {
        for (int i = 0; i < n_proto; i++) {
            suff_stat_emit[t](i, x(t)) += posterior(t, i);
        }
    }
}

void WindowHMM::do_mstep() {
    start = suff_stat_start / suff_stat_start.sum();

    for (int t = 0; t < n_snp - 1; t++) {
        for (int i = 0; i < n_proto; i++) {
            trans[t].row(i) = suff_stat_trans[t].row(i) / suff_stat_trans[t].row(i).sum();
        }
    }

    for (int t = 0; t < n_snp; t++) {
        for (int i = 0; i < n_proto; i++) {
            emit(t, i) = suff_stat_emit[t](i, 1) / (suff_stat_emit[t](i, 0) + suff_stat_emit[t](i, 1));
        }
    }
}

void WindowHMM::init_random_params(const MatrixXi &X) {

    // set random number generator
    std::uniform_real_distribution<double> unif_dist(-1.0, 1.0);

    // exp(uniform(-1,1))
    for (int i = 0; i < n_proto; i++){
        start(i) *= exp(unif_dist(random_engine));
    }
    start /= start.sum();
    // exp(uniform(-1,1))
    for (int t = 0; t < n_snp - 1; t++){
        for (int i = 0; i < n_proto; i++) {
            for (int j = 0; j < n_proto; j++) {
                trans[t](i, j) *= exp(unif_dist(random_engine));
            }
            trans[t].row(i) /= trans[t].row(i).sum();
        }
    }

    VectorXi minor_allele_count = X.colwise().sum();
    VectorXi major_allele_count(minor_allele_count.size());
    for (int i = 0; i < n_snp; i++) {
        major_allele_count(i) = n_hap - minor_allele_count(i);
    }

    double a,b;
    for (int t = 0; t < n_snp; t++) {
        for (int i = 0; i < n_proto; i++) {
            a = (minor_allele_count(t) + 1) * exp(unif_dist(random_engine));
            b = (major_allele_count(t) + 1) * exp(unif_dist(random_engine));
            emit(t, i) = a / (a + b);
        }
    }

}
void WindowHMM::fit(const MatrixXi &X) {
#ifdef MY_DEBUG

    cout << "WindowHMM::fit" << endl;
#endif
    assert(n_snp == X.cols());
    n_hap = X.rows();

    init_random_params(X);
    double prev_total_logprob = std::numeric_limits<double>::lowest();

    // TODO: initialize log_prob
    MatrixXd log_obs = MatrixXd::Zero(n_snp, n_proto);
    VectorXd fwd_cond_prob = VectorXd::Zero(n_snp);
    VectorXd bwd_cond_prob = VectorXd::Zero(n_snp);
    MatrixXd fwd_lattice = MatrixXd::Zero(n_snp, n_proto);
    MatrixXd bwd_lattice = MatrixXd::Zero(n_snp, n_proto);
    MatrixXd posterior = MatrixXd::Zero(n_snp, n_proto);

    for (int i_iter = 0; i_iter < max_iter; i_iter++) {

        init_suff_stat();
        float total_logprob = 0.0;
        for (int i_hap = 0; i_hap < n_hap; i_hap++) {
            compute_obs_loglkl(X.row(i_hap), log_obs);
            forward_pass(log_obs, fwd_lattice, fwd_cond_prob);
            total_logprob += fwd_cond_prob.sum();
            backward_pass(log_obs, bwd_lattice, bwd_cond_prob);
            compute_posterior(fwd_lattice, bwd_lattice, posterior);
            accum_suff_stat(X.row(i_hap), log_obs, fwd_lattice, bwd_lattice, posterior);
        }

        do_mstep();
    #ifdef MY_DEBUG
        cout << "i_iter: " << total_logprob << endl;
    #endif
        if (1 - total_logprob / prev_total_logprob < rel_tol){
            break;
        }
        prev_total_logprob = total_logprob;
    }
}


void WindowHMM::compute_alpha_beta(const VectorXi &x, MatrixXd & log_alpha, MatrixXd & log_beta){
    // NOTE: maybe performance bottleneck
    MatrixXd log_prob(n_snp, n_proto);
    MatrixXd fwd_lattice(n_snp, n_proto);
    MatrixXd bwd_lattice(n_snp, n_proto);
    VectorXd fwd_cond_prob(n_snp);
    VectorXd bwd_cond_prob(n_snp);

    log_alpha.setZero();
    log_beta.setZero();

    compute_obs_loglkl(x, log_prob);
    forward_pass(log_prob, fwd_lattice, fwd_cond_prob);
    backward_pass(log_prob, bwd_lattice, bwd_cond_prob);
    double cumsum = 0;
    for (int i = 0; i < n_snp; i++){
        cumsum += fwd_cond_prob(i);
        for (int j = 0; j < n_proto; j++){
            log_alpha(i, j) = cumsum + logl(fwd_lattice(i, j));
        }
    }
    cumsum = 0;
    for (int i = n_snp - 1; i >= 0; i--){
        for (int j = 0; j < n_proto; j++){
            log_beta(i, j) = cumsum + logl(bwd_lattice(i, j));
        }
        cumsum += bwd_cond_prob(i);
    }
}