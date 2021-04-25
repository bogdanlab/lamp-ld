#include "LampLD.h"
#include "Utils.h"


MatrixXi LampLD::infer_lanc(const MatrixXi &sample_hap) {
    assert(sample_hap.cols() == n_snp);
    int n_hap = sample_hap.rows();
    int n_window = snp_index.size() - 1;

    MatrixXi result_lanc(n_hap, n_snp);
    MatrixXd log_obs(n_window, n_anc);
    vector<MatrixXd> vec_log_trans;

    for (int i_window = 0; i_window < n_window - 1; i_window++) {
        MatrixXd log_trans(n_anc, n_anc);
        int snp_pos_gap = snp_pos[snp_index(i_window + 1)] - snp_pos[snp_index(i_window + 1) - 1];

        double trans_base = recomb_rate * snp_pos_gap;
        for (int i_anc = 0; i_anc < n_anc; i_anc++) {
            for (int j_anc = 0; j_anc < n_anc; j_anc++) {
                log_trans(i_anc, j_anc) = logl(trans_base) * (i_anc != j_anc);
            }
        }
        vec_log_trans.push_back(log_trans);
    }

    VectorXi decoded(n_window);
    for (int i_hap = 0; i_hap < n_hap; i_hap++) {
        VectorXi this_hap = sample_hap.row(i_hap);
        for (int i_window = 0; i_window < n_window; i_window++) {
            int window_start = snp_index[i_window];
            int window_stop = snp_index[i_window + 1];
            for (int i_anc = 0; i_anc < n_anc; i_anc++) {
                log_obs(i_window, i_anc) = hmm_array[i_window][i_anc].compute_total_loglkl(
                        this_hap(seq(window_start, window_stop - 1)));
            }
        }

        decode_viterbi(VectorXd::Zero(n_anc), vec_log_trans, log_obs, decoded);

        for (int i_window = 0; i_window < n_window; i_window++) {
            int window_start = snp_index[i_window];
            int window_stop = snp_index[i_window + 1];
            for (int i_snp = window_start; i_snp < window_stop; i_snp++) {
                result_lanc(i_hap, i_snp) = decoded[i_window];
            }
        }
    }

    if (smooth) {
        smooth_lanc(sample_hap, result_lanc);
    }
    return result_lanc;
}


void LampLD::smooth_lanc(const MatrixXi &admix_hap, MatrixXi &lanc) {

    assert(admix_hap.cols() == n_snp);
    int n_hap = admix_hap.rows();
    int n_window = snp_index.size() - 1;
    for (int i_window = 0; i_window < n_window - 1; i_window++) {

        int window_start = smooth_snp_index[i_window];
        int window_stop = smooth_snp_index[i_window + 1];
        int window_size = window_stop - window_start;
        MatrixXd fwd_array(window_size, 2);
        MatrixXd bwd_array(window_size, 2);
        MatrixXd log_alpha(window_size, n_proto);
        MatrixXd log_beta(window_size, n_proto);
        int bp = snp_index[i_window + 1];
        for (int i_hap = 0; i_hap < n_hap; i_hap++) {
            vector<int> bp_anc = {lanc(i_hap, bp - 1), lanc(i_hap, bp)};
            if (bp_anc[0] != bp_anc[1]) {
                // adjusting ancestry change point
                fwd_array.setZero();
                bwd_array.setZero();

                VectorXi hap_chunk = admix_hap.row(i_hap)(seq(window_start, window_stop - 1));
                // fill in forward / backward
                for (int i_anc = 0; i_anc < 2; i_anc++) {
                    WindowHMM &hmm = smooth_hmm_array[i_window][bp_anc[i_anc]];
                    hmm.compute_alpha_beta(hap_chunk, log_alpha, log_beta);

                    for (int i_snp = 0; i_snp < hap_chunk.size(); i_snp++) {
                        fwd_array(i_snp, i_anc) = logsumexp(log_alpha.row(i_snp));
                        bwd_array(i_snp, i_anc) = logsumexp(log_beta.row(i_snp));
                    }
                }

                // do adjusting
                double best_prob = std::numeric_limits<double>::lowest();
                int i_best = -1;
                for (int i_snp = window_start + 10; i_snp < window_stop - 10; i_snp++) {
                    double rprob = log(recomb_rate * (snp_pos[i_snp] - snp_pos[i_snp - 1]));
                    if (fwd_array(i_snp - window_start, 0) + bwd_array(i_snp - window_start, 1) + rprob > best_prob) {
                        i_best = i_snp;
                        best_prob = fwd_array(i_snp - window_start, 0) + bwd_array(i_snp - window_start, 1) + rprob;
                    }
                }
                if (i_best < bp) {
                    for (int i_snp = i_best; i_snp < bp; i_snp++) {
                        lanc(i_hap, i_snp) = bp_anc[1];
                    }
                } else {
                    for (int i_snp = bp; i_snp < i_best; i_snp++) {
                        lanc(i_hap, i_snp) = bp_anc[0];
                    }
                }
            }
        }
    }
}

LampLD::LampLD(int n_snp, int n_anc, int n_proto, int window_size) : n_snp(n_snp), n_anc(n_anc), n_proto(n_proto) {


    int n_window = (int)ceil(n_snp / window_size);
    // initialize for HMM array
    snp_index = linspaced_int(n_window + 1, 0, n_snp);
    for (int i_window = 0; i_window < n_window; i_window++) {
        hmm_array.push_back(vector<WindowHMM>());
        for (int i_anc = 0; i_anc < n_anc; i_anc++) {
            hmm_array[i_window].push_back(WindowHMM(snp_index(i_window + 1) - snp_index(i_window), n_proto));
        }
    }

    // initialize HMM array for smoothing procedure
    if (smooth) {
        smooth_snp_index = Eigen::VectorXi(n_window);
        for (int i = 0; i < n_window; i++) {
            smooth_snp_index(i) = (snp_index(i) + snp_index(i + 1)) / 2;
        }

        for (int i_window = 0; i_window < n_window - 1; i_window++) {
            smooth_hmm_array.push_back(vector<WindowHMM>());
            for (int i_anc = 0; i_anc < n_anc; i_anc++) {
                smooth_hmm_array[i_window].push_back(
                        WindowHMM(smooth_snp_index(i_window + 1) - smooth_snp_index(i_window), n_proto));
            }
        }
    }
}

void LampLD::set_pos(const VectorXi pos) {
    snp_pos = pos;
}

void LampLD::fit(std::vector<Eigen::MatrixXi> ref_list) {
    int n_window = snp_index.size() - 1;
    for (auto &ref : ref_list) {
        assert(ref.cols() == n_snp);
    }
    assert(ref_list.size() == n_anc);
    int start, stop;
    cout << "Fitting HMMs..." << endl;
    for (int i_window = 0; i_window < n_window; i_window++) {
        print_progress((i_window + 1) / (double)n_window);
        start = snp_index(i_window);
        stop = snp_index(i_window + 1);
        for (int i_anc = 0; i_anc < n_anc; i_anc++) {
            WindowHMM &hmm = hmm_array[i_window][i_anc];
            const MatrixXi &ref_chunk(ref_list[i_anc](Eigen::all, Eigen::seq(start, stop - 1)));

#ifdef MY_DEBUG
            hmm.init_emit_from_X(ref_chunk);
#else

#endif
            hmm.fit(ref_chunk);
        }
    }
    cout << endl;
    if (smooth) {
        cout << "Fitting smoothing HMMs..." << endl;
        for (int i_window = 0; i_window < n_window - 1; i_window++) {
            print_progress((i_window + 1) / (double)(n_window - 1));
            start = smooth_snp_index(i_window);
            stop = smooth_snp_index(i_window + 1);
            for (int i_anc = 0; i_anc < n_anc; i_anc++) {
                WindowHMM &hmm = smooth_hmm_array[i_window][i_anc];
                const MatrixXi &ref_chunk(ref_list[i_anc](Eigen::all, Eigen::seq(start, stop - 1)));
#ifdef MY_DEBUG
                hmm.init_emit_from_X(ref_chunk);
#else

#endif
                hmm.fit(ref_chunk);
            }
        }
        cout << endl;
    }
}