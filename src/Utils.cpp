#include "Utils.h"


void print_progress(double percentage){
    int val =  ceil(percentage * 100);
    int lpad = ceil(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}
double logsumexp(const VectorXd & x) {
    double x_max = x.maxCoeff();

    double acc = 0;
    for (int i = 0; i < x.size(); i++){
        acc += expl(x[i] - x_max);
    }
    return logl(acc) + x_max;
}

MatrixXd from_json(const nlohmann::json &jsonObject) {
    nlohmann::json jsonArray;
    if (jsonObject.is_array()) {
        if (jsonObject.empty()) {
            return MatrixXd();
        }
        jsonArray = jsonObject;
    } else if (jsonObject.is_number()) {
        jsonArray.push_back(jsonObject);
    } else {
        throw nlohmann::detail::type_error::create(0, "");
    }

    nlohmann::json jsonArrayOfArrays;
    if (jsonArray.front().is_array())  // provided matrix
    {
        jsonArrayOfArrays = jsonArray;
    } else  // provided vector
    {
        jsonArrayOfArrays.push_back(jsonArray);
    }

    const unsigned int providedRows = jsonArrayOfArrays.size();
    const unsigned int providedCols = jsonArrayOfArrays.front().size();

    MatrixXd matrix(providedRows, providedCols);
    for (unsigned int r = 0; r < providedRows; ++r) {
        for (unsigned int c = 0; c < providedCols; ++c) {
            matrix(r, c) = jsonArrayOfArrays.at(r).at(c);
        }
    }
    if (providedRows == 1) {
        matrix.transposeInPlace();
    }
    return matrix;
}

VectorXi linspaced_int(int n, int start, int stop) {
    VectorXf index = VectorXf::LinSpaced(n, start, stop);
    VectorXi int_index(index.size());
    for (int i = 0; i < index.size(); i++) {
        int_index(i) = round(index(i));
    }
    return int_index;
}


void decode_viterbi(const VectorXd &log_start, const vector<MatrixXd> &log_trans, const MatrixXd &log_obs,
                    VectorXi &decoded) {
    int n_seq = log_obs.rows();
    int n_state = log_start.size();
    MatrixXd viterbi_lattice = MatrixXd::Zero(n_seq, n_state);
    VectorXd work_buffer = VectorXd::Zero(n_state);
    // initialization
    for (int i = 0; i < n_state; i++) {
        viterbi_lattice(0, i) = log_start(i) + log_obs(0, i);
    }

    // induction
    for (int t = 1; t < n_seq; t++) {
        for (int i = 0; i < n_state; i++) {
            for (int j = 0; j < n_state; j++) {
                work_buffer(j) = log_trans[t - 1](j, i) + viterbi_lattice(t - 1, j);
            }
            viterbi_lattice(t, i) = work_buffer.maxCoeff() + log_obs(t, i);
        }
    }
    MatrixXf::Index where_from;

    // Observation traceback
    viterbi_lattice.row(n_seq - 1).maxCoeff(&where_from);
    decoded(n_seq - 1) = where_from;

    for (int t = n_seq - 2; t >= 0; t--) {
        for (int i = 0; i < n_state; i++) {
            work_buffer(i) = viterbi_lattice(t, i) + log_trans[t](i, where_from);
        }
        work_buffer.maxCoeff(&where_from);
        decoded(t) = where_from;
    }
}



