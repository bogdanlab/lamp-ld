#ifndef LAMPLD_H
#define LAMPLD_H

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
#include <limits>
#include "WindowHMM.h"

class LampLD {
public:
    LampLD(int n_snp, int n_anc, int n_proto, int window_size);

    void fit(std::vector<Eigen::MatrixXi> ref_list);

    void set_pos(const VectorXi pos);

    MatrixXi infer_lanc(const MatrixXi &admix_hap);
    void smooth_lanc(const MatrixXi &admix_hap, MatrixXi &lanc);
    vector<vector<WindowHMM>> hmm_array;
    vector<vector<WindowHMM>> smooth_hmm_array;

private:
    Eigen::VectorXi snp_pos;
    const int n_snp;
    const int n_anc;
    const bool smooth = true;
    const int n_proto;
    const double recomb_rate = 1e-8;
    const bool verbose = true;
    Eigen::VectorXi snp_index;
    Eigen::VectorXi smooth_snp_index;



};


#endif //LAMPLD_H
