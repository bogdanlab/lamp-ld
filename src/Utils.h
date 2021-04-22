#ifndef UTILS_H
#define UTILS_H

#include "../thirdparty/json.hpp"
#include "../thirdparty/Eigen/Core"
#include <cmath>
#include <iostream>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void print_progress(double percentage);

using namespace std;
using namespace Eigen;
MatrixXd from_json(const nlohmann::json &jsonObject);
VectorXi linspaced_int(int n, int start, int stop);
double logsumexp(const VectorXd & x);
void decode_viterbi(const VectorXd &log_start, const vector<MatrixXd> &log_trans, const MatrixXd &log_obs,
                    VectorXi &decoded);
#endif
