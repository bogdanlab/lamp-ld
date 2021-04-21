//
// Created by Hou, Kangcheng on 4/8/21.
//

#ifndef FILEUTILS_H
#define FILEUTILS_H
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
#include <limits>
#include "json.hpp"

using namespace std;
using namespace Eigen;

MatrixXi read_int_mat(const std::string &path);
VectorXi read_pos(const std::string &path);
void write_int_mat(const std::string &path, const MatrixXi& mat);

#endif //FILEUTILS_H
