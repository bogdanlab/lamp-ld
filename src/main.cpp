#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
#include "LampLD.h"
#include "FileUtils.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace Eigen;
using json = nlohmann::json;

//
//bool test_correct(){
//    std::ifstream file("/Users/kangchenghou/work/admix-tools/test/ancestry/data/test_correct.json");
//    json data;
//    file >> data;
//
//    WindowHMM model(
//            data["model.n_snp"],
//            data["model.n_proto"],
//            10, 1e-6, 0.0, 0.0, 0.0
//    );
//
//    model.start = from_json(data["model.start_prob"]);
//    for (int i = 0; i < data["model.trans_prob"].size(); i++) {
//        model.trans[i] = from_json(data["model.trans_prob"].at(i));
//    }
//    model.emit = from_json(data["model.emit_prob"]);
//    cout << model.start << endl;
//    cout << model.trans[0] << endl;
//    cout << model.emit << endl;
//
//    MatrixXi X = from_json(data["X"]).cast<int>();
//    cout << X.rows() << "," << X.cols() << endl;
//    model.fit(X);
//
//    cout << "After fitting:" << endl;
//    cout << model.start << endl;
//    cout << data["fit.start_prob"] << endl;
//    // check start
//    assert(model.start.isApprox(from_json(data["fit.start_prob"]), 1e-6));
//
//    // check transition
//    for (int i = 0; i < model.trans.size(); i++){
//        assert(model.trans[i].isApprox(from_json(data["fit.trans_prob"][i]), 1e-6));
//    }
//
//    // check emit
//    assert(model.emit.isApprox(from_json(data["fit.emit_prob"]), 1e-6));
//}

bool
parse_command_line(int argc, char *argv[], int *window_size, int *n_proto, string &pos_file, string &admix_hap_file,
                   vector<string> &ref_hap_files, string &out_file) {
    const char *help_text =
            "For example case\n"
            "lampld \\\n"
            "--window 300 \\\n"
            "--proto 6 \\\n"
            "--pos pos.txt \\\n"
            "--admix admix.hap \\\n"
            "--ref EUR.hap AFR.hap EAS.hap \\\n"
            "--out out.txt";
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", help_text)
                ("window", po::value<int>(window_size)->default_value(300), "set window size")
                ("proto", po::value<int>(n_proto)->default_value(4), "set number of prototypical stats")
                ("pos", po::value<string>(&pos_file)->required(), "path to the SNP position file")
                ("admix", po::value<string>(&admix_hap_file)->required(), "path to the admixed population haplotype")
                ("ref", po::value<vector<string>>(&ref_hap_files)->multitoken(),
                 "paths to the ancestral population haplotype")
                ("out", po::value<string>(&out_file)->required(), "output prefix");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

    }
    catch (exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return false;
    }
    catch (...) {
        cerr << "Unknown error!\n";
    }
    return true;
}

int main(int argc, char *argv[]) {
    int window_size;
    int n_proto;
    string pos_file;
    string admix_hap_file;
    vector<string> ref_hap_files;
    string out_file;

    bool parse_success = parse_command_line(argc, argv,
                                            &window_size,
                                            &n_proto,
                                            pos_file,
                                            admix_hap_file,
                                            ref_hap_files,
                                            out_file);
    if (!parse_success) {
        return 1;
    }
    // print received parameters
    cout << "Received options: " << endl
         << "--window " << window_size << endl
         << "--proto " << n_proto << endl
         << "--pos " << pos_file << endl
         << "--admix " << admix_hap_file << endl
         << "--ref";

    for (auto &ref: ref_hap_files) {
        cout << " " << ref;
    }
    cout << endl
         << "--out " << out_file << "\n\n";

    cout << "Reading data..." << endl;
    bool VERBOSE = true;

    // read data
    const Eigen::VectorXi pos = read_pos(pos_file);
    const Eigen::MatrixXi admix_hap = read_int_mat(admix_hap_file);

    vector<MatrixXi> vec_ref_hap;
    for (auto &ref_file: ref_hap_files) {
        vec_ref_hap.push_back(read_int_mat(ref_file));
    }

    if (VERBOSE) {
        cout << "Reading position file with " << pos.size() << " SNPs." << endl;
        for (int i = 0; i < vec_ref_hap.size(); i++) {
            cout << "Reading haplotype file from ancestry " << i + 1 << " with " << vec_ref_hap[i].rows()
                 << " haplotypes, " << vec_ref_hap[i].cols()
                 << " SNPs." << endl;
        }
        cout << "Reading haplotype file from admixed population with " << admix_hap.rows() << " haplotypes, "
             << admix_hap.cols() << " SNPs." << endl;
        cout << "Using window size " << window_size << endl;
    }


    cout << "Performing inference..." << endl;
    int n_snp = admix_hap.cols();
    int n_anc = vec_ref_hap.size();
    LampLD lamp(n_snp, n_anc, n_proto, window_size);
    lamp.set_pos(pos);

#ifdef MY_DEBUG
//    int n_window = 10;
//    // DEBUG: initialize from file
//    for (int i_window = 0; i_window < n_window; i_window++) {
//        for (int i_anc = 0; i_anc < n_anc; i_anc++) {
//            lamp.hmm_array[i_window][i_anc].init_from_file(
//                    string("/Users/kangchenghou/work/LAMPLD-v1.3/out/anc_") + to_string(i_anc) + "_" +
//                    to_string(i_window * 300) + "_init.json");
//        }
//    }
//
//    for (int i_window = 0; i_window < n_window - 1; i_window++) {
//        for (int i_anc = 0; i_anc < n_anc; i_anc++) {
//            lamp.smooth_hmm_array[i_window][i_anc].init_from_file(
//                    string("/Users/kangchenghou/work/LAMPLD-v1.3/out/anc_") + to_string(i_anc) + "_" +
//                    to_string(i_window * 300 + 150) + "_init.json");
//        }
//    }
#endif

    lamp.fit(vec_ref_hap);
    MatrixXi estimated_lanc = lamp.infer_lanc(admix_hap);

    // output
    cout << "Writing results to " << out_file << endl;
    write_int_mat(out_file, estimated_lanc);

    return 0;
}



