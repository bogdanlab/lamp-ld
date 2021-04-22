#include <iostream>
#include <vector>
#include "LampLD.h"
#include "FileUtils.h"
#include "../thirdparty/cxxopts.hpp"

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

    cxxopts::Options options("LAMP-LD", "Local ancestry inference");

    options.add_options()
            ("help", "print help text")
            ("window",  "set window size", cxxopts::value<int>()->default_value("300"))
            ("proto", "set number of prototypical states", cxxopts::value<int>()->default_value("4"))
            ("pos", "path to the SNP position file", cxxopts::value<string>())
            ("admix", "path to the admixed population haplotype", cxxopts::value<string>())
            ("ref", "paths to the ancestral population haplotype", cxxopts::value<vector<string>>())
            ("out", "output prefix", cxxopts::value<string>())
            ;

    auto parser = options.parse(argc, argv);
    if (parser.count("help"))
    {
        const char *help_text =
                "Example usage\n"
                "\tlampld \\\n"
                "\t--window 300 \\\n"
                "\t--proto 6 \\\n"
                "\t--pos pos.txt \\\n"
                "\t--admix admix.hap \\\n"
                "\t--ref EUR.hap --ref AFR.hap --ref EAS.hap \\\n"
                "\t--out out.txt";
        std::cout << options.help() << std::endl;
        cout << help_text << endl;
        exit(0);
    }
    *window_size = parser["window"].as<int>();
    *n_proto = parser["proto"].as<int>();
    pos_file = parser["pos"].as<string>();
    admix_hap_file = parser["admix"].as<string>();
    ref_hap_files = parser["ref"].as<vector<string>>();
    out_file = parser["out"].as<string>();
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
         << "--admix " << admix_hap_file << endl;

    for (auto &ref: ref_hap_files) {
        cout << "--ref " << ref << endl;
    }
    cout << "--out " << out_file << "\n\n";

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



