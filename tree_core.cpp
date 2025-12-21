#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>    
#include <thread>    

namespace py = pybind11;

// --- Cấu trúc Dữ liệu ---

struct Node {
    int feature_index;
    double threshold;
    double value;
    bool is_leaf;
    Node* left;
    Node* right;

    Node(double v) : feature_index(-1), threshold(0.0), value(v), is_leaf(true), left(nullptr), right(nullptr) {}
    Node(int fi, double th, Node* l, Node* r) : feature_index(fi), threshold(th), value(0.0), is_leaf(false), left(l), right(r) {}
    
    ~Node() {
        if (!is_leaf) {
            delete left;
            delete right;
        }
    }
};

// --- Hàm Hỗ trợ (Dùng Template để MSVC tự suy luận kiểu) ---

// Hàm tính phương sai
// Tự động nhận diện kiểu của y (không cần viết py::array_t...)
template <typename YAcc>
double calculate_variance(const YAcc& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    double mean = 0.0;
    for(int idx : indices) mean += y(idx);
    mean /= indices.size();

    double variance = 0.0;
    for(int idx : indices) {
        double diff = y(idx) - mean;
        variance += diff * diff;
    }
    return variance / indices.size();
}

// Hàm tính giảm phương sai
template <typename YAcc>
double calculate_variance_reduction(const YAcc& y, 
                                  const std::vector<int>& parent, 
                                  const std::vector<int>& left, 
                                  const std::vector<int>& right) {
    double var_parent = calculate_variance(y, parent);
    double var_left = calculate_variance(y, left);
    double var_right = calculate_variance(y, right);

    double w_left = (double)left.size() / parent.size();
    double w_right = (double)right.size() / parent.size();

    return var_parent - (w_left * var_left + w_right * var_right);
}

// Hàm dự đoán một mẫu
template <typename XAcc>
double predict_one(const XAcc& X, int row_idx, Node* node) {
    if (node->is_leaf) return node->value;
    
    if (X(row_idx, node->feature_index) <= node->threshold) {
        return predict_one(X, row_idx, node->left);
    } else {
        return predict_one(X, row_idx, node->right);
    }
}


// --- Class Cây Quyết Định ---
class DecisionTreeRegressor {
public:
    int min_samples_split;
    int max_depth;
    Node* root;

    DecisionTreeRegressor(int min_samples, int depth) 
        : min_samples_split(min_samples), max_depth(depth), root(nullptr) {}

    ~DecisionTreeRegressor() { delete root; }

    void fit(py::array_t<double> X_np, py::array_t<double> y_np) {
        // Sử dụng auto để lấy accessor
        auto X_acc = X_np.unchecked<2>();
        auto y_acc = y_np.unchecked<1>();
        int n_samples = X_acc.shape(0);

        std::vector<int> sample_indices(n_samples);
        for(int i=0; i<n_samples; ++i) sample_indices[i] = i;

        // Gọi hàm template, trình biên dịch sẽ tự điền kiểu dữ liệu
        root = build_tree(X_acc, y_acc, sample_indices, 0);
    }

    py::array_t<double> predict(py::array_t<double> X_test) {
        auto X_acc = X_test.unchecked<2>();
        int n_samples = X_acc.shape(0);
        
        py::array_t<double> results(n_samples);
        auto res_mutable = results.mutable_unchecked<1>();

        for(int i=0; i<n_samples; ++i) {
            res_mutable(i) = predict_one(X_acc, i, root);
        }
        return results;
    }

    // Hàm đệ quy xây cây (Cũng dùng Template)
    template <typename XAcc, typename YAcc>
    Node* build_tree(const XAcc& X, const YAcc& y, std::vector<int>& indices, int depth) {
        int n_samples = indices.size();
        
        double mean = 0.0;
        for(int idx : indices) mean += y(idx);
        mean /= n_samples;

        if (n_samples < min_samples_split || (max_depth > 0 && depth >= max_depth)) {
            return new Node(mean);
        }

        double best_var_red = -std::numeric_limits<double>::infinity();
        int best_feature = -1;
        double best_threshold = 0.0;
        std::vector<int> best_left_indices;
        std::vector<int> best_right_indices;

        int n_features = X.shape(1);

        for(int f = 0; f < n_features; ++f) {
            std::vector<double> values;
            values.reserve(n_samples);
            for(int idx : indices) values.push_back(X(idx, f));

            std::vector<double> unique_vals = values;
            std::sort(unique_vals.begin(), unique_vals.end());
            auto last = std::unique(unique_vals.begin(), unique_vals.end());
            unique_vals.erase(last, unique_vals.end());

            for(size_t i = 0; i < unique_vals.size() - 1; ++i) {
                double threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0;

                std::vector<int> left_indices, right_indices;
                for(int idx : indices) {
                    if (X(idx, f) <= threshold) left_indices.push_back(idx);
                    else right_indices.push_back(idx);
                }

                if (left_indices.size() < min_samples_split || right_indices.size() < min_samples_split) continue;

                double var_red = calculate_variance_reduction(y, indices, left_indices, right_indices);

                if (var_red > best_var_red) {
                    best_var_red = var_red;
                    best_feature = f;
                    best_threshold = threshold;
                    best_left_indices = left_indices;
                    best_right_indices = right_indices;
                }
            }
        }

        if (best_var_red > 0) {
            Node* left_child = build_tree(X, y, best_left_indices, depth + 1);
            Node* right_child = build_tree(X, y, best_right_indices, depth + 1);
            return new Node(best_feature, best_threshold, left_child, right_child);
        }

        return new Node(mean);
    }
};

// Hàm lấy mẫu bootstrap
std::vector<int> get_bootstrap_indices(int n_samples) {
    std::vector<int> indices;
    indices.reserve(n_samples);
    static std::random_device rd;
    static std::mt19937 generator(rd());
    std::uniform_int_distribution<> distrib(0, n_samples - 1);

    for(int i = 0; i < n_samples; ++i) {
        indices.push_back(distrib(generator));
    }
    return indices;
}


// --- Class Random Forest ---
class RandomForestRegressor {
public:
    int n_estimators;
    int min_samples_split;
    int max_depth;
    std::vector<DecisionTreeRegressor*> trees;

    RandomForestRegressor(int n_est, int min_samples, int depth) 
        : n_estimators(n_est), min_samples_split(min_samples), max_depth(depth) {}
    
    ~RandomForestRegressor() {
        for(auto tree : trees) {
            delete tree;
        }
    }

    void fit(py::array_t<double> X_np, py::array_t<double> y_np) {
        for(auto tree : trees) delete tree;
        trees.clear();

        auto X_acc = X_np.unchecked<2>();
        auto y_acc = y_np.unchecked<1>();
        int n_samples = X_acc.shape(0);

        std::vector<std::thread> threads;
        
        for (int i = 0; i < n_estimators; ++i) {
            DecisionTreeRegressor* new_tree = new DecisionTreeRegressor(min_samples_split, max_depth);
            trees.push_back(new_tree);
            
            std::vector<int> bootstrap_indices = get_bootstrap_indices(n_samples);

            threads.emplace_back([new_tree, X_acc, y_acc, bootstrap_indices]() {
                std::vector<int> indices_copy = bootstrap_indices; 
                new_tree->root = new_tree->build_tree(X_acc, y_acc, indices_copy, 0);
            });
        }
        
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    py::array_t<double> predict(py::array_t<double> X_test) {
        auto X_acc = X_test.unchecked<2>();
        int n_samples = X_acc.shape(0);
        
        py::array_t<double> results(n_samples);
        auto res_mutable = results.mutable_unchecked<1>();

        for(int i=0; i<n_samples; ++i) {
            double sum_pred = 0.0;
            for(auto tree : trees) {
                sum_pred += predict_one(X_acc, i, tree->root);
            }
            res_mutable(i) = sum_pred / trees.size();
        }
        return results;
    }
};

PYBIND11_MODULE(tree_core, m) {
    m.doc() = "Module C++ Decision Tree & Random Forest";
    
    py::class_<DecisionTreeRegressor>(m, "DecisionTreeRegressor")
        .def(py::init<int, int>(), py::arg("min_samples_split") = 2, py::arg("max_depth") = -1)
        .def("fit", &DecisionTreeRegressor::fit)
        .def("predict", &DecisionTreeRegressor::predict);
    
    py::class_<RandomForestRegressor>(m, "RandomForestRegressor")
        .def(py::init<int, int, int>(), py::arg("n_estimators") = 10, py::arg("min_samples_split") = 2, py::arg("max_depth") = -1)
        .def("fit", &RandomForestRegressor::fit)
        .def("predict", &RandomForestRegressor::predict);
}