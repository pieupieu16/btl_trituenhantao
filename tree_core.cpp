// File: tree_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

namespace py = pybind11;

// Cấu trúc một Node trong cây
struct Node {
    int feature_index;
    double threshold;
    double value;
    bool is_leaf;
    Node* left;
    Node* right;

    Node(double v) : feature_index(-1), threshold(0.0), value(v), is_leaf(true), left(nullptr), right(nullptr) {}
    Node(int fi, double th, Node* l, Node* r) : feature_index(fi), threshold(th), value(0.0), is_leaf(false), left(l), right(r) {}
};

// Class Cây Quyết Định (C++)
class DecisionTreeRegressor {
public:
    int min_samples_split;
    int max_depth;
    Node* root;

    DecisionTreeRegressor(int min_samples, int depth) : min_samples_split(min_samples), max_depth(depth), root(nullptr) {}

    // Hàm Fit: Nhận dữ liệu từ Python (Numpy Array)
    void fit(py::array_t<double> X_np, py::array_t<double> y_np) {
        auto X_acc = X_np.unchecked<2>(); // Truy cập nhanh không kiểm tra biên
        auto y_acc = y_np.unchecked<1>();

        int n_samples = X_acc.shape(0);
        int n_features = X_acc.shape(1);

        // Chuyển dữ liệu sang vector C++ để dễ xử lý (hoặc dùng con trỏ raw pointer để nhanh hơn nữa)
        // Ở đây dùng vector index để tham chiếu dòng
        std::vector<int> sample_indices(n_samples);
        for(int i=0; i<n_samples; ++i) sample_indices[i] = i;

        // Xây cây
        root = build_tree(X_acc, y_acc, sample_indices, 0);
    }

    // Hàm Predict: Nhận X_test từ Python và trả về dự đoán
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

private:
    // Đệ quy xây cây
    template <typename XType, typename YType>
    Node* build_tree(XType& X, YType& y, std::vector<int>& indices, int depth) {
        int n_samples = indices.size();
        
        // Tính giá trị trung bình (Prediction Value)
        double mean = 0.0;
        for(int idx : indices) mean += y(idx);
        mean /= n_samples;

        // Điều kiện dừng
        if (n_samples < min_samples_split || (max_depth > 0 && depth >= max_depth)) {
            return new Node(mean);
        }

        double best_var_red = -1.0;
        int best_feature = -1;
        double best_threshold = 0.0;
        std::vector<int> best_left_indices;
        std::vector<int> best_right_indices;

        int n_features = X.shape(1);

        // --- PHẦN TỐI ƯU TỐC ĐỘ (C++ LOOP) ---
        // Duyệt qua từng feature để tìm điểm cắt tốt nhất
        for(int f = 0; f < n_features; ++f) {
            // Lấy tất cả giá trị của feature f trong node hiện tại
            std::vector<double> values;
            values.reserve(n_samples);
            for(int idx : indices) values.push_back(X(idx, f));

            // Tìm các ngưỡng unique (để cắt)
            std::vector<double> unique_vals = values;
            std::sort(unique_vals.begin(), unique_vals.end());
            auto last = std::unique(unique_vals.begin(), unique_vals.end());
            unique_vals.erase(last, unique_vals.end());

            // Loop qua các ngưỡng cắt
            for(size_t i = 0; i < unique_vals.size() - 1; ++i) {
                // Threshold là trung bình cộng giữa 2 điểm liên tiếp (chuẩn thuật toán)
                double threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0;

                std::vector<int> left_indices, right_indices;
                // Phân chia dữ liệu
                for(int idx : indices) {
                    if (X(idx, f) <= threshold) left_indices.push_back(idx);
                    else right_indices.push_back(idx);
                }

                if (left_indices.empty() || right_indices.empty()) continue;

                // Tính Variance Reduction
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

    template <typename YType>
    double calculate_variance(YType& y, const std::vector<int>& indices) {
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

    template <typename YType>
    double calculate_variance_reduction(YType& y, const std::vector<int>& parent, 
                                      const std::vector<int>& left, const std::vector<int>& right) {
        double var_parent = calculate_variance(y, parent);
        double var_left = calculate_variance(y, left);
        double var_right = calculate_variance(y, right);

        double w_left = (double)left.size() / parent.size();
        double w_right = (double)right.size() / parent.size();

        return var_parent - (w_left * var_left + w_right * var_right);
    }

    template <typename XType>
    double predict_one(XType& X, int row_idx, Node* node) {
        if (node->is_leaf) return node->value;
        
        if (X(row_idx, node->feature_index) <= node->threshold) {
            return predict_one(X, row_idx, node->left);
        } else {
            return predict_one(X, row_idx, node->right);
        }
    }
};

// Binding code: Kết nối C++ với Python
PYBIND11_MODULE(tree_core, m) {
    py::class_<DecisionTreeRegressor>(m, "DecisionTreeRegressor")
        .def(py::init<int, int>())
        .def("fit", &DecisionTreeRegressor::fit)
        .def("predict", &DecisionTreeRegressor::predict);
}