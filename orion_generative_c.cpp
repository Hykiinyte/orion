#include <iostream> // so
#include <stdlib.h> // many
#include <vector> // stuff
#include <string> // and
#include <map> // things
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cstdint>
#include <cstdio>

#ifdef USE_CUDA // use cuda variable
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace fs = std::filesystem;

std::string priority = "gpu"; // use gpu with CUDA if available, otherwise fallback to CPU
bool use_gpu = true;
bool cuda_supported = true;

#ifdef USE_CUDA // use cuda variable
cublasHandle_t cublas_handle = nullptr;

#define CUDA_CHECK(call) do { // cuda checks or return false
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; 
        return false; 
    } 
} while (0)

#define CUBLAS_CHECK(call) do { 
    cublasStatus_t status = call; 
    if (status != CUBLAS_STATUS_SUCCESS) { 
        std::cerr << "cuBLAS error: " << status << "\n"; 
        return false; 
    } 
} while (0)

bool init_cuda() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA device not found.\n";
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "Unable to set CUDA device.\n";
        return false;
    }
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Unable to create cuBLAS handle.\n";
        return false;
    }
    return true;
}

Vector gpu_matrix_vector_mul(const Matrix& m, const Vector& v) { //matrices and vectors on gpu
    Vector result(m.rows);
    float alpha = 1.0f;
    float beta = 0.0f;
    float* dA = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(float) * m.rows * m.cols));
    CUDA_CHECK(cudaMalloc((void**)&dx, sizeof(float) * v.size));
    CUDA_CHECK(cudaMalloc((void**)&dy, sizeof(float) * result.size));
    CUDA_CHECK(cudaMemcpy(dA, m.data.data(), sizeof(float) * m.rows * m.cols, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, v.data.data(), sizeof(float) * v.size, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, m.rows, m.cols, &alpha, dA, m.cols, dx, 1, &beta, dy, 1));
    CUDA_CHECK(cudaMemcpy(result.data.data(), dy, sizeof(float) * result.size, cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    return result;
}

Vector gpu_matrix_transpose_vector_mul(const Matrix& m, const Vector& v) {
    Vector result(m.cols);
    float alpha = 1.0f;
    float beta = 0.0f;
    float* dA = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(float) * m.rows * m.cols));
    CUDA_CHECK(cudaMalloc((void**)&dx, sizeof(float) * v.size));
    CUDA_CHECK(cudaMalloc((void**)&dy, sizeof(float) * result.size));
    CUDA_CHECK(cudaMemcpy(dA, m.data.data(), sizeof(float) * m.rows * m.cols, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, v.data.data(), sizeof(float) * v.size, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, m.cols, m.rows, &alpha, dA, m.cols, dx, 1, &beta, dy, 1));
    CUDA_CHECK(cudaMemcpy(result.data.data(), dy, sizeof(float) * result.size, cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    return result;
}
#endif

std::string run_command_capture(const std::string& cmd) {
    std::string output;
    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) return output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    _pclose(pipe);
    return output;
}

std::string quote_powershell_path(const std::string& path) {
    std::string escaped = path;
    std::replace(escaped.begin(), escaped.end(), '\'', '"');
    return escaped;
}

std::string extract_text_from_docx_xml(const std::string& xml) {
    std::string result;
    size_t pos = 0;
    while (true) {
        size_t start = xml.find("<w:t", pos);
        if (start == std::string::npos) break;
        size_t gt = xml.find('>', start);
        if (gt == std::string::npos) break;
        size_t end = xml.find("</w:t>", gt);
        if (end == std::string::npos) break;
        result += xml.substr(gt + 1, end - gt - 1);
        result += ' ';
        pos = end + 6;
    }
    return result;
}

std::string read_docx_text(const std::string& path) { // powershell to extract text from docx by unzipping and reading document.xml
    std::string temp_dir = (fs::temp_directory_path() / ("orion_docx_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()))).string();
    fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);
    std::string quoted_path = quote_powershell_path(path);
    std::string quoted_temp = quote_powershell_path(temp_dir);
    std::string command = "powershell -NoProfile -Command \"Add-Type -AssemblyName System.IO.Compression.FileSystem; ";
    command += "[IO.Compression.ZipFile]::ExtractToDirectory('" + quoted_path + "','" + quoted_temp + "'); ";
    command += "Get-Content '" + quoted_temp + "/word/document.xml' | Out-String\"";
    std::string xml = run_command_capture(command);
    fs::remove_all(temp_dir);
    if (xml.empty()) {
        std::cerr << "Failed to extract text from DOCX: " << path << "\n";
        return std::string();
    }
    return extract_text_from_docx_xml(xml);
}

std::string read_pdf_text(const std::string& path) { // python subprocess to read pdf bc pdf parsing in c++ is a nightmare and pdfplumber is great
    std::string command = "python extract_pdf.py \"" + path + "\"";
    std::string output = run_command_capture(command);
    if (output.empty()) {
        std::cerr << "Failed to extract text from PDF; ensure Python and pdfplumber are installed.\n";
    }
    return output;
}

struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r = 0, int c = 0) : rows(r), cols(c), data(r * c, 0.0f) {}
    float& operator()(int i, int j) { return data[i * cols + j]; }
    const float& operator()(int i, int j) const { return data[i * cols + j]; }
};

struct Vector {
    int size;
    std::vector<float> data;

    Vector(int n = 0) : size(n), data(n, 0.0f) {}
    float& operator[](int i) { return data[i]; }
    const float& operator[](int i) const { return data[i]; }
};

Matrix matrix_random(int rows, int cols, float scale = 0.01f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    Matrix m(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        m.data[i] = dist(gen);
    }
    return m;
}

Matrix matrix_zero(int rows, int cols) {
    return Matrix(rows, cols);
}

Vector vector_zero(int n) {
    return Vector(n);
}

Vector matrix_vector_mul(const Matrix& m, const Vector& v) {
#ifdef USE_CUDA
    if (use_gpu && cuda_supported) {
        return gpu_matrix_vector_mul(m, v);
    }
#endif
    Vector result(m.rows);
    for (int i = 0; i < m.rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < m.cols; j++) {
            sum += m(i, j) * v[j];
        }
        result[i] = sum;
    }
    return result;
}

Vector matrix_transpose_vector_mul(const Matrix& m, const Vector& v) {
#ifdef USE_CUDA
    if (use_gpu && cuda_supported) {
        return gpu_matrix_transpose_vector_mul(m, v);
    }
#endif
    Vector result(m.cols);
    for (int j = 0; j < m.cols; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m.rows; i++) {
            sum += m(i, j) * v[i];
        }
        result[j] = sum;
    }
    return result;
}

Matrix outer_product(const Vector& a, const Vector& b) {
    Matrix m(a.size, b.size);
    for (int i = 0; i < a.size; i++) {
        for (int j = 0; j < b.size; j++) {
            m(i, j) = a[i] * b[j];
        }
    }
    return m;
}

Vector add(const Vector& a, const Vector& b) {
    Vector result(a.size);
    for (int i = 0; i < a.size; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vector add_scalar(const Vector& a, float scalar) {
    Vector result(a.size);
    for (int i = 0; i < a.size; i++) {
        result[i] = a[i] + scalar;
    }
    return result;
}

Vector tanh_vec(const Vector& v) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = std::tanh(v[i]);
    }
    return result;
}

Vector exp_vec(const Vector& v) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = std::exp(v[i]);
    }
    return result;
}

Vector clip_vec(const Vector& v, float low, float high) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = std::min(std::max(v[i], low), high);
    }
    return result;
}

float sum_vec(const Vector& v) {
    float sum = 0.0f;
    for (int i = 0; i < v.size; i++) {
        sum += v[i];
    }
    return sum;
}

Vector softmax(const Vector& v) {
    float max_val = v.data[0];
    for (int i = 1; i < v.size; i++) {
        max_val = std::max(max_val, v[i]);
    }
    Vector shifted(v.size);
    for (int i = 0; i < v.size; i++) {
        shifted[i] = v[i] - max_val;
    }
    Vector exp_scores = exp_vec(shifted);
    float sum_scores = sum_vec(exp_scores);
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = exp_scores[i] / (sum_scores + 1e-12f);
    }
    return result;
}

Vector square_vec(const Vector& v) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = v[i] * v[i];
    }
    return result;
}

Vector sqrt_vec(const Vector& v) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = std::sqrt(v[i]);
    }
    return result;
}

Vector clip_vec(const Vector& v, float threshold) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = std::min(std::max(v[i], -threshold), threshold);
    }
    return result;
}

Matrix add(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows * a.cols; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

Matrix add_scalar(const Matrix& a, float scalar) {
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows * a.cols; i++) {
        result.data[i] = a.data[i] + scalar;
    }
    return result;
}

Matrix clip_mat(const Matrix& m, float threshold) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; i++) {
        result.data[i] = std::min(std::max(m.data[i], -threshold), threshold);
    }
    return result;
}

Matrix square_mat(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; i++) {
        result.data[i] = m.data[i] * m.data[i];
    }
    return result;
}

Matrix sqrt_mat(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; i++) {
        result.data[i] = std::sqrt(m.data[i]);
    }
    return result;
}

void add_inplace(Matrix& a, const Matrix& b) {
    for (int i = 0; i < a.rows * a.cols; i++) {
        a.data[i] += b.data[i];
    }
}

void add_inplace(Vector& a, const Vector& b) {
    for (int i = 0; i < a.size; i++) {
        a[i] += b[i];
    }
}

void add_inplace(Vector& a, float scalar) {
    for (int i = 0; i < a.size; i++) {
        a[i] += scalar;
    }
}

Matrix scale_mat(const Matrix& m, float scalar) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; i++) {
        result.data[i] = m.data[i] * scalar;
    }
    return result;
}

Vector scale_vec(const Vector& v, float scalar) {
    Vector result(v.size);
    for (int i = 0; i < v.size; i++) {
        result[i] = v[i] * scalar;
    }
    return result;
}

Vector divide_vec(const Vector& a, const Vector& b) {
    Vector result(a.size);
    for (int i = 0; i < a.size; i++) {
        result[i] = a[i] / (b[i] + 1e-12f);
    }
    return result;
}

Vector divide_vec(const Vector& a, float scalar) {
    Vector result(a.size);
    for (int i = 0; i < a.size; i++) {
        result[i] = a[i] / scalar;
    }
    return result;
}

Matrix add_inplace(Matrix& a, float scalar) {
    for (int i = 0; i < a.rows * a.cols; i++) {
        a.data[i] += scalar;
    }
    return a;
}

// Hyperparameters
const int SAVE_ITER = 10; // save every N iterations
const int SAMPLE_ITER = 1000; // sample from model every N iterations
const int PROGRESS_ITER = 10; // print progress every N iterations
const int TOKENID_COUNT = 256; // number of unique byte tokens (0-255)
const int HIDDEN_SIZE = 5000; // number of neurons in hidden layer
const int SEQ_LENGTH = 50; // number of steps to unroll the RNN for during training
const float LEARNING_RATE = 1e-4f; // learning rate for parameter updates

// establish parameters and their momentum buffers (for Adagrad)
Matrix Wxh, Whh, Why;
Vector bh, by;
Matrix mWxh, mWhh, mWhy;
Vector mbh, mby;

class BPETokenizer { // simple byte-level BPE tokenizer implementation
public:
    std::map<int, std::vector<uint8_t>> vocab;
    std::map<std::pair<int, int>, int> merges;
    std::map<std::vector<uint8_t>, int> inverse_vocab;

    std::map<std::pair<int, int>, int> get_stats(const std::vector<int>& ids) {
        std::map<std::pair<int, int>, int> counts;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            counts[{ids[i], ids[i + 1]}]++;
        }
        return counts;
    }

    std::vector<int> merge_ids(const std::vector<int>& ids,
                               const std::pair<int, int>& pair,
                               int idx) {
        std::vector<int> newids;
        newids.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == pair.first && ids[i + 1] == pair.second) {
                newids.push_back(idx);
                i++;
            } else {
                newids.push_back(ids[i]);
            }
        }
        return newids;
    }

    std::vector<int> train(const std::string& text, int vocab_size = 5000, bool verbose = true) {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<int>(c));
        }
        int num_merges = vocab_size - TOKENID_COUNT;

        merges.clear();
        vocab.clear();
        for (int idx = 0; idx < TOKENID_COUNT; idx++) {
            vocab[idx] = {static_cast<uint8_t>(idx)};
        }

        if (verbose) {
            std::cout << "Training BPE... Goal: " << vocab_size << " tokens\n";
        }

        for (int i = 0; i < num_merges; i++) {
            auto stats = get_stats(ids);
            if (stats.empty()) {
                break;
            }

            auto max_pair = stats.begin();
            for (auto it = stats.begin(); it != stats.end(); ++it) {
                if (it->second > max_pair->second) {
                    max_pair = it;
                }
            }
            auto pair = max_pair->first;
            int count = max_pair->second;
            int new_idx = TOKENID_COUNT + i;
            merges[pair] = new_idx;

            auto v1 = vocab[pair.first];
            auto v2 = vocab[pair.second];
            v1.insert(v1.end(), v2.begin(), v2.end());
            vocab[new_idx] = std::move(v1);

            ids = merge_ids(ids, pair, new_idx);

            if (verbose && ((i + 1) % 100 == 0 || i + 1 == num_merges)) {
                std::cout << "Merge " << (i + 1) << "/" << num_merges
                          << ": byte pair (Found " << count << " times)\n";
            }
        }

        inverse_vocab.clear();
        for (auto& kv : vocab) {
            inverse_vocab[kv.second] = kv.first;
        }
        return ids;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<int>(c));
        }

        while (ids.size() >= 2) {
            auto stats = get_stats(ids);
            std::pair<int, int> best_pair = {-1, -1};
            int best_idx = INT_MAX;
            for (auto& p : stats) {
                auto it = merges.find(p.first);
                if (it != merges.end() && it->second < best_idx) {
                    best_idx = it->second;
                    best_pair = p.first;
                }
            }
            if (best_pair.first == -1) {
                break;
            }
            ids = merge_ids(ids, best_pair, best_idx);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::vector<uint8_t> tokens;
        for (int id : ids) {
            auto it = vocab.find(id);
            if (it != vocab.end()) {
                tokens.insert(tokens.end(), it->second.begin(), it->second.end());
            }
        }
        return std::string(tokens.begin(), tokens.end());
    }
};

BPETokenizer tokenizer;

std::string load_training_data() { // load text from .txt, .docx, and .pdf files in training_data/text directory
    std::string full_text_data;
    std::string base_path = "training_data/text/";
    if (!fs::exists(base_path)) {
        return full_text_data;
    }

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".txt") {
            std::ifstream file(entry.path(), std::ios::binary);
            if (!file) {
                std::cout << "Error opening " << entry.path().filename() << "\n";
                continue;
            }
            std::string chunk((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            full_text_data += chunk;
            full_text_data += '\n';
            std::cout << "Loaded: " << entry.path().filename() << "\n";
        } else if (ext == ".docx") {
            std::string chunk = read_docx_text(entry.path().string());
            if (!chunk.empty()) {
                full_text_data += chunk;
                full_text_data += '\n';
                std::cout << "Loaded DOCX: " << entry.path().filename() << "\n";
            }
        } else if (ext == ".pdf") {
            std::string chunk = read_pdf_text(entry.path().string());
            if (!chunk.empty()) {
                full_text_data += chunk;
                full_text_data += '\n';
                std::cout << "Loaded PDF: " << entry.path().filename() << "\n";
            }
        }
    }
    return full_text_data;
}

void save_checkpoint(const std::string& filename = "my_rnn_model.bin") {
    std::cout << "Saving model to " << filename << "...\n";
    std::ofstream file(filename, std::ios::binary);
    int h_size = Wxh.rows;
    int v_size = Wxh.cols;
    file.write(reinterpret_cast<const char*>(&h_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&v_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(Wxh.data.data()), h_size * v_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(Whh.data.data()), h_size * h_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(Why.data.data()), v_size * h_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(bh.data.data()), h_size * sizeof(float));
    file.write(reinterpret_cast<const char*>(by.data.data()), v_size * sizeof(float));
}

bool load_checkpoint(const std::string& filename = "my_rnn_model.bin") {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "No checkpoint found at " << filename << ". Starting from scratch.\n";
        return false;
    }
    int h_size, v_size;
    file.read(reinterpret_cast<char*>(&h_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&v_size), sizeof(int));
    Wxh = Matrix(h_size, v_size);
    Whh = Matrix(h_size, h_size);
    Why = Matrix(v_size, h_size);
    bh = Vector(h_size);
    by = Vector(v_size);
    file.read(reinterpret_cast<char*>(Wxh.data.data()), h_size * v_size * sizeof(float));
    file.read(reinterpret_cast<char*>(Whh.data.data()), h_size * h_size * sizeof(float));
    file.read(reinterpret_cast<char*>(Why.data.data()), v_size * h_size * sizeof(float));
    file.read(reinterpret_cast<char*>(bh.data.data()), h_size * sizeof(float));
    file.read(reinterpret_cast<char*>(by.data.data()), v_size * sizeof(float));
    std::cout << "Successfully loaded weights from " << filename << "\n";
    return true;
}

struct LossResult { // total loss and gradients from lossFun
    float loss;
    Matrix dWxh;
    Matrix dWhh;
    Matrix dWhy;
    Vector dbh;
    Vector dby;
    Vector hprev;
};

LossResult lossFun(const std::vector<int>& inputs, const std::vector<int>& targets, const Vector& hprev) {
    int vocab_size = Why.rows;
    int h_size = Wxh.rows;
    int seq_len = static_cast<int>(inputs.size());
    std::map<int, Vector> xs;
    std::map<int, Vector> hs;
    std::map<int, Vector> ys;
    std::map<int, Vector> ps;
    hs[-1] = hprev;
    float loss = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        xs[t] = Vector(vocab_size);
        xs[t][inputs[t]] = 1.0f;
        Vector wx = matrix_vector_mul(Wxh, xs[t]);
        Vector uh = matrix_vector_mul(Whh, hs[t - 1]);
        Vector sum = add(add(wx, uh), bh);
        hs[t] = tanh_vec(sum);
        ys[t] = add(matrix_vector_mul(Why, hs[t]), by);
        ys[t] = clip_vec(ys[t], -500.0f, 500.0f);
        ps[t] = softmax(ys[t]);
        if (targets[t] >= 0 && targets[t] < vocab_size) {
            loss += -std::log(std::max(ps[t][targets[t]], 1e-12f));
        }
    }

    Matrix dWxh(h_size, vocab_size);
    Matrix dWhh(h_size, h_size);
    Matrix dWhy(vocab_size, h_size);
    Vector dbh(h_size);
    Vector dby(vocab_size);
    Vector dhnext(h_size);

    for (int t = seq_len - 1; t >= 0; t--) {
        Vector dy = ps[t];
        dy[targets[t]] -= 1.0f;
        add_inplace(dWhy, outer_product(dy, hs[t]));
        add_inplace(dby, dy);
        Vector dh = add(matrix_transpose_vector_mul(Why, dy), dhnext);
        Vector dhraw(h_size);
        for (int i = 0; i < h_size; i++) {
            dhraw[i] = dh[i] * (1.0f - hs[t][i] * hs[t][i]);
        }
        add_inplace(dbh, dhraw);
        add_inplace(dWxh, outer_product(dhraw, xs[t]));
        add_inplace(dWhh, outer_product(dhraw, hs[t - 1]));
        dhnext = matrix_transpose_vector_mul(Whh, dhraw);
    }

    dWxh = clip_mat(dWxh, 5.0f);
    dWhh = clip_mat(dWhh, 5.0f);
    dWhy = clip_mat(dWhy, 5.0f);
    dbh = clip_vec(dbh, 5.0f);
    dby = clip_vec(dby, 5.0f);

    return {loss, dWxh, dWhh, dWhy, dbh, dby, hs[seq_len - 1]};
}

std::string sample_from_model(const Vector& h, int vocab_size, int sample_length = 200) {
    std::vector<int> predicted_ids;
    Vector x(vocab_size);
    Vector h_curr = h;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < sample_length; i++) {
        Vector wx = matrix_vector_mul(Wxh, x);
        Vector uh = matrix_vector_mul(Whh, h_curr);
        Vector sum = add(add(wx, uh), bh);
        h_curr = tanh_vec(sum);
        Vector y = add(matrix_vector_mul(Why, h_curr), by);
        y = clip_vec(y, -500.0f, 500.0f);
        Vector p_dist = softmax(y);
        float rand_val = dis(gen);
        float cumsum = 0.0f;
        int ix = 0;
        for (int j = 0; j < vocab_size; j++) {
            cumsum += p_dist[j];
            if (rand_val < cumsum) {
                ix = j;
                break;
            }
        }
        predicted_ids.push_back(ix);
        x = Vector(vocab_size);
        x[ix] = 1.0f;
    }
    return tokenizer.decode(predicted_ids);
}

int main(int argc, char* argv[]) {
    std::cout << "=== Orion Generative Model (C++) ===\n\n";
#ifdef USE_CUDA
    std::cout << "Build mode: CUDA support compiled in.\n";
#else
    std::cout << "Build mode: CUDA support NOT compiled in. Define USE_CUDA and link CUDA libs to enable GPU.\n";
#endif
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--priority=gpu" || arg == "--gpu") {
            priority = "gpu";
        } else if (arg == "--priority=cpu" || arg == "--cpu") {
            priority = "cpu";
        }
    }

#ifdef USE_CUDA
    if (priority == "gpu") {
        cuda_supported = init_cuda();
        if (cuda_supported) {
            use_gpu = true;
            std::cout << "Using GPU priority.\n";
        } else {
            std::cout << "GPU requested but CUDA could not be initialized; falling back to CPU.\n";
        }
    } else {
        std::cout << "Using CPU priority.\n";
    }
#else
    if (priority == "gpu") {
        std::cout << "GPU requested but this build does not include CUDA support; using CPU instead.\n";
    } else {
        std::cout << "Using CPU priority.\n";
    }
#endif

    std::cout << "Reading files for BPE training...\n";
    std::string full_text_data = load_training_data();
    if (full_text_data.empty()) {
        std::cerr << "No training data loaded!\n";
        return 1;
    }

    int target_vocab_size = 8000;
    auto tokenize_start = std::chrono::high_resolution_clock::now();
    std::vector<int> data = tokenizer.train(full_text_data, target_vocab_size);
    auto tokenize_end = std::chrono::high_resolution_clock::now();
    int vocab_size = static_cast<int>(tokenizer.vocab.size());
    auto tokenize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tokenize_end - tokenize_start).count();
    std::cout << "\nData tokenized! Original chars: " << full_text_data.size()
              << " -> Tokens: " << data.size() << " in "
              << (tokenize_duration / 1000.0f) << " seconds\n";

    if (!load_checkpoint("my_rnn_model.bin")) {
        std::cout << "Initializing new model...\n";
        Wxh = matrix_random(HIDDEN_SIZE, vocab_size, 0.01f);
        Whh = matrix_random(HIDDEN_SIZE, HIDDEN_SIZE, 0.01f);
        Why = matrix_random(vocab_size, HIDDEN_SIZE, 0.01f);
        bh = vector_zero(HIDDEN_SIZE);
        by = vector_zero(vocab_size);
    }

    int total_params = (HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * vocab_size) * 2 + HIDDEN_SIZE + vocab_size;
    float model_size_mb = (total_params * sizeof(float)) / (1024.0f * 1024.0f);
    std::cout << "Model size: " << std::fixed << std::setprecision(2) << model_size_mb
              << " MB (float32)\n";
    std::cout << "Parameters: " << total_params << "\n\n";

    mWxh = matrix_zero(HIDDEN_SIZE, vocab_size);
    mWhh = matrix_zero(HIDDEN_SIZE, HIDDEN_SIZE);
    mWhy = matrix_zero(vocab_size, HIDDEN_SIZE);
    mbh = vector_zero(HIDDEN_SIZE);
    mby = vector_zero(vocab_size);

    int n = 0;
    int p = 0;
    Vector hprev = vector_zero(HIDDEN_SIZE);

    std::cout << "Starting training loop...\n";
    while (true) {
        if (p + SEQ_LENGTH + 1 >= static_cast<int>(data.size()) || n == 0) {
            hprev = vector_zero(HIDDEN_SIZE);
            p = 0;
        }
        std::vector<int> inputs(data.begin() + p, data.begin() + p + SEQ_LENGTH);
        std::vector<int> targets(data.begin() + p + 1, data.begin() + p + SEQ_LENGTH + 1);
        auto loss_result = lossFun(inputs, targets, hprev);
        hprev = loss_result.hprev;

        auto pupdate_start = std::chrono::high_resolution_clock::now();
        add_inplace(mWxh, square_mat(loss_result.dWxh));
        add_inplace(mWhh, square_mat(loss_result.dWhh));
        add_inplace(mWhy, square_mat(loss_result.dWhy));
        for (int i = 0; i < Wxh.rows * Wxh.cols; i++) {
            Wxh.data[i] += -LEARNING_RATE * loss_result.dWxh.data[i] / (std::sqrt(mWxh.data[i]) + 1e-8f);
        }
        for (int i = 0; i < Whh.rows * Whh.cols; i++) {
            Whh.data[i] += -LEARNING_RATE * loss_result.dWhh.data[i] / (std::sqrt(mWhh.data[i]) + 1e-8f);
        }
        for (int i = 0; i < Why.rows * Why.cols; i++) {
            Why.data[i] += -LEARNING_RATE * loss_result.dWhy.data[i] / (std::sqrt(mWhy.data[i]) + 1e-8f);
        }

        add_inplace(mbh, square_vec(loss_result.dbh));
        add_inplace(mby, square_vec(loss_result.dby));
        for (int i = 0; i < bh.size; i++) {
            bh[i] += -LEARNING_RATE * loss_result.dbh[i] / (std::sqrt(mbh[i]) + 1e-8f);
        }
        for (int i = 0; i < by.size; i++) {
            by[i] += -LEARNING_RATE * loss_result.dby[i] / (std::sqrt(mby[i]) + 1e-8f);
        }

        auto pupdate_end = std::chrono::high_resolution_clock::now();
        if (n % PROGRESS_ITER == 0) {
            std::cout << "iter " << n << ", loss: " << std::fixed << std::setprecision(4)
                      << loss_result.loss << std::endl;
        }
        if (n % SAMPLE_ITER == 0) {
            auto sample_start = std::chrono::high_resolution_clock::now();
            std::string sampled_text = sample_from_model(hprev, vocab_size, 200);
            auto sample_end = std::chrono::high_resolution_clock::now();
            auto update_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pupdate_end - pupdate_start).count();
            auto sample_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sample_end - sample_start).count();
            std::cout << "----\n" << sampled_text << "\n----\n";
            std::cout << "Parameter update: " << (update_duration / 1000.0f) << "s, ";
            std::cout << "Sampling: " << (sample_duration / 1000.0f) << "s\n\n" << std::flush;
        }
        if (n % SAVE_ITER == 0 && n > 0) {
            save_checkpoint("my_rnn_model.bin");
        }
        p += SEQ_LENGTH;
        n++;
        if (n > 100000) break;
    }

    std::cout << "Training completed!\n";
    return 0;
}
