//
// Created by Solmaz on Jan 2025
//

#include "alg.h"  // Assuming alg.h contains necessary class and function definitions  
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <set>
#include <omp.h> // For OpenMP parallelization

int _lsh_UB = 0;

#if defined(unix) || defined(__unix__)
std::string data_fold = "/home/xizhao/dataset/";
std::string index_fold = "./indexes/";
#else
std::string data_fold = "E:/Dataset_for_c/";
std::string index_fold = data_fold + "graphIndex/";
#endif

double convertToSeconds(double milliseconds) {
    return milliseconds / 1000.0;
}

// Replace invalid values (e.g., -1) with defaults
int replaceInvalidValue(int value, int defaultValue) {
    return (value == -1) ? defaultValue : value;
}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
    return std::string(buffer);
}

int main(int argc, char const* argv[]) {
    // Ensure C++17 support
    #if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
    std::cout << "C++17 enabled!\n";
    #else
    std::cout << "Using C++ version < 17\n";
    #endif

    // Log: Starting parameter initialization
    std::cout << "Initializing parameters...\n";

    // Parameters initialization
    float c = 1.5;
    unsigned k = 50;
    unsigned L = 2, K = 18;
    float beta = 0.1;
    unsigned Qnum = 100;
    float W = 1.0f;
    int T = 24;
    int efC = 80;  // Set default ef to 80
    L = 2;
    K = 18;
    double pC = 0.95, pQ = 0.9;
    std::string datasetName;
    bool isBuilt = false;
    _lsh_UB = 0;

    // Read command-line arguments
    if (argc > 1) datasetName = argv[1];
    if (argc > 2) isBuilt = std::atoi(argv[2]);
    if (argc > 3) k = std::atoi(argv[3]);
    if (argc > 4) L = std::atoi(argv[4]);
    if (argc > 5) K = std::atoi(argv[5]);
    if (argc > 6) T = std::atoi(argv[6]);
    if (argc > 7) efC = replaceInvalidValue(std::atoi(argv[7]), 80); // Replace invalid ef value with 80
    if (argc > 8) pC = std::atof(argv[8]);
    if (argc > 9) pQ = std::atof(argv[9]);
    if (argc > 10) _lsh_UB = std::atoi(argv[10]);

    // Set default dataset and parameters
    if (argc == 1) {
        const std::string datas[] = {"audio", "mnist", "cifar", "NUS", "Trevi", "gist", "deep1m", "skew_10M_8d", "gauss_8d", "gauss_25w_128"};
        datasetName = datas[0];
        setW(datasetName, W);
        std::cout << "Using the default configuration!\n\n";
    }

    // Log: Dataset and parameters
    std::cout << "\n*** Graph Initialization ***\n";
    std::cout << "Dataset: " << datasetName << "\n";
    std::cout << "Parameters:\n"
              << "c = " << c << "\n"
              << "k = " << k << "\n"
              << "L = " << L << "\n"
              << "K = " << K << "\n"
              << "T = " << T << "\n"
              << "efC = " << efC << "\n"
              << "pC = " << pC << "\n"
              << "pQ = " << pQ << "\n";

    // Preprocessing
    std::cout << "Loading dataset and preparing benchmarks...\n";
    auto startPreprocessing = std::chrono::high_resolution_clock::now();
    Preprocess prep(data_fold + datasetName + ".data", data_fold + "ANN/" + datasetName + ".bench_graph");
    auto endPreprocessing = std::chrono::high_resolution_clock::now();
    double preprocessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endPreprocessing - startPreprocessing).count();
    std::cout << "Dataset Preprocessing Completed in: " << convertToSeconds(preprocessingTime) << " seconds\n";

    // Graph initialization
    std::string path = index_fold + datasetName + ".index";
    Parameter param1(prep, L, K, 1.0f);
    param1.W = 0.3f;
    divGraph* divG = nullptr;

    auto startConstruction = std::chrono::high_resolution_clock::now();
    if (isBuilt && find_file(path + "_divGraph")) {
        std::cout << "Loading existing graph index...\n";
        divG = new divGraph(&prep, path + "_divGraph", pQ);
        divG->L = L;
    } else {
        std::cout << "Building new graph index...\n";
        if (!GenericTool::CheckPathExistence(index_fold.c_str())) {
            GenericTool::EnsurePathExistence(index_fold.c_str());
        }

        auto startDistanceCalc = std::chrono::high_resolution_clock::now();
        divG = new divGraph(prep, param1, path + "_divGraph", T, efC, pC, pQ);
        auto endDistanceCalc = std::chrono::high_resolution_clock::now();
        double distanceCalcTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDistanceCalc - startDistanceCalc).count();
        std::cout << "Distance Calculation Completed in: " << convertToSeconds(distanceCalcTime) << " seconds\n";
    }
    auto endConstruction = std::chrono::high_resolution_clock::now();

    double constructionTime = std::chrono::duration_cast<std::chrono::milliseconds>(endConstruction - startConstruction).count();
    std::cout << "\nGraph Construction Completed in: " << convertToSeconds(constructionTime) << " seconds\n";

    // Incremental Graph Construction
    std::cout << "\n*** Incremental Graph Construction ***\n";
    std::vector<int> incremental_nodes(prep.data.N);
    std::iota(incremental_nodes.begin(), incremental_nodes.end(), 0);
    std::shuffle(incremental_nodes.begin(), incremental_nodes.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    size_t batch_size = 3000;
    size_t total_inserted = 0;

    auto incrementalStartTime = std::chrono::high_resolution_clock::now();
    for (size_t batch_start = 0; batch_start < incremental_nodes.size(); batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, incremental_nodes.size());

        auto batchStartTime = std::chrono::high_resolution_clock::now();
        std::cout << getCurrentTimestamp() << " - Processing batch " << (batch_start / batch_size + 1) << "...\n";

 //       #pragma omp parallel for num_threads(8)
 //       for (size_t i = batch_start; i < batch_end; ++i) {
 //           divG->insert(incremental_nodes[i]);
 //       }

        auto batchEndTime = std::chrono::high_resolution_clock::now();
        auto batchDuration = std::chrono::duration_cast<std::chrono::microseconds>(batchEndTime - batchStartTime).count();

        total_inserted += (batch_end - batch_start);
        std::cout << getCurrentTimestamp() << " - Batch completed in " << (batchDuration / 1000.0) << " milliseconds. Total inserted: " << total_inserted << "\n";
    }
    auto incrementalEndTime = std::chrono::high_resolution_clock::now();
    double incrementalTime = std::chrono::duration_cast<std::chrono::milliseconds>(incrementalEndTime - incrementalStartTime).count();

    // Batch Graph Construction
    std::cout << "\n*** Batch Graph Construction ***\n";
    auto batchStartTime = std::chrono::high_resolution_clock::now();
    divGraph* batchDivG = new divGraph(prep, param1, path + "_divGraph", T, efC, pC, pQ);
	divG->ef = 200;
 //   #pragma omp parallel for num_threads(8)
 //   for (size_t i = 0; i < incremental_nodes.size(); ++i) {
 //       batchDivG->insert(incremental_nodes[i]);
 //   }

    auto batchEndTime = std::chrono::high_resolution_clock::now();
    double batchTime = std::chrono::duration_cast<std::chrono::milliseconds>(batchEndTime - batchStartTime).count();

    std::cout << "Incremental Time: " << convertToSeconds(incrementalTime) << " seconds\n";
    std::cout << "Batch Time: " << convertToSeconds(batchTime) << " seconds\n";

    // Evaluate Recall and Query Performance
    std::cout << "\n*** Evaluating Graph ***\n";
	std::cout << "******************************************************************************************************\n";
    std::cout << std::left << std::setw(18) << "Algorithm"
              << std::setw(8) << "k"
              << std::setw(7) << "ef"
              << std::setw(13) << "Time(s)"
              << std::setw(12) << "Recall"
              << std::setw(12) << "Cost"
              << std::setw(12) << "CPQ1"
              << std::setw(11) << "CPQ2"
              << std::setw(12) << "Pruning" << "\n";
std::cout << "******************************************************************************************************\n";
    auto startQuery = std::chrono::high_resolution_clock::now();
    graphSearch(c, k, divG, prep, beta, datasetName, data_fold, efC); // Pass efC directly
    auto endQuery = std::chrono::high_resolution_clock::now();

    double queryTime = std::chrono::duration_cast<std::chrono::milliseconds>(endQuery - startQuery).count();
    std::cout << "\nQuery Completed in: " << convertToSeconds(queryTime) << " seconds\n";

    // Summary
    std::cout << "\n*** Summary ***\n";
    std::cout << "Total Preprocessing Time: " << convertToSeconds(preprocessingTime) << " seconds\n";
    std::cout << "Total Graph Construction Time: " << convertToSeconds(constructionTime) << " seconds\n";
    std::cout << "Total Query Time: " << convertToSeconds(queryTime) << " seconds\n";
    std::cout << "Total Inserted Nodes: " << total_inserted << "\n";
	std::cout << "Current ef value: " << divG->ef << "\n";
    // Cleanup
    delete divG;
    delete batchDivG;

    return 0;
}
