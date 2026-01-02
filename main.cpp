#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <sys/stat.h>

//platform specific directory creation
#ifdef _WIN32
    #include <direct.h>
    #define mkdir(dir, mode) _mkdir(dir)
#endif

//external cec2017 benchmark function
extern "C" {
    void cec17_test_func(double *x, double *f, int nx, int mx, int func_num);
}

using namespace std;
using namespace std::chrono;

//CEC2017 known optimal values (F1-F30)
const double CEC17_OPTIMA[30] = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000
};

//safe random number generation with proper seed management
class RNG {
private:
    static mt19937 gen;
    static uniform_real_distribution<double> dis;
    
public:
    static double rand01() {
        return dis(gen);
    }
    
    static void setSeed(unsigned int seed) {
        gen.seed(seed);
    }
};

//static number initilization
mt19937 RNG::gen(12345);
uniform_real_distribution<double> RNG::dis(0.0, 1.0);

//Structure to hold convergence results
struct ConvergenceResult {
    double final_fitness;
    int CFEs;           //Convergence Function Evaluations
    double CTs;         //Calculation Time (seconds)
    bool converged;     //Did it converge within maxFE?
};

//Particle structure
struct Particle {
    double* x;
    double* v;
    double* pbest;
    double* grad_curr;
    double* grad_prev;
    double fitness;
    double pbest_fitness;
};

//Fitness function
double fitnessFunction(const double* x, int D, int FUNC_ID) {
    double f;
    cec17_test_func(const_cast<double*>(x), &f, D, 1, FUNC_ID);
    return f;
}

//LDS generation - DES
void generate_LDS_DES(double* P, int size) {
    for (int i = 0; i < size; i++)
        P[i] = fmod(0.6180339887 * (i + 1), 1.0);
}

//LDS generation - OHS
void generate_LDS_OHS(double* P, int size) {
    for (int i = 0; i < size; i++)
        P[i] = RNG::rand01();
}

//Initialize swarm
void initializeSwarm_LDS(Particle* swarm, int N, int D,
                         double a, double b, double* P0,
                         int& FE, int FUNC_ID) {
    for (int i = 0; i < N; i++) {
        swarm[i].x = new double[D];
        swarm[i].v = new double[D];
        swarm[i].pbest = new double[D];
        swarm[i].grad_curr = new double[D];
        swarm[i].grad_prev = new double[D];
        
        //Initialize position using LDS for uniform coverage
        for (int d = 0; d < D; d++) {
            int idx = i * D + d;
            swarm[i].x[d] = a + P0[idx] * (b - a);
            swarm[i].v[d] = 0.0;
            swarm[i].pbest[d] = swarm[i].x[d];
            swarm[i].grad_curr[d] = 0.0;
            swarm[i].grad_prev[d] = 0.0;
        }

        //Evaluate initial fitness
        swarm[i].fitness = fitnessFunction(swarm[i].x, D, FUNC_ID);
        swarm[i].pbest_fitness = swarm[i].fitness;
        FE++;
    }
}

//Get global best index
int getGbest(Particle* swarm, int N) {
    int best = 0;
    for (int i = 1; i < N; i++)
        if (swarm[i].fitness < swarm[best].fitness)
            best = i;
    return best;
}

//LDS-VUS algorithm
void LDS_VUS(Particle* swarm, int N, int N1, int D, double* P1, double* P2, double* pg, double* gbest,double G, double g) {
    //calculate time varient parameter
    double w  = 0.99 - 0.79 * g / G; 
    double k  = 3.0  - 0.5  * g / G;
    double c1 = 2.5  - 2.0  * g / G;
    double c2 = 0.5  + 2.0  * g / G;

    //update velocity for all the particles
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            if (i < N1) {
                int idx = i * D + d;
                swarm[i].v[d] = w * swarm[i].v[d] + k * P1[idx] * (pg[i * D + d] - swarm[i].x[d]);
            } else {
                int idx = (i - N1) * D + d;
                double eps = RNG::rand01();
                swarm[i].v[d] = w * swarm[i].v[d] + c1 * P2[idx] * (pg[i * D + d] - swarm[i].x[d]) + c2 * eps * (gbest[d] - swarm[i].x[d]);
            }
        }
    }
}

//CCG-VUS algorithm
void CCG_VUS(Particle* swarm, int N, int N1, int D, double* P1, double* P2, double* pg, double* gbest, double G, double g) {
    double w = 0.99 - 0.79 * g / G;
    double k = 3.0  - 0.5  * g / G;
    double c1 = 2.5 - 2.0 * g / G;
    double c2 = 0.5 + 2.0 * g / G;

    //Update exploration subpopulation
    for (int i = 0; i < N1; i++) {
        for (int d = 0; d < D; d++) {
            int idx = i * D + d;
            swarm[i].v[d] = w * swarm[i].v[d] + k * P1[idx] * (pg[i * D + d] - swarm[i].x[d]);
        }
    }

    //Update exploitation subpopulation with CCG(conjugate gradient)
    for (int i = N1; i < N; i++) {
        int idx_base = (i - N1) * D;
        
        //Store previous gradient
        for (int d = 0; d < D; d++) {
            swarm[i].grad_prev[d] = swarm[i].grad_curr[d];
        }
        
        //Compute current gradient
        for (int d = 0; d < D; d++) {
            double eps = RNG::rand01();
            int idx = idx_base + d;
            swarm[i].grad_curr[d] = -c1 * P2[idx] * (pg[i * D + d] - swarm[i].x[d]) - c2 * eps * (gbest[d] - swarm[i].x[d]);
        }
        
        //Compute conjugate inertia coefficient
        double norm_curr_sq = 0.0;
        double norm_prev_sq = 0.0;
        
        for (int d = 0; d < D; d++) {
            norm_curr_sq += swarm[i].grad_curr[d] * swarm[i].grad_curr[d];
            norm_prev_sq += swarm[i].grad_prev[d] * swarm[i].grad_prev[d];
        }
        
        double wCG = (norm_prev_sq > 1e-12) ? (norm_curr_sq / norm_prev_sq) : 1.0;
        
        //Update velocity based on wCG value
        if (wCG > 1.0) {
            //Use LDS-VUS when wCG > 1
            for (int d = 0; d < D; d++) {
                double eps = RNG::rand01();
                int idx = idx_base + d;
                swarm[i].v[d] = w * swarm[i].v[d] + c1 * P2[idx] * (pg[i * D + d] - swarm[i].x[d]) + c2 * eps * (gbest[d] - swarm[i].x[d]);
            }
        } else {
            //Use CCG-VUS when wCG <= 1
            for (int d = 0; d < D; d++) {
                double eps = RNG::rand01();
                int idx = idx_base + d;
                swarm[i].v[d] = wCG * swarm[i].v[d] + c1 * P2[idx] * (pg[i * D + d] - swarm[i].x[d]) + c2 * eps * (gbest[d] - swarm[i].x[d]);
            }
        }
    }
}

//Update comprehensive learning vector
void update_pg(Particle* swarm, double* pg, int N, int D) {
    for (int i = 0; i < N; i++) {
        //calculate learning probability-increased based on the particle index
        double Pc = 0.05 + 0.45 * (exp(10.0 * i / (N - 1)) - 1.0) / (exp(10.0) - 1.0);
        int exemplar = i;

        if (RNG::rand01() < Pc) {
            int r1 = (int)(RNG::rand01() * N);
            int r2 = (int)(RNG::rand01() * N);
            exemplar = (swarm[r1].pbest_fitness < swarm[r2].pbest_fitness) ? r1 : r2;
        }

        //copy p best to the learning vector
        for (int d = 0; d < D; d++)
            pg[i * D + d] = swarm[exemplar].pbest[d];
    }
}

//Position update
void updatePosition(Particle& p, int D, double a, double b) {
    for (int d = 0; d < D; d++) {
        p.x[d] += p.v[d];
        p.x[d] = max(a, min(b, p.x[d]));
    }
}

//Check convergence based on epsilon tolerance
bool checkConvergence(double current_fitness, double optimal_value, double epsilon_tol) {
    double relative_error = fabs((current_fitness - optimal_value) / optimal_value) * 100.0;
    return relative_error < epsilon_tol;
}

//Core HCLPSO algorithm with convergence tracking
ConvergenceResult run_HCLPSO_single(bool isDC, int N, int D, int FUNC_ID, int maxFE, double epsilon_tol) {
    ConvergenceResult result;
    result.converged = false;
    result.CFEs = maxFE;  //set to maxFE if not converged
    
    int N1 = N / 2;
    double gCG = 0.6;
    int FE = 0;

    //Start timing
    auto start_time = high_resolution_clock::now();

    //generate initial LDS for population initialization
    double* P0 = new double[N * D];
    if (isDC) generate_LDS_DES(P0, N * D);
    else      generate_LDS_OHS(P0, N * D);
    
    //initialize particle swarm
    Particle* swarm = new Particle[N];
    initializeSwarm_LDS(swarm, N, D, -100, 100, P0, FE, FUNC_ID);

    //generate LDS sequences for velocity update 
    double* P1 = new double[N1 * D];
    double* P2 = new double[(N - N1) * D];
    double* pg = new double[N * D];

    if (isDC) {
        generate_LDS_DES(P1, N1 * D);
        generate_LDS_DES(P2, (N - N1) * D);
    } else {
        generate_LDS_OHS(P1, N1 * D);
        generate_LDS_OHS(P2, (N - N1) * D);
    }

    //take the initial global best value
    int gbestIdx = getGbest(swarm, N);
    double optimal_value = CEC17_OPTIMA[FUNC_ID - 1];

    //Main optimization loop
    while (FE < maxFE) {
        //take the values and update the comprehensive learning factors
        update_pg(swarm, pg, N, D);

        //based on the current the current phase apply the velocity update stratergy 
        if ((double)FE / maxFE < gCG)
            LDS_VUS(swarm, N, N1, D, P1, P2, pg, swarm[gbestIdx].x, maxFE, FE);
        else
            CCG_VUS(swarm, N, N1, D, P1, P2, pg, swarm[gbestIdx].x, maxFE, FE);

        //Update positions, evaluate fitness, and update personal bests
        for (int i = 0; i < N; i++) {
            updatePosition(swarm[i], D, -100, 100);
            swarm[i].fitness = fitnessFunction(swarm[i].x, D, FUNC_ID);
            FE++;
            
            if (swarm[i].fitness < swarm[i].pbest_fitness) {
                swarm[i].pbest_fitness = swarm[i].fitness;
                for (int d = 0; d < D; d++)
                    swarm[i].pbest[d] = swarm[i].x[d];
            }
        }
        
        gbestIdx = getGbest(swarm, N);
        
        //Check convergence
        if (!result.converged && checkConvergence(swarm[gbestIdx].fitness, optimal_value, epsilon_tol)) {
            result.converged = true;
            result.CFEs = FE;
            auto end_time = high_resolution_clock::now();
            result.CTs = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
        }
    }

    result.final_fitness = swarm[gbestIdx].fitness;
    
    //If never converged, record full time
    if (!result.converged) {
        auto end_time = high_resolution_clock::now();
        result.CTs = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    }

    //Cleanup memory
    for (int i = 0; i < N; i++) {
        delete[] swarm[i].x;
        delete[] swarm[i].v;
        delete[] swarm[i].pbest;
        delete[] swarm[i].grad_curr;
        delete[] swarm[i].grad_prev;
    }
    delete[] swarm;
    delete[] P1;
    delete[] P2;
    delete[] pg;
    delete[] P0;

    return result;
}

//Statistical calculations -calculate the arithmatic mean of data
double calculateMean(double* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) sum += data[i];
    return sum / size;
}

//calculate standard deviation given mean value
double calculateStd(double* data, int size, double mean) {
    double variance = 0.0;
    for (int i = 0; i < size; i++) 
        variance += (data[i] - mean) * (data[i] - mean);
    return sqrt(variance / size);
}

//calculate median
double calculateMedian(double* data, int size) {
    double* temp = new double[size];
    for (int i = 0; i < size; i++) temp[i] = data[i];
    sort(temp, temp + size);
    double result = (size % 2 == 0) ? (temp[size/2 - 1] + temp[size/2]) / 2.0 : temp[size/2];
    delete[] temp;
    return result;
}

double getMin(double* data, int size) {
    double minVal = data[0];
    for (int i = 1; i < size; i++)
        if (data[i] < minVal) minVal = data[i];
    return minVal;
}

double getMax(double* data, int size) {
    double maxVal = data[0];
    for (int i = 1; i < size; i++)
        if (data[i] > maxVal) maxVal = data[i];
    return maxVal;
}

//epsilon tolerance based on dimension
double getEpsilonTol(int D) {
    if (D == 10) return 5.0;      //5% for D=10
    else if (D == 30) return 20.0; //20% for D=30
    else if (D == 50) return 30.0; //30% for D=50
    else return 40.0;              //40% for D=100
}

//max FE based on dimension
int getMaxFE(int D) {
    if (D == 10) return 100000;       //100k for D=10
    else if (D == 30) return 300000;  //300k for D=30
    else if (D == 50) return 500000;  //500k for D=50
    else return 1000000;              //1M for D=100
}

//Print table helpers
void printSeparator(ofstream& file, int width = 140) {
    file << string(width, '=') << "\n";
}

void printDashedLine(ofstream& file, int width = 140) {
    file << string(width, '-') << "\n";
}

//Run experiments for one dimension
void runExperimentsForDimension(bool isDC, int N, int D, int numRuns, const string& outputDir) {
    
    string mode = isDC ? "DC" : "OC";
    string algoName = "HCLPSO-" + mode;
    
    double epsilon_tol = getEpsilonTol(D);
    int maxFE = getMaxFE(D);
    
    string dimDir = outputDir + "/D" + to_string(D);
    mkdir(dimDir.c_str(), 0777);
    
    //CALCULATION ACCURACY (Mean, Std, Median, Best, Worst)
    ofstream accuracyFile(dimDir + "/" + algoName + "_D" + to_string(D) + "_accuracy.txt");
    accuracyFile << fixed << setprecision(6);
    
    accuracyFile << "\n";
    printSeparator(accuracyFile);
    accuracyFile << "TABLE: CALCULATION ACCURACY - " << algoName << " (D=" << D << ")\n";
    accuracyFile << "Population: N=" << N << " | Runs: " << numRuns << " | MaxFE: " << maxFE 
                 << " | εtol: " << epsilon_tol << "%\n";
    printSeparator(accuracyFile);
    accuracyFile << left << setw(12) << "Function"
                 << right << setw(18) << "Mean"
                 << right << setw(18) << "Std"
                 << right << setw(18) << "Median"
                 << right << setw(18) << "Best"
                 << right << setw(18) << "Worst" << "\n";
    printDashedLine(accuracyFile);
    
    //CONVERGENCE SPEED (CFEs and Success Rate)
    ofstream speedFile(dimDir + "/" + algoName + "_D" + to_string(D) + "_convergence_speed.txt");
    speedFile << fixed << setprecision(2);
    
    speedFile << "\n";
    printSeparator(speedFile);
    speedFile << "TABLE: CONVERGENCE SPEED - " << algoName << " (D=" << D << ")\n";
    speedFile << "Population: N=" << N << " | Runs: " << numRuns << " | MaxFE: " << maxFE 
              << " | εtol: " << epsilon_tol << "%\n";
    printSeparator(speedFile);
    speedFile << left << setw(12) << "Function"
              << right << setw(18) << "Mean CFEs"
              << right << setw(18) << "Std CFEs"
              << right << setw(18) << "Success Rate"
              << right << setw(18) << "Mean CTs (s)"
              << right << setw(18) << "Std CTs (s)" << "\n";
    printDashedLine(speedFile);
    
    cout << "\n" << string(80, '=') << "\n";
    cout << "Running " << algoName << " on CEC2017 (D=" << D << ")\n";
    cout << "Population: " << N << " | Runs: " << numRuns << " | MaxFE: " << maxFE 
         << " | εtol: " << epsilon_tol << "%\n";
    cout << string(80, '=') << "\n\n";
    
    //Test functions
    int funcIDs[29];
    int funcCount = 0;
    for (int i = 1; i <= 30; i++) {
        if (i != 2) funcIDs[funcCount++] = i;
    }
    
    double* all_mean_errors = new double[29];  //For average rank calculation
    int errorIdx = 0;
    
    for (int idx = 0; idx < funcCount; idx++) {
        int funcID = funcIDs[idx];
        cout << "F" << setw(2) << funcID << " " << flush;
        
        //Allocate arrays for results
        double* errors = new double[numRuns];
        double* cfes_list = new double[numRuns];
        double* cts_list = new double[numRuns];
        int convergence_count = 0;
        
        //Individual function detail file
        ofstream detailFile(dimDir + "/" + algoName + "_F" + to_string(funcID) + "_D" + to_string(D) + "_details.txt");
        detailFile << fixed << setprecision(6);
        detailFile << "Function F" << funcID << " - " << algoName << " (D=" << D << ")\n";
        printDashedLine(detailFile, 100);
        detailFile << left << setw(8) << "Run"
                   << right << setw(20) << "Error"
                   << right << setw(20) << "Fitness"
                   << right << setw(15) << "CFEs"
                   << right << setw(15) << "CTs (s)"
                   << right << setw(12) << "Converged" << "\n";
        printDashedLine(detailFile, 100);
        
        for (int run = 0; run < numRuns; run++) {
            RNG::setSeed(run * 54321 + funcID * 9876 + D * 111);
            
            ConvergenceResult res = run_HCLPSO_single(isDC, N, D, funcID, maxFE, epsilon_tol);
            double error = res.final_fitness - CEC17_OPTIMA[funcID - 1];
            
            errors[run] = error;
            cfes_list[run] = res.CFEs;
            cts_list[run] = res.CTs;
            if (res.converged) convergence_count++;
            
            detailFile << left << setw(8) << (run + 1)
                      << right << setw(20) << scientific << error
                      << right << setw(20) << fixed << res.final_fitness
                      << right << setw(15) << res.CFEs
                      << right << setw(15) << setprecision(4) << res.CTs
                      << right << setw(12) << (res.converged ? "Yes" : "No") << "\n";
        }
        printDashedLine(detailFile, 100);
        detailFile.close();
        
        //Calculate statistics
        double mean_error = calculateMean(errors, numRuns);
        double std_error = calculateStd(errors, numRuns, mean_error);
        double median_error = calculateMedian(errors, numRuns);
        double best_error = getMin(errors, numRuns);
        double worst_error = getMax(errors, numRuns);
        
        double mean_cfes = calculateMean(cfes_list, numRuns);
        double std_cfes = calculateStd(cfes_list, numRuns, mean_cfes);
        double mean_cts = calculateMean(cts_list, numRuns);
        double std_cts = calculateStd(cts_list, numRuns, mean_cts);
        double success_rate = (convergence_count * 100.0) / numRuns;
        
        all_mean_errors[errorIdx++] = mean_error;
        
        //Write to accuracy table
        accuracyFile << left << setw(12) << ("F" + to_string(funcID))
                    << right << setw(18) << scientific << setprecision(4) << mean_error
                    << right << setw(18) << std_error
                    << right << setw(18) << median_error
                    << right << setw(18) << best_error
                    << right << setw(18) << worst_error << "\n";
        
        //Write to convergence speed table
        speedFile << left << setw(12) << ("F" + to_string(funcID))
                 << right << setw(18) << fixed << setprecision(0) << mean_cfes
                 << right << setw(18) << std_cfes
                 << right << setw(18) << setprecision(1) << success_rate << "%"
                 << right << setw(18) << setprecision(4) << mean_cts
                 << right << setw(18) << std_cts << "\n";
        
        cout << "✓ (Mean: " << scientific << setprecision(2) << mean_error 
             << ", SR: " << fixed << setprecision(0) << success_rate << "%)\n";
        
        //Clean up arrays
        delete[] errors;
        delete[] cfes_list;
        delete[] cts_list;
    }
    
    printDashedLine(accuracyFile);
    printDashedLine(speedFile);
    
    //Calculate overall statistics
    double overall_mean = calculateMean(all_mean_errors, errorIdx);
    
    accuracyFile << "\nOVERALL STATISTICS:\n";
    accuracyFile << "Average Mean Error: " << scientific << overall_mean << "\n";
    
    printSeparator(accuracyFile);
    printSeparator(speedFile);
    
    accuracyFile.close();
    speedFile.close();
    
    delete[] all_mean_errors;
    
    cout << "\n" << string(80, '=') << "\n";
    cout << "D=" << D << " experiments completed!\n";
    cout << "Files saved to: " << dimDir << "/\n";
    cout << string(80, '=') << "\n\n";
}

//Run all experiments
void runAllExperiments(bool isDC, const string& outputDir) {
    mkdir(outputDir.c_str(), 0777);
    
    int N = 40;
    int numRuns = 51;
    
    int dimensions[] = {10, 30, 50, 100};
    int dimCount = 4;
    
    for (int i = 0; i < dimCount; i++) {
        int D = dimensions[i];
        cout << "\n********** TESTING DIMENSION D=" << D << " **********\n";
        runExperimentsForDimension(isDC, N, D, numRuns, outputDir);
    }
    
    //Create summary comparison file
    ofstream summaryFile(outputDir + "/OVERALL_SUMMARY.txt");
    summaryFile << "\n";
    printSeparator(summaryFile);
    summaryFile << "HCLPSO-" << (isDC ? "DC" : "OC") << " - OVERALL SUMMARY ACROSS ALL DIMENSIONS\n";
    printSeparator(summaryFile);
    summaryFile << "\nThis experiment tested the algorithm on:\n";
    summaryFile << "- 29 CEC2017 functions (F1-F30, excluding F2)\n";
    summaryFile << "- 4 dimensions: D = 10, 30, 50, 100\n";
    summaryFile << "- 51 independent runs per function\n";
    summaryFile << "\nMetrics tracked:\n";
    summaryFile << "1. Calculation Accuracy: Mean, Std, Median, Best, Worst\n";
    summaryFile << "2. Convergence Speed: CFEs (function evaluations to converge)\n";
    summaryFile << "3. Success Rate: Percentage of runs that converged\n";
    summaryFile << "4. Calculation Time: CTs (seconds)\n";
    summaryFile << "\nResults are organized by dimension in separate subdirectories.\n";
    printSeparator(summaryFile);
    summaryFile.close();
}

//Main
int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: ./hclpso DC|OC\n";
        return 1;
    }

    string outputDir = "cec_results";
    
    if (string(argv[1]) == "DC") {
        runAllExperiments(true, outputDir);
    } else if (string(argv[1]) == "OC") {
        runAllExperiments(false, outputDir);
    } else {
        cerr << "Invalid mode. Use DC or OC\n";
        return 1;
    }

    return 0;
}