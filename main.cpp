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

//CEC2017 optimal values (F1-F30)
const double CEC17_OPTIMA[30] = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000
};

//RNG with proper seed management
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
void LDS_VUS(Particle* swarm, int N, int N1, int D,
             double* P1, double* P2, double* pg, double* gbest,
             int G, int g) {
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
void CCG_VUS(Particle* swarm, int N, int N1, int D,
             double* P1, double* P2, double* pg, double* gbest,
             int G, int g) {
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

    //Update exploitation subpopulation with CCG
    for (int i = N1; i < N; i++) {
        int idx_base = (i - N1) * D;
        
        //Store previous gradient
        for (int d = 0; d < D; d++) {
            swarm[i].grad_prev[d] = swarm[i].grad_curr[d];
        }
        
        //Compute current gradient (Eq. 24)
        for (int d = 0; d < D; d++) {
            double eps = RNG::rand01();
            int idx = idx_base + d;
            swarm[i].grad_curr[d] = -c1 * P2[idx] * (pg[i * D + d] - swarm[i].x[d]) - c2 * eps * (gbest[d] - swarm[i].x[d]);
        }
        
        //Compute conjugate inertia coefficient (Eq. 23 - FR method)
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
    result.CFEs = maxFE;  // set to maxFE if not converged
    
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
double calculateMean(const vector<double>& data) {
    double sum = 0.0;
    for (double val : data) sum += val;
    return sum / data.size();
}

//calculate standard deviation given mean value
double calculateStd(const vector<double>& data, double mean) {
    double variance = 0.0;
    for (double val : data) {
        variance += (val - mean) * (val - mean);
    }
    return sqrt(variance / data.size());
}

//calculate median
double calculateMedian(vector<double> data) {
    sort(data.begin(), data.end());
    int n = data.size();
    if (n % 2 == 0)
        return (data[n/2 - 1] + data[n/2]) / 2.0;
    else
        return data[n/2];
}

//Print formatted table
void printTableHeader(ofstream& file, const string& title) {
    file << "\n" << string(120, '=') << "\n";
    file << title << "\n";
    file << string(120, '=') << "\n";
}

void printTableRow(ofstream& file, const string& col1, const string& col2, 
                   const string& col3, const string& col4, 
                   const string& col5, const string& col6) {
    file << left << setw(10) << col1 
         << right << setw(18) << col2 
         << right << setw(18) << col3
         << right << setw(18) << col4
         << right << setw(18) << col5
         << right << setw(18) << col6 << "\n";
}

void printTableRow7(ofstream& file, const string& col1, const string& col2, 
                    const string& col3, const string& col4, 
                    const string& col5, const string& col6, const string& col7) {
    file << left << setw(10) << col1 
         << right << setw(15) << col2 
         << right << setw(15) << col3
         << right << setw(15) << col4
         << right << setw(15) << col5
         << right << setw(15) << col6
         << right << setw(15) << col7 << "\n";
}

//Run experiments for one dimension
void runExperimentsForDimension(bool isDC, int N, int D, int numRuns, 
                                 double epsilon_tol, const string& outputDir) {
    
    string mode = isDC ? "DC" : "OC";
    string algoName = "HCLPSO-" + mode;
    
    //Determine maxFE based on dimension
    int maxFE = (D == 10) ? 100000 : 300000;
    
    //Create dimension-specific directory
    string dimDir = outputDir + "/D" + to_string(D);
    mkdir(dimDir.c_str(), 0777);
    
    //Open summary files
    ofstream summaryFile(dimDir + "/" + algoName + "_D" + to_string(D) + "_summary.txt");
    ofstream cfesFile(dimDir + "/" + algoName + "_D" + to_string(D) + "_CFEs.txt");
    ofstream ctsFile(dimDir + "/" + algoName + "_D" + to_string(D) + "_CTs.txt");
    
    summaryFile << fixed << setprecision(6);
    cfesFile << fixed << setprecision(2);
    ctsFile << fixed << setprecision(4);
    
    printTableHeader(summaryFile, algoName + " Accuracy Results (D=" + to_string(D) + ", Runs=" + to_string(numRuns) + ", εtol=" + to_string(epsilon_tol) + "%)");
    summaryFile << "\n";
    printTableRow(summaryFile, "Function", "Mean", "Std", "Median", "Best", "Worst");
    summaryFile << string(120, '-') << "\n";
    
    printTableHeader(cfesFile, algoName + " Convergence Speed (CFEs) - D=" + to_string(D));
    cfesFile << "\n";
    printTableRow7(cfesFile, "Function", "Mean CFEs", "Std CFEs", "Conv. Rate", "Mean CTs", "Std CTs", "Min CTs");
    cfesFile << string(120, '-') << "\n";
    
    cout << "\n" << string(80, '=') << "\n";
    cout << "Running " << algoName << " on CEC2017 (D=" << D << ", εtol=" << epsilon_tol << "%)\n";
    cout << "Population: " << N << " | Runs: " << numRuns << " | MaxFE: " << maxFE << "\n";
    cout << string(80, '=') << "\n\n";
    
    //Test functions
    vector<int> funcIDs;
    for (int i = 1; i <= 30; i++) {
        if (i != 2) funcIDs.push_back(i);  //Exclude F2
    }
    
    for (int funcID : funcIDs) {
        cout << "F" << setw(2) << funcID << " ";
        cout.flush();
        
        vector<double> errors;
        vector<double> cfes_list;
        vector<double> cts_list;
        int convergence_count = 0;
        
        //Individual function file
        ofstream funcFile(dimDir + "/" + algoName + "_F" + to_string(funcID) + "_D" + to_string(D) + ".txt");
        funcFile << fixed << setprecision(6);
        
        printTableHeader(funcFile, "F" + to_string(funcID) + " - " + algoName + " (D=" + to_string(D) + ")");
        funcFile << "\n";
        funcFile << left << setw(8) << "Run" 
                 << right << setw(18) << "Error" 
                 << right << setw(18) << "Fitness"
                 << right << setw(15) << "CFEs"
                 << right << setw(12) << "CTs (s)"
                 << right << setw(12) << "Converged" << "\n";
        funcFile << string(85, '-') << "\n";
        
        //Run multiple trials
        for (int run = 0; run < numRuns; run++) {
            RNG::setSeed(run * 54321 + funcID * 9876);
            
            ConvergenceResult res = run_HCLPSO_single(isDC, N, D, funcID, maxFE, epsilon_tol);
            double error = res.final_fitness - CEC17_OPTIMA[funcID - 1];
            
            errors.push_back(error);
            cfes_list.push_back(res.CFEs);
            cts_list.push_back(res.CTs);
            if (res.converged) convergence_count++;
            
            funcFile << left << setw(8) << (run + 1)<< right << setw(18) << scientific << error<< right << setw(18) << fixed << res.final_fitness<< right << setw(15) << res.CFEs << right << setw(12) << setprecision(4) << res.CTs<< right << setw(12) << (res.converged ? "Yes" : "No") << "\n";
        }
        
        funcFile << string(85, '-') << "\n";
        funcFile.close();
        
        //Calculate statistics
        double mean_error = calculateMean(errors);
        double std_error = calculateStd(errors, mean_error);
        double median_error = calculateMedian(errors);
        double best_error = *min_element(errors.begin(), errors.end());
        double worst_error = *max_element(errors.begin(), errors.end());
        
        double mean_cfes = calculateMean(cfes_list);
        double std_cfes = calculateStd(cfes_list, mean_cfes);
        double mean_cts = calculateMean(cts_list);
        double std_cts = calculateStd(cts_list, mean_cts);
        double min_cts = *min_element(cts_list.begin(), cts_list.end());
        double conv_rate = (convergence_count * 100.0) / numRuns;
        
        //Write to summary
        ostringstream fn, me, se, md, be, we;
        fn << "F" << funcID;
        me << scientific << setprecision(4) << mean_error;
        se << scientific << setprecision(4) << std_error;
        md << scientific << setprecision(4) << median_error;
        be << scientific << setprecision(4) << best_error;
        we << scientific << setprecision(4) << worst_error;
        printTableRow(summaryFile, fn.str(), me.str(), se.str(), md.str(), be.str(), we.str());
        
        //Write CFEs and CTs
        ostringstream fn2, mc, sc, cr, mt, st, mint;
        fn2 << "F" << funcID;
        mc << fixed << setprecision(0) << mean_cfes;
        sc << fixed << setprecision(0) << std_cfes;
        cr << fixed << setprecision(1) << conv_rate << "%";
        mt << fixed << setprecision(4) << mean_cts;
        st << fixed << setprecision(4) << std_cts;
        mint << fixed << setprecision(4) << min_cts;
        printTableRow7(cfesFile, fn2.str(), mc.str(), sc.str(), cr.str(), mt.str(), st.str(), mint.str());
        
        cout << "✓ Mean: " << scientific << setprecision(2) << mean_error 
             << " | CFEs: " << fixed << setprecision(0) << mean_cfes << "\n";
    }
    
    summaryFile << string(120, '=') << "\n";
    cfesFile << string(120, '=') << "\n";
    summaryFile.close();
    cfesFile.close();
    ctsFile.close();
    
    cout << "\n" << string(80, '=') << "\n";
    cout << "D=" << D << " experiments completed!\n";
    cout << string(80, '=') << "\n\n";
}

//Main experiment runner
void runAllExperiments(bool isDC, const string& outputDir) {
    mkdir(outputDir.c_str(), 0777);
    
    int N = 40;
    int numRuns = 51;  //Mentioned in the paper
    
    //D=10 with εtol = 5% (error tolarance)
    cout << "\n********** Testing D=10 (εtol=5%) **********\n";
    runExperimentsForDimension(isDC, N, 10, numRuns, 5.0, outputDir);
    
    // D=30 with εtol = 20% (error tolarance)
    cout << "\n********** Testing D=30 (εtol=20%) **********\n";
    runExperimentsForDimension(isDC, N, 30, numRuns, 20.0, outputDir);
}

//Main function
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