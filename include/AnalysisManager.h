#ifndef RIVER_ANALYSIS_MANAGER_H
#define RIVER_ANALYSIS_MANAGER_H

#include "Analysis.h"

class AnalysisManager {
public:
    AnalysisManager();
    ~AnalysisManager();

    static int Initalize(const char* argv0, const char* model_path);
    static void Uninitalize();

    int run();
    int LoadConfig(const char* argv, const char* config_path);

private:
    std::vector<Analysis*> inst_vec;//分析实例
    ConfigList config;
};

#endif
