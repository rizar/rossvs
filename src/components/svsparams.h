#pragma once

#include <cstdlib>

struct SVSParams {
    int Seed = 1;

    float MaxAlpha = 32;
    float SupportSize = 0.1;
    float SupportMinKernel = 1e-2;
    float KernelThreshold = 1e-3;
    float TerminateEps = 1e-2;

    float SmoothingRange = 5;
    float StepWidth = 0.1; // in support sizes

    float TakeProb = 1.0;

    int NumFP = 100;
    float FPSpace = 0.5; // in support sizes

    size_t CacheSize = 1 << 30;

    bool UseGrid = true;
    bool UseNormals = true;

    void Save(const char* path);
    void Load(const char* path);
};

