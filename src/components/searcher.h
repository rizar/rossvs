#pragma once

#include "common.h"

class FeaturePointSearcher {
public:
    FeaturePointSearcher(PointCloud::ConstPtr input, int numFP,
                         std::vector<float> const& adjustedGradientNorms);

    void Search();

private:
    void OrderAscendingly();

private:
    int const Width_;
    int const Height_;

    PointCloud::ConstPtr Input_;
    int NumFP_;
    std::vector<float> AdjustedGradientNorms_; // pixel indices

    std::vector<int> SortedOrder_;
    std::vector< std::vector<bool > > Taken_;

public:
    PointCloud::Ptr FeaturePoints;
};
