#include "searcher.h"
#include "griditer.h"

FeaturePointSearcher::FeaturePointSearcher(PointCloud::ConstPtr input, int numFP,
        std::vector<float> const& adjustedGradientNorms)
    : Width_(input->width)
    , Height_(input->height)
    , Input_(input)
    , NumFP_(numFP)
    , AdjustedGradientNorms_(adjustedGradientNorms)
{
    FeaturePoints.reset(new PointCloud);
    Taken_.resize(Input_->height, std::vector<bool>(Input_->width));
}

void FeaturePointSearcher::OrderAscendingly() {
    std::vector< std::pair<float, int> > ps(Input_->size());
    for (int i = 0; i < ps.size(); ++i) {
        ps[i].first = AdjustedGradientNorms_[i];
        ps[i].second = i;
    }
    std::sort(ps.begin(), ps.end());
    std::reverse(ps.begin(), ps.end());

    SortedOrder_.resize(ps.size());
    for (int i = 0; i < ps.size(); ++i) {
        SortedOrder_[i] = ps[i].second;
    }
}

void FeaturePointSearcher::Search() {
    OrderAscendingly();

    for (int idx : SortedOrder_) {
        int const y0 = idx / Width_;
        int const x0 = idx / Height_;

        bool neighborTaken = false;
        GridRadiusTraversal grt(Height_, Width_);
        grt.TraverseRectangle(y0, x0, 10, [this, &neighborTaken, y0, x0] (int y, int x) {
                    if (Taken_[y][x]) {
                        neighborTaken = true;
                    }
                });
        if (! neighborTaken) {
            Taken_[y0][x0] = true;
            FeaturePoints->push_back(Input_->at(idx));
        }

        if (FeaturePoints->size() == NumFP_) {
            break;
        }
    }
}
