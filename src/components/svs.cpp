#include "svs.h"
#include "trainset.h"
#include "analysis.h"

#include "pcl/features/integral_image_normal.h"
#include "pcl/common/time.h"
#include "pcl/filters/filter.h"

#include <fstream>

void SupportVectorShape::LoadAsText(std::istream & istr) {
    float x, y, z;
    while (istr >> x >> y >> z) {
        FeaturePoints_->push_back(createPoint<PointType>(x, y, z));
    }
}

void SupportVectorShape::SaveAsText(std::ostream & ostr) {
    for (int i = 0; i < FeaturePoints_->size(); ++i) {
        PointType const& p = FeaturePoints_->at(i);
        ostr << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    ostr.flush();
}

void BaseBuilder::SetInputCloud(PointCloud::ConstPtr input) {
    Input = input;

    Height_ = Input->height;
    Width_ = Input->width;

    std::vector<int> tmp;
    InputNoNan.reset(new PointCloud);
    for (int i = 0; i < Input->size(); ++i) {
        if (! pointHasNan(Input->at(i))) {
            InputNoNan->push_back(Input->at(i));
        }
    }
    CalcIndicesInOriginal();

    InputKDTree_.reset(new pcl::search::KdTree<PointType>);
    InputKDTree_->setInputCloud(InputNoNan);
    Resolution = computeCloudResolution(InputNoNan, *InputKDTree_);
}

void BaseBuilder::CalcIndicesInOriginal() {
    assert(RawIndex2Pixel.empty());
    Pixel2RawIndex.resize(Input->size(), -1);
    for (int i = 0; i < Input->size(); ++i) {
        if (pointHasNan(Input->at(i))) {
            continue;
        }
        Pixel2RawIndex[i] = RawIndex2Pixel.size();
        RawIndex2Pixel.push_back(i);
    }
}

void BaseBuilder::CalcDistanceToNN() {
    if (DistToNN.size()) {
        return;
    }

    std::vector<int> indices;
    std::vector<float> dist2;
    DistToNN.resize(InputNoNan->size());
    for (int i = 0; i < InputNoNan->size(); ++i) {
        InputKDTree_->nearestKSearch(i, 2, indices, dist2);
        DistToNN[i] = sqrt(dist2[1]);
    }

    LocalResolution.resize(InputNoNan->size());
    for (int i = 0; i < InputNoNan->size(); ++i) {
        int const pixelIdx = RawIndex2Pixel[i];
        int const x0 = pixelIdx % Width_;
        int const y0 = pixelIdx / Width_;

        int nbhCount = 0;
        GridRadiusTraversal grt(Height_, Width_);
        grt.TraverseRectangle(y0, x0, 5,
                [this, i, &nbhCount] (int y, int x) {
                    int const nbh = Pixel2RawIndex[y * Width_ + x];
                    if (nbh != -1) {
                        LocalResolution[i] += DistToNN[nbh];
                        nbhCount++;
                    }
                    return true;
                });

        LocalResolution[i] /= nbhCount;
    }
}

void SVSBuilder::SetParams(const SVSParams &params) {
    Params_ = params;

    // at the distance SupportSize / 2 the kernel value must be 1e-2
    Gamma = -4 * log(Params_.SupportMinKernel) / sqr(Params_.SupportSize);
    std::cout << "Gamma: " << Gamma << '\n';

    // at the distance sqrt(Radius2) the kernel value must be KernelThreshold
    KernelRadius2 = -log(Params_.KernelThreshold) / Gamma;

    KernelRadius = sqrt(KernelRadius2);
    std::cout << "KernelRadius: " << KernelRadius << '\n';
}

void SVSBuilder::GenerateTrainingSet() {
    CalcDistanceToNN();

    srand(Params_.Seed);
    TrainingSetGenerator tsg(
            1, // temporary hack
            Params_.TakeProb,
            Params_.StepWidth);
    if (Params_.UseNormals) {
        CalcNormals();
        tsg.GenerateUsingNormals(*Input, *Normals, Params_.SupportSize);
    } else {
        tsg.GenerateFromSensor(*Input, LocalResolution);
    }

    Objects.reset(new PointCloud(tsg.Objects));
    Labels = tsg.Labels;
    Num2Grid_ = tsg.Num2Grid;
    Grid2Num_ = tsg.Grid2Num;
    Pixel2TrainNum = tsg.Pixel2Num;
}

void SVSBuilder::CalcNormals() {
    if (Normals.get()) {
        return;
    }
    Normals.reset(new NormalCloud);

    pcl::IntegralImageNormalEstimation<PointType, NormalType> ne;
    ne.setNormalSmoothingSize(Params_.SmoothingRange);
    ne.setInputCloud(Input);
    ne.compute(*Normals);
}

void SVSBuilder::SetInputCloud(PointCloud::ConstPtr input) {
    BaseBuilder::SetInputCloud(input);
}

void SVSBuilder::InitSVM(std::vector<SVMFloat> const& alphas) {
    assert(Objects->size() == alphas.size());
    SVM_.Init(*Objects, Labels, alphas);
    BuildGrid2SV();
}

void SVSBuilder::Learn() {
    if (Params_.UseGrid) {
        Strategy.reset(new GridNeighbourModificationStrategy(
                    Input->height, Input->width,
                    Grid2Num_, Num2Grid_,
                    KernelRadius,
                    Pixel2RawIndex,
                    LocalResolution,
                    Params_.CacheSize));
        SVM_.SetStrategy(Strategy);
    }
    SVM_.SetParams(Params_.MaxAlpha, Gamma, Params_.TerminateEps);

    {
        pcl::ScopeTime st("SVM");
        SVM_.Train(*Objects, Labels);
    }

    BuildGrid2SV();
}

void SVSBuilder::CalcGradients() {
    if (GradientNorms.size()) {
        return;
    }
    pcl::ScopeTime st("CalcGradientNorms");

    Gradients.reset(new NormalCloud);
    Gradients->points.resize(Input->size(), toNormal(nanPoint()));
    NumCloseSV.resize(InputNoNan->size());
    GradientNorms.resize(InputNoNan->size());
    AdjustedGradientNorms.resize(InputNoNan->size());

    DecisionFunction df;
    for (int i = 0; i < InputNoNan->size(); ++i) {
        PointType const& point = InputNoNan->at(i);
        int const pixel = RawIndex2Pixel.at(i);
        int const y = pixel / Width_;
        int const x = pixel % Width_;
        BuildDF(y, x, &df);
        auto const grad = df.Gradient(point);
        Gradients->at(RawIndex2Pixel[i]) = toNormal(grad);
        NumCloseSV[i] = df.SVCount();
        GradientNorms[i] = grad.getVector3fMap().norm();
        if (Params_.DoNormalizeGradient) {
            AdjustedGradientNorms[i] = NumCloseSV[i] ? GradientNorms[i] / NumCloseSV[i] : 0;
        } else {
            AdjustedGradientNorms[i] = GradientNorms[i];
        }
    }
}

void SVSBuilder::BuildGrid2SV() {
    Grid2SV_.Resize(Height_, Width_);
    for (int i = 0; i < Objects->size(); ++i) {
        if (SVM().Alphas()[i] > 0) {
            auto pos = Num2Grid_[i];
            Grid2SV_.at(pos.first, pos.second).push_back(i);
        }
    }
}

void SVSBuilder::BuildDF(int y, int x, DecisionFunction * df) {
    df->Reset(Gamma, SVM().Rho);
    PointType const& point = Input->at(x, y);

    float const rawIndex = Pixel2RawIndex[y * Width_ + x];
    // always +5 to for feeling safe...
    float const kernelRadiusInPixels = (int)ceil(KernelRadius / LocalResolution[rawIndex]) + 5;

    Grid2SV_.TraverseRectangle(y, x, kernelRadiusInPixels,
            [this, &point, &df] (int /*y*/, int /*x*/, int idx) {
                PointType const& sv = Objects->operator[](idx);
                if (pcl::squaredEuclideanDistance(point, sv) <= KernelRadius2) {
                    df->AddSupportVector(sv, SVM().Alphas()[idx]);
                }
                return true;
            });
}

void SVSBuilder::ToImageLayout(std::vector<float> const& data, std::vector<float> * res) {
    res->resize(Input->size(), -1);
    for (int i = 0; i < data.size(); ++i) {
        res->at(RawIndex2Pixel[i]) = data[i];
    }
}

void SVSBuilder::FeaturePointSearch() {
    BuildFPOrder();

    FeaturePoints.reset(new PointCloud);

    for (int idx : FPOrder_) {
        PointType const& point = InputNoNan->at(idx);

        int const pixelIdx = RawIndex2Pixel[idx];
        int const x0 = pixelIdx % Width_;
        int const y0 = pixelIdx / Width_;

        float const minSpace = Params_.SupportSize * Params_.FPSpace;
        float const minSpace2 = sqr(minSpace);
        int const checkInPixels = (int)ceil(minSpace / LocalResolution[idx]);

        bool neighborBetter = false;
        GridRadiusTraversal grt(Height_, Width_);
        grt.TraverseRectangle(y0, x0, checkInPixels,
                [this, &neighborBetter, idx, point, y0, x0, minSpace2] (int y, int x) {
                    int const rawIndex = Pixel2RawIndex[y * Width_ + x];
                    PointType const& neighbor = Input->at(x, y);
                    if (rawIndex >= 0 &&
                        pcl::squaredEuclideanDistance(point, neighbor) < minSpace2 &&
                        GradientNorms[rawIndex] > GradientNorms[idx])
                    {
                        neighborBetter = true;
                        return false;
                    }
                    return true;
                });
        if (! neighborBetter) {
            FeaturePoints->push_back(InputNoNan->at(idx));
        }
        if (FeaturePoints->size() == Params_.NumFP) {
            break;
        }
    }
}

void SVSBuilder::BuildFPOrder() {
    std::vector< std::pair<float, int> > ps(InputNoNan->size());
    for (int i = 0; i < ps.size(); ++i) {
        ps[i].first = AdjustedGradientNorms.at(i);
        ps[i].second = i;
    }
    std::sort(ps.begin(), ps.end());
    std::reverse(ps.begin(), ps.end());

    FPOrder_.resize(ps.size());
    for (int i = 0; i < ps.size(); ++i) {
        FPOrder_[i] = ps[i].second;
    }
}
