#include "gridstrategy.h"
#include "griditer.h"

void GridNeighbourModificationStrategy::InitializeFor(SVM3D * parent) {
    IGradientModificationStrategy::InitializeFor(parent);

    QValues_.resize(parent->PointCount());
    Neighbors_.resize(parent->PointCount());
    LastAccess_.resize(parent->PointCount(), -1);
}

void GridNeighbourModificationStrategy::OptimizePivots(int * i, int * j) {
    float best = Parent()->PivotsOptimality(*i, *j);
    int const startJ = *j;

    InitializeNeighbors(*i);
    std::vector<int> const& nbh = Neighbors_[*i];
    std::vector<float> const& qvls = QValues_[*i];

    for (int k = 0; k < nbh.size(); ++k) {
        float const current = Parent()->PivotsOptimality(*i, nbh[k], qvls[k]);
        if (current > best) {
            best = current;
            *j = nbh[k];
        }
    }
    assert(*j!= *i && 0 <= *j && *j < Num2Grid_.size());

    NumOptimizeFailures += startJ == *j;
}

void GridNeighbourModificationStrategy::ReflectAlphaChange(int idx, SVMFloat delta) {
    InitializeNeighbors(idx);
    std::vector<int> const& nbh = Neighbors_[idx];
    std::vector<float> const& qv = QValues_[idx];

    for (int k = 0; k < nbh.size(); ++k) {
        ModifyGradient(nbh[k], qv[k] * delta);
    }
}

void GridNeighbourModificationStrategy::InitializeNeighbors(int idx) {
    bool const firstTime = LastAccess_[idx] == -1;
    LogAccess(idx);

    std::vector<int> & nbh = Neighbors_[idx];
    std::vector<float> & qvls = QValues_[idx];

    if (nbh.size()) {
        return;
    }
    if (! firstTime) {
        NumCacheMisses++;
    }

    int const y = Num2Grid_.at(idx).first;
    int const x = Num2Grid_.at(idx).second;
    // always +5 to for feeling safe...
    float const rawIndex = Pixel2RawIndex_.at(y * GridWidth_ + x);
    float const kernelRadiusInPixels = (int)ceil(KernelRadius_ / LocalResolution_.at(rawIndex)) + 5;

    Grid2Num_.TraverseRectangle(y, x, kernelRadiusInPixels,
            [this, &idx, &nbh, &qvls] (int /*y*/, int /*x*/, int nbhIdx) {
                float const dist2 = Parent()->Dist2(idx, nbhIdx);
                if (dist2 <= KernelRadius2_) {
                    nbh.push_back(nbhIdx);
                    qvls.push_back(Parent()->QValue(idx, nbhIdx, dist2));
                }
            });

    NumNeighborsCalculations++;
    TotalNeighborsProcessed += nbh.size();

    RepackNeighbors(idx);
    RegisterNewNeighbors(nbh.size());
}

void GridNeighbourModificationStrategy::LogAccess(int idx) {
    History_.push_back(idx);
    LastAccess_[idx] = History_.size() - 1;
}

void GridNeighbourModificationStrategy::RegisterNewNeighbors(int num) {
    TotalNeighbours_ += num;
    while (TotalNeighbours_ > MaxTotalNeighbors_) {
        int const pointIdx = History_[HistoryIndex_];
        if (LastAccess_[pointIdx] == HistoryIndex_) {
            TotalNeighbours_ -= Neighbors_[pointIdx].size();
            FreeNeighbors(pointIdx);
        }
        HistoryIndex_++;
    }
}

void GridNeighbourModificationStrategy::RepackNeighbors(int idx) {
    std::vector<int> tmpInts(Neighbors_[idx]);
    std::vector<float> tmpFloats(QValues_[idx]);
    Neighbors_[idx].swap(tmpInts);
    QValues_[idx].swap(tmpFloats);
}

void GridNeighbourModificationStrategy::FreeNeighbors(int idx) {
    std::vector<int> tmpInts;
    std::vector<float> tmpFloats;
    Neighbors_[idx].swap(tmpInts);
    QValues_[idx].swap(tmpFloats);
}

float GridNeighbourModificationStrategy::QValue(int i, int j) {
    // there is no need for a speed here
    auto cellI = Num2Grid_.at(i);
    auto cellJ = Num2Grid_.at(j);
    if (Parent()->Dist2(i, j) <= KernelRadius2_) {
        return Parent()->QValue(i, j);
    }
    return 0.0;
}
