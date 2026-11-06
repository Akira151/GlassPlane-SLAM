/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Optimizer_plane.h"

#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"
#include "landmark_plane.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>

#include "OptimizableTypes.h"
#include "OptimizableTypes_plane.h"

namespace ORB_SLAM3{

namespace {

bool IsValidPlaneCoeffs(const Eigen::Vector4d &coeffs)
{
    return coeffs.allFinite() && coeffs.head<3>().norm() > 1e-6;
}

bool IsFiniteChi2(double chi2)
{
    return std::isfinite(chi2);
}

bool PlaneBaDebugEnabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv("ORB_SLAM3_PLANE_BA_DEBUG");
        if (!env)
            return false;
        const std::string value(env);
        return value != "0" && value != "false" && value != "FALSE";
    }();
    return enabled;
}

int PlaneBaDebugEvery()
{
    static const int every = []() {
        const char *env = std::getenv("ORB_SLAM3_PLANE_BA_DEBUG_EVERY");
        if (!env)
            return 1;
        const int value = std::atoi(env);
        return value > 0 ? value : 1;
    }();
    return every;
}

bool ShouldPrintPlaneBaDebug(const unsigned long kfId, const size_t nPlaneEdges)
{
    return PlaneBaDebugEnabled() && nPlaneEdges > 0 && (kfId % PlaneBaDebugEvery() == 0);
}

double ReadPositiveEnvDouble(const char *name)
{
    const char *env = std::getenv(name);
    if (!env)
        return -1.0;
    const double value = std::atof(env);
    return value > 0.0 ? value : -1.0;
}

double PlaneInfoWeightOverride()
{
    static const double weight = ReadPositiveEnvDouble("ORB_SLAM3_PLANE_INFO_WEIGHT");
    return weight;
}

double ComputePlaneInfoWeight(const size_t nPointEdges, const size_t nPlaneEdges)
{
    const double overrideWeight = PlaneInfoWeightOverride();
    if (overrideWeight > 0.0)
        return overrideWeight;

    if (nPointEdges == 0 || nPlaneEdges == 0)
        return 1.0;

    const double edgeRatio = static_cast<double>(nPointEdges) / static_cast<double>(nPlaneEdges);
    // const double weight = 50.0 * edgeRatio;
    const double weight = 1000;
    std::cout<<"dynamic plane edge weight: "<<50.0 * edgeRatio<<std::endl;
    return std::min(20000.0, std::max(100.0, weight));
}

void ConfigurePlaneEdges(std::vector<ORB_SLAM3::reproj_edge_plane3d *> &edges,
                         const double planeInfoWeight,
                         const double huberDelta)
{
    const double safeWeight = planeInfoWeight > 0.0 ? planeInfoWeight : 1.0;
    const Eigen::Matrix3d planeInformation = Eigen::Matrix3d::Identity() * safeWeight;
    const double weightedHuberDelta = huberDelta * std::sqrt(safeWeight);

    for (ORB_SLAM3::reproj_edge_plane3d *edge : edges)
    {
        if (!edge)
            continue;
        edge->setInformation(planeInformation);
        if (edge->robustKernel())
            edge->robustKernel()->setDelta(weightedHuberDelta);
    }
}

double RawPlaneChi2(const ORB_SLAM3::reproj_edge_plane3d *edge)
{
    if (!edge)
        return std::numeric_limits<double>::infinity();
    const double rawChi2 = edge->error().squaredNorm();
    return std::isfinite(rawChi2) ? rawChi2 : std::numeric_limits<double>::infinity();
}

struct EdgeContributionStats
{
    size_t count = 0;
    size_t finite = 0;
    size_t outliers = 0;
    size_t badDepth = 0;
    double chi2 = 0.0;
    double robustChi2 = 0.0;
    double rawError2 = 0.0;
    double infoTrace = 0.0;
    double poseHessianTrace = 0.0;
    double maxChi2 = 0.0;
};

void AddStats(EdgeContributionStats &dst, const EdgeContributionStats &src)
{
    dst.count += src.count;
    dst.finite += src.finite;
    dst.outliers += src.outliers;
    dst.badDepth += src.badDepth;
    dst.chi2 += src.chi2;
    dst.robustChi2 += src.robustChi2;
    dst.rawError2 += src.rawError2;
    dst.infoTrace += src.infoTrace;
    dst.poseHessianTrace += src.poseHessianTrace;
    dst.maxChi2 = std::max(dst.maxChi2, src.maxChi2);
}

template <typename EdgeT>
bool IsDepthPositiveForStats(EdgeT *)
{
    return true;
}

bool IsDepthPositiveForStats(ORB_SLAM3::EdgeSE3ProjectXYZ *edge)
{
    return edge->isDepthPositive();
}

bool IsDepthPositiveForStats(ORB_SLAM3::EdgeSE3ProjectXYZToBody *edge)
{
    return edge->isDepthPositive();
}

bool IsDepthPositiveForStats(g2o::EdgeStereoSE3ProjectXYZ *edge)
{
    return edge->isDepthPositive();
}

template <typename EdgeT>
double RobustChi2ForStats(const EdgeT *edge, const double chi2)
{
    if (!std::isfinite(chi2))
        return 0.0;
    if (!edge->robustKernel())
        return chi2;

    Eigen::Vector3d rho;
    edge->robustKernel()->robustify(chi2, rho);
    return std::isfinite(rho[0]) ? rho[0] : 0.0;
}

template <typename EdgeT>
double RobustScaleForStats(const EdgeT *edge, const double chi2)
{
    if (!std::isfinite(chi2) || !edge->robustKernel())
        return 1.0;

    Eigen::Vector3d rho;
    edge->robustKernel()->robustify(chi2, rho);
    return std::isfinite(rho[1]) ? rho[1] : 1.0;
}

template <typename EdgeT>
double EstimatePoseHessianTrace(EdgeT *edge, const int poseVertexIndex)
{
    if (!edge || poseVertexIndex < 0 || poseVertexIndex >= static_cast<int>(edge->vertices().size()))
        return 0.0;

    g2o::OptimizableGraph::Vertex *poseVertex =
        dynamic_cast<g2o::OptimizableGraph::Vertex *>(edge->vertex(poseVertexIndex));
    if (!poseVertex || poseVertex->fixed() || poseVertex->dimension() != 6)
        return 0.0;

    const int errorDim = edge->dimension();
    const int poseDim = poseVertex->dimension();
    const double delta = 1e-6;
    Eigen::MatrixXd jacobian(errorDim, poseDim);
    std::vector<double> update(poseDim, 0.0);

    for (int d = 0; d < poseDim; ++d)
    {
        poseVertex->push();
        update[d] = delta;
        poseVertex->oplus(update.data());
        edge->computeError();
        const Eigen::Map<const Eigen::VectorXd> errorPlus(edge->errorData(), errorDim);
        const Eigen::VectorXd plus = errorPlus;
        poseVertex->pop();

        poseVertex->push();
        update[d] = -delta;
        poseVertex->oplus(update.data());
        edge->computeError();
        const Eigen::Map<const Eigen::VectorXd> errorMinus(edge->errorData(), errorDim);
        const Eigen::VectorXd minus = errorMinus;
        poseVertex->pop();

        jacobian.col(d) = (plus - minus) / (2.0 * delta);
        update[d] = 0.0;
    }

    edge->computeError();
    if (!jacobian.allFinite())
        return 0.0;

    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> info(
        edge->informationData(), errorDim, errorDim);
    const double robustScale = RobustScaleForStats(edge, edge->chi2());
    const double trace = (jacobian.transpose() * (robustScale * info) * jacobian).trace();
    return std::isfinite(trace) ? trace : 0.0;
}

template <typename EdgeT>
EdgeContributionStats CollectEdgeStats(const std::vector<EdgeT *> &edges, const double chi2Threshold, const int poseVertexIndex)
{
    EdgeContributionStats stats;
    for (EdgeT *edge : edges)
    {
        if (!edge)
            continue;

        ++stats.count;
        edge->computeError();

        const double chi2 = edge->chi2();
        const double rawError2 = edge->error().squaredNorm();
        const bool finite = std::isfinite(chi2) && std::isfinite(rawError2);
        const bool depthPositive = IsDepthPositiveForStats(edge);

        if (!depthPositive)
            ++stats.badDepth;
        if (!finite || chi2 > chi2Threshold || !depthPositive)
            ++stats.outliers;

        if (!finite)
            continue;

        ++stats.finite;
        stats.chi2 += chi2;
        stats.robustChi2 += RobustChi2ForStats(edge, chi2);
        stats.rawError2 += rawError2;
        stats.infoTrace += edge->information().trace();
        stats.poseHessianTrace += EstimatePoseHessianTrace(edge, poseVertexIndex);
        stats.maxChi2 = std::max(stats.maxChi2, chi2);
    }
    return stats;
}

double SafeRatio(const double numerator, const double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

void PrintGroupStats(const char *label, const EdgeContributionStats &stats, const double totalRobustChi2, const double totalPoseHessianTrace)
{
    const double denom = static_cast<double>(std::max<size_t>(stats.finite, 1));
    std::cout << " " << label
              << " edges=" << stats.count
              << " finite=" << stats.finite
              << " out=" << stats.outliers
              << " badDepth=" << stats.badDepth
              << " chi2=" << stats.chi2
              << " robust=" << stats.robustChi2
              << " robustPct=" << SafeRatio(stats.robustChi2, totalRobustChi2)
              << " meanChi2=" << stats.chi2 / denom
              << " meanRawNorm=" << std::sqrt(stats.rawError2 / denom)
              << " meanInfoTrace=" << stats.infoTrace / denom
              << " poseHTrace=" << stats.poseHessianTrace
              << " poseHTracePct=" << SafeRatio(stats.poseHessianTrace, totalPoseHessianTrace)
              << " maxChi2=" << stats.maxChi2 <<endl;
}

void PrintLbaDiagnostics(const char *stage,
                         const unsigned long kfId,
                         const size_t nLocalKFs,
                         const size_t nFixedKFs,
                         const size_t nLocalMapPoints,
                         const size_t nLocalPlanes,
                         const double planeInfoWeight,
                         const EdgeContributionStats &monoStats,
                         const EdgeContributionStats &bodyStats,
                         const EdgeContributionStats &stereoStats,
                         const EdgeContributionStats &planeStats)
{
    EdgeContributionStats pointStats;
    AddStats(pointStats, monoStats);
    AddStats(pointStats, bodyStats);
    AddStats(pointStats, stereoStats);

    const double totalRobustChi2 = pointStats.robustChi2 + planeStats.robustChi2;
    const double totalPoseHessianTrace = pointStats.poseHessianTrace + planeStats.poseHessianTrace;

    std::streamsize oldPrecision = std::cout.precision();
    std::ios::fmtflags oldFlags = std::cout.flags();
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "[LBA-DBG][plane][" << stage << "]"
              << " KF=" << kfId
              << " optKF=" << nLocalKFs
              << " fixedKF=" << nFixedKFs
              << " localMP=" << nLocalMapPoints
              << " localPlane=" << nLocalPlanes
              << " planeInfoWeight=" << planeInfoWeight
              << std::endl;

    std::cout << "[LBA-DBG][plane][" << stage << "]";
    PrintGroupStats("point", pointStats, totalRobustChi2, totalPoseHessianTrace);
    PrintGroupStats("plane", planeStats, totalRobustChi2, totalPoseHessianTrace);
    std::cout << std::endl;

    std::cout << "[LBA-DBG][plane][" << stage << "]";
    PrintGroupStats("mono", monoStats, totalRobustChi2, totalPoseHessianTrace);
    PrintGroupStats("body", bodyStats, totalRobustChi2, totalPoseHessianTrace);
    PrintGroupStats("stereo", stereoStats, totalRobustChi2, totalPoseHessianTrace);
    std::cout << std::endl;

    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}

} // namespace
/**************************************以下为局部地图优化**************************************************************/

/**
 * @brief Local Bundle Adjustment LocalMapping::Run() 使用，纯视觉
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @param 剩下的都是调试用的
 *
 * 总结下与ORBSLAM2的不同
 * 前面操作基本一样，但优化时2代去掉了误差大的点又进行优化了，3代只是统计但没有去掉继续优化，而后都是将误差大的点干掉
 */
void Optimizer_plane::LocalBundleAdjustment_plane(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &num_fixedKF, int &num_OptKF, int &num_MPs, int &num_edges)
{
    // 该优化函数用于LocalMapping线程的局部BA优化
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame *> lLocalKeyFrames;

    // 步骤1：将当前关键帧加入lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId; //防止这个keyfrm在当前的local BA轮次被重复添加
    Map *pCurrentMap = pKF->GetMap();

    // 步骤2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        // 记录局部优化id，该数为不断变化，数值等于局部化的关键帧的id，该id用于防止重复添加
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    // 步骤3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint *> lLocalMapPoints;
    set<MapPoint *> sNumObsMP;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        if (pKFi->mnId == pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    // 每个mappoint的mnBALocalForKF都会在当前localBA中被赋予成当前keyfrm的id，和邻接keyfrm无关。
                    // 这个变量的含义是，这个mappoint最后一次在哪个关键帧的localBA中被优化。作用仅是防止在同一轮localBA中重复添加同一个mappoint的顶点。
                    // 例子：
                    // mnBALocalForKF == 0   // 说明这个点从没参加过 Local BA
                    // mnBALocalForKF == 7   // 上一次是 KF7 的 Local BA。完全正常。当前是KF12的local BA，可以加入。
                    // mnBALocalForKF == pKF->mnId   // 已经在这一轮BA中被加入过，跳过
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
        }
    }

    // 步骤3-plus：遍历lLocalKeyFrames中关键帧，将它们观测的Plane_landmarks加入到lLocalPlaneLandmarks
    list<Plane *> lLocalPlaneLandmarks;
    // set<MapPoint *> sNumObsMP;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        std::map<Plane *, unsigned int> mpPlanes = pKFi->get_observed_planes();
        for (std::map<Plane *, unsigned int>::iterator vit = mpPlanes.begin(), vend = mpPlanes.end(); vit != vend; vit++)
        {
            Plane *pPlane = vit->first;
            if (pPlane)
                if (pPlane->is_valid() && pPlane->GetMap() == pCurrentMap)
                {
                    if (pPlane->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalPlaneLandmarks.push_back(pPlane);
                        pPlane->mnBALocalForKF = pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 步骤4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化
    // note: 实际上对Plane也应该做这一系列的操作，即基于plane查找相邻关键帧等。但是在原型机的实现过程中只能先忽略掉。
    list<KeyFrame *> lFixedCameras;
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    // 步骤4.1：相比ORBSLAM2多出了判断固定关键帧的个数，最起码要两个固定的,如果实在没有就把lLocalKeyFrames中最早的KF固定，还是不够再加上第二早的KF固定
    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    // 1.0 版本没有以下这段
    // if (num_fixedKF < 2)
    // {
    //     list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
    //     int lowerId = pKF->mnId;
    //     KeyFrame *pLowerKf;
    //     int secondLowerId = pKF->mnId;
    //     KeyFrame *pSecondLowerKF;

    //     for (; lit != lLocalKeyFrames.end(); lit++)
    //     {
    //         KeyFrame *pKFi = *lit;
    //         if (pKFi == pKF || pKFi->mnId == pMap->GetInitKFid())
    //         {
    //             continue;
    //         }

    //         if (pKFi->mnId < lowerId)
    //         {
    //             lowerId = pKFi->mnId;
    //             pLowerKf = pKFi;
    //         }
    //         else if (pKFi->mnId < secondLowerId)
    //         {
    //             secondLowerId = pKFi->mnId;
    //             pSecondLowerKF = pKFi;
    //         }
    //     }
    //     lFixedCameras.push_back(pLowerKf);
    //     lLocalKeyFrames.remove(pLowerKf);
    //     num_fixedKF++;
    //     if (num_fixedKF < 2)
    //     {
    //         lFixedCameras.push_back(pSecondLowerKF);
    //         lLocalKeyFrames.remove(pSecondLowerKF);
    //         num_fixedKF++;
    //     }
    // }

    if (num_fixedKF == 0)
    {
        std::cout << "[LBA-DBG][plane] abort num_fixedKF==0 at KF " << pKF->mnId
                  << " local_kf=" << lLocalKeyFrames.size()
                  << " local_mp=" << lLocalMapPoints.size()
                  << " local_plane=" << lLocalPlaneLandmarks.size()
                  << std::endl;
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Plane-only observations do not constrain the full 6-DoF camera pose.
    // When the local window has no point constraints, keep poses fixed and
    // only refine plane landmarks to avoid rank-deficient SE3 updates.
    const bool bFixLocalPoseVertices = lLocalMapPoints.empty();

    // Setup optimizer
    // 步骤5：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    // Set Local KeyFrame vertices
    // 步骤6：添加顶点：Pose of Local KeyFrame
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed((pKFi->mnId == pMap->GetInitKFid()) || bFixLocalPoseVertices);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed KeyFrame vertices
    // 步骤7：添加顶点：Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)。
    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }

    // Set MapPoint vertices
    // 步骤7：添加3D顶点
    // 存放的方式(举例)
    // 边id: 1 2 3 4 5 6 7 8 9
    // KFid: 1 2 3 4 1 2 3 2 3
    // MPid: 1 1 1 1 2 2 2 3 3
    // 所以这个个数大约是点数×帧数，实际肯定比这个要少
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    // 存放单目时的边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    // 存放双目鱼眼时另一个相机的边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);
    // 存放单目时的KF
    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    // 存放单目时的MP

    // 存放双目鱼眼时另一个相机的KF
    vector<KeyFrame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);
    // 存放单目时的MP
    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    // 存放双目鱼眼时另一个相机的MP
    vector<MapPoint *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    // 存放双目时的边
    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    // 存放双目时的KF
    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    // 存放双目时的MP
    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    const float thHuberPlane = sqrt(7.815);
    const double chi2Plane = 7.815;
    const double chi2PlaneInit = 50.0;

    int nPoints = 0;
    unsigned long maxMPid = 0;

    int nEdges = 0;

    // 添加顶点：MapPoint
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        if (pMP->mnId > maxMPid)
            maxMPid = pMP->mnId;
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        // 这里的边缘化与滑动窗口不同，而是为了加速稀疏矩阵的计算BlockSolver_6_3默认了6维度的不边缘化，3自由度的三维点被边缘化，所以所有三维点都设置边缘化
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        // Set edges
        //  步骤8：对每一对关联的MapPoint和KeyFrame构建边
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                // 单目
                if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] >= 0) // Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if (pKFi->mpCamera2)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }
    num_edges = nEdges;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;


    int nPlanes=0;
    const unsigned long plane_id_offset = maxKFid + maxMPid + 1000000;
    std::vector<ORB_SLAM3::reproj_edge_plane3d *> vpEdgesPlane;
    std::vector<KeyFrame *> vpEdgeKFPlane;
    std::vector<Plane *> vpPlaneLandmarkEdge;
    vpEdgesPlane.reserve(lLocalPlaneLandmarks.size() * std::max<size_t>(1, lLocalKeyFrames.size() + lFixedCameras.size()));
    vpEdgeKFPlane.reserve(vpEdgesPlane.capacity());
    vpPlaneLandmarkEdge.reserve(vpEdgesPlane.capacity());
    // 步骤8-plus：添加顶点：Plane_landmarks
    for (list<Plane *>::iterator lit = lLocalPlaneLandmarks.begin(), lend = lLocalPlaneLandmarks.end(); lit != lend; lit++)
    {
        Plane *pPlane = *lit;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double d = 0.0;
        pPlane->get_equation(a, b, c, d);
        const Eigen::Vector4d planeCoeffs(a, b, c, d);
        if (!IsValidPlaneCoeffs(planeCoeffs))
        {
            continue;
        }
        // 待做1：新建自定义plane_landmark g2o节点。
        landmark_vertex_plane *vPlane = new landmark_vertex_plane();
        vPlane->setEstimate(Plane3D(planeCoeffs));
        int id = pPlane->_id + plane_id_offset;
        vPlane->setId(id);
        // 这里的边缘化与滑动窗口不同，而是为了加速稀疏矩阵的计算BlockSolver_6_3默认了6维度的不边缘化，3自由度的三维点被边缘化，所以所有三维点都设置边缘化
        // marginalized的初衷当然是为了速度。但是如果这里不set true就会引发问题。
        // 在使用BlockSolver_6_3的情况下，所有landmark都必须被marginalized，不然就会被当成是6*6的pose来优化。
        vPlane->setMarginalized(true);
        optimizer.addVertex(vPlane);
        nPlanes++;

        const map<KeyFrame *, unsigned int> observations = pPlane->get_observations();

        // Set edges
        // 对每一个能够观测到这个3d plane landmark pPlane的keyfrm，创建keyfrm和pPlane之间的边
        for (map<KeyFrame *, unsigned int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                // keypln和keyfrm的关系由pPlane中的observations_保存，keypln的下标是由add_observation的时候定义的。
                const int keypln_index = mit->second;

                // Monocular observation
                // 单目
                if (keypln_index != -1)
                {
                    if (keypln_index < 0 || keypln_index >= static_cast<int>(pKFi->keyplns_.size()))
                    {
                        continue;
                    }
                    const keyplane *keypln = pKFi->keyplns_[keypln_index];
                    if (!keypln)
                    {
                        continue;
                    }
                    // 待做2：设置observation的格式并将其加入到vertex中。
                    // 这个observation是指landmark_plane在camera坐标系中的obs。即keyplane的平面参数。
                    // obs和measurement都是Vec4_t类型。这是在reproj_edge_plane3d的基类模板中定死的，与顶点类型无关。
                    Eigen::Vector4d obs;
                    keypln->get_equation(obs);
                    if (!obs.allFinite() || obs.head<3>().norm() < 1e-6)
                    {
                        continue;
                    }
                    

                    // 待做3：定义属于面的边
                    ORB_SLAM3::reproj_edge_plane3d *e = new ORB_SLAM3::reproj_edge_plane3d();

                    // 按照id来定义这条边的两个顶点。
                    // 顶点1是
                    g2o::OptimizableGraph::Vertex *vKF =
                        dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
                    // 顶点2是plane_lm（estimate)。
                    g2o::OptimizableGraph::Vertex *vPL =
                        dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id));
                    if (!vKF || !vPL)
                    {
                        continue;
                    }
                    e->setVertex(0, vKF);
                    e->setVertex(1, vPL);
                    
                    // 待做2
                    e->setMeasurement(obs);
                    // const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    // e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    // Keep construction and initial gating in the unweighted residual scale.
                    e->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberPlane);

                    // 平面投影并不涉及特定的相机，所以暂时注释掉。
                    // e->pCamera = pKFi->mpCamera;

                    e->computeError();
                    const double initChi2 = RawPlaneChi2(e);
                    if (!IsFiniteChi2(initChi2) || initChi2 > chi2PlaneInit)
                    {
                        delete e;
                        continue;
                    }

                    optimizer.addEdge(e);
                    vpEdgesPlane.push_back(e);
                    vpEdgeKFPlane.push_back(pKFi);
                    vpPlaneLandmarkEdge.push_back(pPlane);

                    nEdges++;
                }
                
            }
        }
    }
    num_edges = nEdges;

    const size_t nPointEdges = vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size();
    const double planeInfoWeight = ComputePlaneInfoWeight(nPointEdges, vpEdgesPlane.size());
    ConfigurePlaneEdges(vpEdgesPlane, planeInfoWeight, thHuberPlane);

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    if (nEdges == 0)
        return;

    const bool bPrintPlaneBaDebug = ShouldPrintPlaneBaDebug(pKF->mnId, vpEdgesPlane.size());
    if (bPrintPlaneBaDebug)
    {
        const EdgeContributionStats monoStats =
            CollectEdgeStats(vpEdgesMono, 5.991, 1);
        const EdgeContributionStats bodyStats =
            CollectEdgeStats(vpEdgesBody, 5.991, 1);
        const EdgeContributionStats stereoStats =
            CollectEdgeStats(vpEdgesStereo, 7.815, 1);
        const EdgeContributionStats planeStats =
            CollectEdgeStats(vpEdgesPlane, chi2Plane * planeInfoWeight, 0);
        PrintLbaDiagnostics("pre",
                            pKF->mnId,
                            lLocalKeyFrames.size(),
                            lFixedCameras.size(),
                            lLocalMapPoints.size(),
                            lLocalPlaneLandmarks.size(),
                            planeInfoWeight,
                            monoStats,
                            bodyStats,
                            stereoStats,
                            planeStats);
    }

    // 步骤9：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    if (bPrintPlaneBaDebug)
    {
        const EdgeContributionStats monoStats =
            CollectEdgeStats(vpEdgesMono, 5.991, 1);
        const EdgeContributionStats bodyStats =
            CollectEdgeStats(vpEdgesBody, 5.991, 1);
        const EdgeContributionStats stereoStats =
            CollectEdgeStats(vpEdgesStereo, 7.815, 1);
        const EdgeContributionStats planeStats =
            CollectEdgeStats(vpEdgesPlane, chi2Plane * planeInfoWeight, 0);
        PrintLbaDiagnostics("post",
                            pKF->mnId,
                            lLocalKeyFrames.size(),
                            lFixedCameras.size(),
                            lLocalMapPoints.size(),
                            lLocalPlaneLandmarks.size(),
                            planeInfoWeight,
                            monoStats,
                            bodyStats,
                            stereoStats,
                            planeStats);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());
    vector<pair<KeyFrame *, Plane *>> vToErasePlane;
    vToErasePlane.reserve(vpEdgesPlane.size());

    // Check inlier observations
    // 步骤10：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
        MapPoint *pMP = vpMapPointEdgeBody[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++)
    {
        ORB_SLAM3::reproj_edge_plane3d *e = vpEdgesPlane[i];
        Plane *pPlane = vpPlaneLandmarkEdge[i];

        if (!pPlane || !pPlane->is_valid())
            continue;

        e->computeError();
        const double chi2 = RawPlaneChi2(e);
        if (!IsFiniteChi2(chi2) || chi2 > chi2Plane)
        {
            KeyFrame *pKFi = vpEdgeKFPlane[i];
            vToErasePlane.push_back(make_pair(pKFi, pPlane));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if (!vToErasePlane.empty())
    {
        for (size_t i = 0; i < vToErasePlane.size(); i++)
        {
            KeyFrame *pKFi = vToErasePlane[i].first;
            Plane *pPlane = vToErasePlane[i].second;
            if (!pKFi || !pPlane)
                continue;
            pKFi->EraseObservedPlane(pPlane);
            pPlane->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // LocalBA的最终输出：改变keyfrm.pose和mappoints.worldPos的值。所以新建一个独立的plane optimizer是可能的。
    // Keyframes
    bool bShowStats = false;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
    }

    // Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // Plane landmarks
    for (list<Plane *>::iterator lit = lLocalPlaneLandmarks.begin(), lend = lLocalPlaneLandmarks.end(); lit != lend; lit++)
    {
        Plane *pPlane = *lit;
        if (!pPlane || !pPlane->is_valid() || pPlane->ObservationCount() == 0)
            continue;
        landmark_vertex_plane *vPlane = static_cast<landmark_vertex_plane *>(optimizer.vertex(pPlane->_id + plane_id_offset));
        if (!vPlane)
            continue;
        Eigen::Vector4d coeffs = vPlane->estimate().toVector();
        pPlane->set_equation(coeffs(0), coeffs(1), coeffs(2), coeffs(3));
    }

    pMap->IncreaseChangeIndex();
}

} // end namespace ORB_SLAM3
