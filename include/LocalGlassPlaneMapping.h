
#ifndef LOCALGLASSPLANEMAPPING_H
#define LOCALGLASSPLANEMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"
#include "GlassPlaneMappingConfig.h"

#include <mutex>
#include <vector>
#include <utility>  

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;
class Atlas;
class PointCloudMapping;

class LocalGlassPlaneMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalGlassPlaneMapping(System* pSys, Atlas* pAtlas, const float bMonocular, PointCloudMapping* pPointCloudMapping,
                           const GlassPlaneMappingConfig& config);

    // glass plane
    bool process_new_kf_glass(KeyFrame* keyfrm);
    bool create_GlassToPlane(KeyFrame* keyfrm);
    int glass_block_keyplane_fitting_RANSAC(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                                            std::vector<double>& pl_equation,
                                            double* inlier_ratio = nullptr,
                                            double* mean_inlier_dist = nullptr);
    bool add_keyplane_to_map(KeyFrame* keyfrm, const std::vector<double>& pl_equation);
    bool add_keyplane_to_map(KeyFrame* keyfrm, keyplane* keypln);
    
public:
    Atlas* mpAtlas;
    const float mbMonocular;
    System* mpSys;
    PointCloudMapping* mpPointCloudMapping;
private:
    struct PlaneCandidateState
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // 平面参数
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        double offset = 0.0;
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        bool has_centroid = false;
        // 连续关键帧计数
        int consecutive = 0;
        // 用来判断是否连续
        long unsigned int last_kf_id = 0;
        // 保存观测
        std::vector<std::pair<KeyFrame*, unsigned int>> observations;
    };
    // 清空全部候选
    void ResetPlaneCandidate();
    // 新建一个候选
    void StartPlaneCandidate(const Eigen::Vector3d& n_w, double d_w, const Eigen::Vector3d& centroid_w,
                             bool has_centroid_w, KeyFrame* keyfrm, unsigned int keypln_idx);
    // 判断是否是同一个平面，基于角度和距离
    
    // 删除过期的候选。即不连续的候选
    void PruneStalePlaneCandidates(long unsigned int current_kf_id);

    bool checkPlaneMatch(const Eigen::Vector3d& ncurrent, double dcurrent, const Eigen::Vector3d& centroid_current,
                         bool has_centroid_current, const PlaneCandidateState& candidate) const;
    bool checkPlaneMatch(const Vec3_t &ncurrent, const double dcurrent, const Eigen::Vector3d& centroid_current,
                         bool has_centroid_current, const Plane* plane_lm);
    bool checkPlaneMatch(const keyplane* pln1, const keyplane* pln2);
    

    void AttachNearbyMapPoints(Plane* plane, KeyFrame* keyfrm);

    std::vector<PlaneCandidateState, Eigen::aligned_allocator<PlaneCandidateState>> mPlaneCandidates;
    GlassPlaneMappingConfig mConfig;
};


} //namespace ORB_SLAM

#endif // LOCALGLASSPLANEMAPPING_H
