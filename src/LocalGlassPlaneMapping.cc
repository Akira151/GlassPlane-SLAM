

#include "LocalGlassPlaneMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include "keyplane.h"
#include "landmark_plane.h"
#include "pointcloudmapping.h"

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>   // 必须
#include <pcl/filters/extract_indices.h>         // 如果要剔除平面点
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include<mutex>
#include<chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <Eigen/Eigenvalues>

namespace ORB_SLAM3
{
    struct DSU_plane_candidate{
        vector<int> p, sz;

        DSU_plane_candidate(int size) : p(size), sz(size,1){
            for(int i=0; i<p.size(); i++){
                p[i]=i;
            }
        }
        int find(int x){
            if(p[x]==x) return x;
            else{
                p[x]=find(p[x]);
                return p[x];
            }
        }

        bool unite(int a, int b){
            int ra = find(a);
            int rb = find(b);
            if(ra == rb) return false;
            if(sz[ra] < sz[rb])  swap(ra, rb);
            p[rb]=ra;
            sz[ra]+=sz[rb];
            return true;
        }

        std::unordered_map<int, std::vector<int>> output_clusters(){
            // 聚类完成。遍历并查集的p，输出结果
            std::unordered_map<int, std::vector<int>> keypln_clusters;
            for (int i=0; i<p.size(); i++){
                int root=find(i);
                keypln_clusters[root].push_back(i);
            }
            return keypln_clusters;
        }
        
    };
    int glass_pointcloud_generate(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, KeyFrame* keyfrm, int glass_mask_threshold);
    int glass_block_pointcloud_generate(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, KeyFrame* keyfrm, cv::Mat depth_GS_img, int glass_mask_threshold);
    bool fit_plane_from_cluster(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                                const pcl::PointIndices& cluster_indices,
                                Eigen::Vector3d& normal,
                                double& offset,
                                double& mean_dist);
    float read_depth_in_meters(const cv::Mat& depth_img, int y, int x);
    bool compute_cloud_centroid(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, Vec3_t& centroid);


    Vec3_t coortrans_pix2cam(int y, int x, float depth, Eigen::Matrix3f K);

    float read_depth_in_meters(const cv::Mat& depth_img, int y, int x)
    {
        if (depth_img.empty() || y < 0 || y >= depth_img.rows || x < 0 || x >= depth_img.cols)
        {
            return 0.0f;
        }

        float depth_m = 0.0f;
        switch (depth_img.type())
        {
        case CV_32F:
            depth_m = depth_img.at<float>(y, x);
            break;
        case CV_64F:
            depth_m = static_cast<float>(depth_img.at<double>(y, x));
            break;
        case CV_16U:
            depth_m = static_cast<float>(depth_img.at<uint16_t>(y, x)) / 1000.0f;
            break;
        default:
            return 0.0f;
        }

        if (!std::isfinite(depth_m) || depth_m <= 0.0f)
        {
            return 0.0f;
        }
        return depth_m;
    }

    bool compute_cloud_centroid(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, Vec3_t& centroid)
    {
        if (!cloud || cloud->points.empty())
        {
            return false;
        }

        centroid = Vec3_t::Zero();
        size_t valid_count = 0;
        for (const pcl::PointXYZRGBA& pt : cloud->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            {
                continue;
            }
            centroid += Vec3_t(pt.x, pt.y, pt.z);
            ++valid_count;
        }

        if (valid_count < 3)
        {
            return false;
        }

        centroid /= static_cast<double>(valid_count);
        return centroid.allFinite();
    }

    bool has_valid_centroid(const Vec3_t& centroid)
    {
        return centroid.allFinite() && centroid.squaredNorm() > 1e-8;
    }

    double point_plane_distance(const Vec3_t& normal, const double offset, const Vec3_t& point)
    {
        const double normal_norm = normal.norm();
        if (!normal.allFinite() || !point.allFinite() || !std::isfinite(offset) || normal_norm <= 1e-12)
        {
            return std::numeric_limits<double>::infinity();
        }
        return std::abs(normal.dot(point) + offset) / normal_norm;
    }

    bool align_plane_for_match(const Vec3_t& reference_normal,
                               Vec3_t& normal_to_align,
                               double& offset_to_align,
                               const double min_normal_dot)
    {
        const double normal_dot = reference_normal.dot(normal_to_align);
        if (std::abs(normal_dot) <= min_normal_dot)
        {
            return false;
        }

        if (normal_dot < 0.0)
        {
            normal_to_align = -normal_to_align;
            offset_to_align = -offset_to_align;
        }
        return true;
    }

    LocalGlassPlaneMapping::LocalGlassPlaneMapping(System* pSys, Atlas *pAtlas, const float bMonocular,
                                                   PointCloudMapping* pPointCloudMapping, const GlassPlaneMappingConfig& config)
    :mpAtlas(pAtlas), mbMonocular(bMonocular), mpSys(pSys), mpPointCloudMapping(pPointCloudMapping), mConfig(config)
    {

    }

    void LocalGlassPlaneMapping::ResetPlaneCandidate()
    {
        mPlaneCandidates.clear();
    }

    void LocalGlassPlaneMapping::StartPlaneCandidate(const Eigen::Vector3d& n_w, double d_w, const Eigen::Vector3d& centroid_w,
                                                     bool has_centroid_w, KeyFrame* keyfrm, unsigned int keypln_idx)
    {
        PlaneCandidateState candidate;
        candidate.normal = n_w;
        candidate.offset = d_w;
        candidate.centroid = centroid_w;
        candidate.has_centroid = has_centroid_w && has_valid_centroid(centroid_w);
        candidate.consecutive = 1;
        candidate.last_kf_id = keyfrm ? keyfrm->mnId : 0;
        if (keyfrm)
        {
            candidate.observations.emplace_back(keyfrm, keypln_idx);
        }
        mPlaneCandidates.push_back(std::move(candidate));
    }

    // 输入当前帧的id，删除mPlaneCandidates中当前帧没被更新的候选平面。
    void LocalGlassPlaneMapping::PruneStalePlaneCandidates(long unsigned int current_kf_id)
    {
        // 一个经典的erase-remove_if组合。erase负责删除，remove_if接受三个输入(begin, end, lambda表达式)。
        // remove_if会自动遍历begin到end，将当前指针的元素传入lambda表达式，然后将符合lambda表达式的元素送到末尾，最后使用erase删除掉这段末尾。
        mPlaneCandidates.erase(
            std::remove_if(mPlaneCandidates.begin(), mPlaneCandidates.end(),
                           [current_kf_id](const PlaneCandidateState& candidate)
                           {
                                //如果candidate并没有在当前帧被更新，则说明它是断裂的，删除。如果被更新，这两个应该相等。
                                return current_kf_id > candidate.last_kf_id;
                           }),
            mPlaneCandidates.end());
    }


    bool LocalGlassPlaneMapping::checkPlaneMatch(const Eigen::Vector3d& ncurrent, double dcurrent, const Eigen::Vector3d& centroid_current,
                                                bool has_centroid_current, const PlaneCandidateState& candidate) const
    {
        Eigen::Vector3d ncandidate=candidate.normal;
        double dcandidate=candidate.offset;

        if (!align_plane_for_match(ncurrent, ncandidate, dcandidate, mConfig.plane_match_thr_ang))
        {
            return false;
        }

        const bool current_centroid_valid = has_centroid_current && has_valid_centroid(centroid_current);
        if (current_centroid_valid)
        {
            const double dist_current_to_candidate = point_plane_distance(ncandidate, dcandidate, centroid_current);
            if (dist_current_to_candidate >= mConfig.plane_match_thr_d)
            {
                return false;
            }

            if (candidate.has_centroid && has_valid_centroid(candidate.centroid))
            {
                const double dist_candidate_to_current = point_plane_distance(ncurrent, dcurrent, candidate.centroid);
                return dist_candidate_to_current < mConfig.plane_match_thr_d;
            }
            return true;
        }

        return std::abs(dcurrent - dcandidate) < mConfig.plane_match_thr_d;
    }

    bool LocalGlassPlaneMapping::checkPlaneMatch(const Vec3_t &ncurrent, const double dcurrent, const Eigen::Vector3d& centroid_current,
                                                bool has_centroid_current, const Plane* plane_lm){
        if (!plane_lm)
        {
            return false;
        }
        Eigen::Vector3d nw=plane_lm->get_normal();
        double dw=plane_lm->get_offset();

        if(!align_plane_for_match(ncurrent, nw, dw, mConfig.plane_match_thr_ang))
        {
            return false;
        }

        const bool current_centroid_valid = has_centroid_current && has_valid_centroid(centroid_current);
        if (current_centroid_valid)
        {
            const double dist_current_to_lm = point_plane_distance(nw, dw, centroid_current);
            if (dist_current_to_lm >= mConfig.plane_match_thr_d)
            {
                return false;
            }

            const Vec3_t lm_centroid = plane_lm->getCentroid();
            if (has_valid_centroid(lm_centroid))
            {
                const double dist_lm_to_current = point_plane_distance(ncurrent, dcurrent, lm_centroid);
                return dist_lm_to_current < mConfig.plane_match_thr_d;
            }
            return true;
        }

        return std::abs(dcurrent-dw) < mConfig.plane_match_thr_d;
    }

    bool LocalGlassPlaneMapping::checkPlaneMatch(const keyplane* pln1, const keyplane* pln2){
        if (!pln1 || !pln2)
        {
            return false;
        }
        Eigen::Vector3d n1=pln1->get_normal();
        double d1=pln1->get_offset();
        Eigen::Vector3d n2=pln2->get_normal();
        double d2=pln2->get_offset();

        if(!align_plane_for_match(n1, n2, d2, mConfig.plane_match_thr_ang))
        {
            return false;
        }

        const Vec3_t c1 = pln1->getCentroid();
        const Vec3_t c2 = pln2->getCentroid();
        const bool has_c1 = has_valid_centroid(c1);
        const bool has_c2 = has_valid_centroid(c2);
        if (has_c1 && has_c2)
        {
            return point_plane_distance(n2, d2, c1) < mConfig.plane_match_thr_d &&
                   point_plane_distance(n1, d1, c2) < mConfig.plane_match_thr_d;
        }
        if (has_c1)
        {
            return point_plane_distance(n2, d2, c1) < mConfig.plane_match_thr_d;
        }
        if (has_c2)
        {
            return point_plane_distance(n1, d1, c2) < mConfig.plane_match_thr_d;
        }

        return std::abs(d1-d2) < mConfig.plane_match_thr_d;
    }

    bool fit_plane_from_cluster(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                                const pcl::PointIndices& cluster_indices,
                                Eigen::Vector3d& normal,
                                double& offset,
                                double& mean_dist)
    {
        if (!cloud || cluster_indices.indices.size() < 3)
        {
            return false;
        }

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        size_t count = 0;
        for (const int idx : cluster_indices.indices)
        {
            const pcl::PointXYZRGBA& p = cloud->points[idx];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                continue;
            }
            centroid += Eigen::Vector3d(p.x, p.y, p.z);
            ++count;
        }
        if (count < 3)
        {
            return false;
        }
        centroid /= static_cast<double>(count);

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (const int idx : cluster_indices.indices)
        {
            const pcl::PointXYZRGBA& p = cloud->points[idx];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                continue;
            }
            const Eigen::Vector3d q = Eigen::Vector3d(p.x, p.y, p.z) - centroid;
            cov += q * q.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
        if (eig.info() != Eigen::Success)
        {
            return false;
        }

        normal = eig.eigenvectors().col(0); // smallest eigenvalue direction
        if (!normal.allFinite() || normal.norm() < 1e-9)
        {
            return false;
        }
        normal.normalize();
        offset = -normal.dot(centroid);

        if (offset < 0.0)
        {
            normal = -normal;
            offset = -offset;
        }
        else if (std::abs(offset) < 1e-8)
        {
            int idx = 0;
            double maxc = std::abs(normal(0));
            if (std::abs(normal(1)) > maxc) { maxc = std::abs(normal(1)); idx = 1; }
            if (std::abs(normal(2)) > maxc) { idx = 2; }
            if (normal(idx) < 0.0) { normal = -normal; }
        }

        double dist_sum = 0.0;
        size_t valid_dist_count = 0;
        for (const int idx : cluster_indices.indices)
        {
            const pcl::PointXYZRGBA& p = cloud->points[idx];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                continue;
            }
            const Eigen::Vector3d pw(p.x, p.y, p.z);
            dist_sum += std::abs(normal.dot(pw) + offset);
            ++valid_dist_count;
        }
        if (valid_dist_count == 0)
        {
            return false;
        }
        mean_dist = dist_sum / static_cast<double>(valid_dist_count);
        return std::isfinite(mean_dist);
    }

    bool LocalGlassPlaneMapping::process_new_kf_glass(KeyFrame* keyfrm){
        if(keyfrm->isBad()){
            // ResetPlaneCandidate();
            return false;
        }
        // 判断当前关键帧是否有足够的玻璃。并对mpCurrentKeyFrame->has_good_plane进行更改。


        return create_GlassToPlane(keyfrm);


    }

    void LocalGlassPlaneMapping::AttachNearbyMapPoints(Plane* plane, KeyFrame* keyfrm)
    {
        if (!plane || !keyfrm || plane->GetMap() != mpAtlas->GetCurrentMap())
        {
            return;
        }
        const std::vector<MapPoint*> vpMPs = keyfrm->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (!pMP || pMP->isBad())
            {
                continue;
            }
            if (pMP->GetMap() != mpAtlas->GetCurrentMap())
            {
                continue;
            }
            const Eigen::Vector3f pos_f = pMP->GetWorldPos();
            const Vec3_t pos_w = pos_f.cast<double>();
            const double dist = plane->calculate_distance(pos_w);
            if (dist < mConfig.plane_attach_dist)
            {
                plane->add_map_point(pMP);
            }
        }
    }

    bool LocalGlassPlaneMapping::create_GlassToPlane(KeyFrame* keyfrm){
        Map* current_map = mpAtlas->GetCurrentMap();
        if (!current_map || !keyfrm || keyfrm->GetMap() != current_map || keyfrm->isBad())
        {
            return false;
        }

        cv::Mat mask_GS=keyfrm->get_mask_GS();
        cv::Mat depth_GS_img=keyfrm->get_depth_GS_fused_img();

        if (mask_GS.empty() || depth_GS_img.empty())
        {
            // ResetPlaneCandidate();
            return false;
        }

        //筛除那些玻璃mask部分整体占比不超过20%的帧
        cv::Mat whiteMask = (mask_GS >= mConfig.frame_mask_white_threshold);
        int white_count = cv::countNonZero(whiteMask);
        double ratio = static_cast<double>(white_count) / mask_GS.total();
        std::cout<<"white_count: "<<white_count<<" mask_GS.total()"<<mask_GS.total()<<std::endl;
        std::cout<<"当前帧玻璃区域占比为: "<<ratio<<std::endl;
        if (ratio < mConfig.min_frame_glass_ratio)
        {
            // ResetPlaneCandidate();
            return false;
        }

        std::vector<cv::Mat> vDepthGS=keyfrm->get_depth_GS_imgs();
        if(vDepthGS.size()<=0){
            return false;
        }

        // 对每个玻璃块的depth进行RANSAC plane fitting，计算当前帧玻璃blocks中能检测到的keypln
        std::vector<std::unique_ptr<keyplane>> keypln_candidates;
        // 将每个平面的indices个数作为评分标准，进行评分
        std::vector<int> keypln_indices_sizes;
        for (int i = 0; i < vDepthGS.size(); ++i) {
            // 玻璃block占比过低，跳过
            const double block_ratio = static_cast<double>(cv::countNonZero(vDepthGS[i])) / vDepthGS[i].total();
            if (block_ratio < mConfig.min_block_glass_ratio){
                continue;
            }
            // 生成当前玻璃block的点云
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
            glass_block_pointcloud_generate(cloud, keyfrm, vDepthGS[i], mConfig.pointcloud_mask_threshold);
            Vec3_t centroid_c = Vec3_t::Zero();
            const bool has_centroid = compute_cloud_centroid(cloud, centroid_c);

            // 代入点云，用RANSAC计算出平面方程。如果能检测出有效平面，则加入keypln_candidate。
            std::vector<double> pl_equation;
            double inlier_ratio = 0.0;
            double mean_inlier_dist = std::numeric_limits<double>::infinity();
            int indices_size = glass_block_keyplane_fitting_RANSAC(cloud, pl_equation, &inlier_ratio, &mean_inlier_dist);
            if(indices_size){
                if (inlier_ratio < mConfig.min_ransac_inlier_ratio)
                {
                    continue;
                }
                if (mConfig.max_ransac_mean_distance > 0.0f &&
                    (!std::isfinite(mean_inlier_dist) || mean_inlier_dist > mConfig.max_ransac_mean_distance))
                {
                    continue;
                }
                if (pl_equation.size() < 4)
                {
                    std::cerr << "error: plane equation size invalid." << std::endl;
                    return false;
                }
                std::unique_ptr<keyplane> keypln = std::make_unique<keyplane>(keyfrm);
                keypln->set_equation(pl_equation[0],pl_equation[1],pl_equation[2],pl_equation[3]);
                if (has_centroid)
                {
                    keypln->setCentroid(centroid_c);
                }
                keypln->set_valid();
                keypln_candidates.push_back(std::move(keypln));
                keypln_indices_sizes.push_back(indices_size);
            }
        }
        
        // keypln_candidates中可能包含若干处于相同平面的候选。首先使用并查集存储平面之间的聚类关系
        DSU_plane_candidate keypln_DSU(keypln_candidates.size());
        for (int i = 0; i < keypln_candidates.size(); i++) {
            for(int j=i+1; j< keypln_candidates.size(); j++){
                if( checkPlaneMatch(keypln_candidates[i].get(), keypln_candidates[j].get())){
                    //将j和i合并。这里需要使用并查集作为数据结构来存合并好的平面。
                    keypln_DSU.unite(i,j);
                }
            }
        }
        std::unordered_map<int, std::vector<int>> keypln_clusters=keypln_DSU.output_clusters();


        // 从每个聚类中，选取内点最多的平面作为代表平面加入最终不冗余的keypln_fused
        std::vector<size_t> keyplns_fused_indices;
        keyplns_fused_indices.reserve(keypln_clusters.size());
        for(const auto &[k,v] : keypln_clusters){
            int most_indices_idx=v[0];
            for(int x:v){
                if(keypln_indices_sizes[x]>keypln_indices_sizes[most_indices_idx]){
                    most_indices_idx=x;
                }
            }
            keyplns_fused_indices.push_back(static_cast<size_t>(most_indices_idx));
        }


        // 将合并后的keyplns加入地图
        bool has_good_plane=false;
        for (size_t i = 0; i < keyplns_fused_indices.size(); i++){
            const size_t idx = keyplns_fused_indices[i];
            if (idx >= keypln_candidates.size() || !keypln_candidates[idx])
            {
                continue;
            }
            keyplane* keypln = keypln_candidates[idx].release();
            if(add_keyplane_to_map(keyfrm, keypln)){
                has_good_plane=true;
            }
        }

        // 当前帧的候选池更新完成。去除不连续的平面。
        PruneStalePlaneCandidates(keyfrm->mnId);
        return has_good_plane;


    }
    int LocalGlassPlaneMapping::glass_block_keyplane_fitting_RANSAC(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                                                                    std::vector<double>& pl_equation,
                                                                    double* inlier_ratio,
                                                                    double* mean_inlier_dist){

        if (inlier_ratio)
        {
            *inlier_ratio = 0.0;
        }
        if (mean_inlier_dist)
        {
            *mean_inlier_dist = std::numeric_limits<double>::infinity();
        }
        if (!cloud || cloud->empty())
        {
            return 0;
        }

        // ------------------ pcl自带的RANSAC方法拟合平面 -------------------
        // 用于存储分割结果
        pcl::ModelCoefficients coefficients;
        pcl::PointIndices inliers;
        // 创建分割对象
        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        seg.setOptimizeCoefficients(true);          // 可选：对模型系数进行优化
        seg.setModelType(pcl::SACMODEL_PLANE);      // 模型类型：平面
        seg.setMethodType(pcl::SAC_RANSAC);         // 方法类型：RANSAC
        seg.setDistanceThreshold(mConfig.ransac_distance_threshold);             // 内点距离阈值（单位：米，根据点云尺度调整）
        seg.setInputCloud(cloud);

        // 执行分割
        seg.segment(inliers, coefficients);

        if (inliers.indices.empty()) {
            std::cerr << "未找到平面." << std::endl;
            return 0;
        } else {
            if (coefficients.values.size() < 4)
            {
                std::cerr << "平面系数数量不足." << std::endl;
                return 0;
            }
            const double a = coefficients.values[0];
            const double b = coefficients.values[1];
            const double c = coefficients.values[2];
            const double d = coefficients.values[3];
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d))
            {
                std::cerr << "平面系数异常 (NaN/Inf)." << std::endl;
                return 0;
            }
            const double n_norm = std::sqrt(a * a + b * b + c * c);
            if (n_norm < 1e-6)
            {
                std::cerr << "平面法向量范数过小." << std::endl;
                return 0;
            }
            double mean_dist = 0.0;
            size_t finite_inlier_count = 0;
            for (int idx : inliers.indices)
            {
                if (idx < 0 || idx >= static_cast<int>(cloud->points.size()))
                {
                    continue;
                }
                const pcl::PointXYZRGBA& pt = cloud->points[idx];
                if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                {
                    continue;
                }
                const double dist = std::abs(a * pt.x + b * pt.y + c * pt.z + d) / n_norm;
                if (!std::isfinite(dist))
                {
                    continue;
                }
                mean_dist += dist;
                ++finite_inlier_count;
            }
            if (finite_inlier_count == 0)
            {
                return 0;
            }
            mean_dist /= static_cast<double>(finite_inlier_count);
            if (inlier_ratio)
            {
                *inlier_ratio = static_cast<double>(finite_inlier_count) / static_cast<double>(cloud->points.size());
            }
            if (mean_inlier_dist)
            {
                *mean_inlier_dist = mean_dist;
            }
            std::cout << "平面系数: "
                    << a << "x + "
                    << b << "y + "
                    << c << "z + "
                    << d << " = 0" << std::endl;
            // std::cout << "内点数量: " << inliers->indices.size() << std::endl;
            // std::cout << "外点数量: " << valid_point_count-inliers->indices.size() << std::endl;

            // // 将平面传给pointcloudmapping线程
            // mpPointCloudMapping->insertGlassPlaneCloud(cloud,keyfrm->GetPoseInverse().matrix());
            pl_equation.clear();
            pl_equation.reserve(4);
            pl_equation.push_back(a);
            pl_equation.push_back(b);
            pl_equation.push_back(c);
            pl_equation.push_back(d);

            // 返回内点数量
            return inliers.indices.size();
        }
    }


    bool LocalGlassPlaneMapping::add_keyplane_to_map(KeyFrame* keyfrm, keyplane* keypln){

        Map* current_map = mpAtlas->GetCurrentMap();
        if (!current_map || keyfrm->GetMap() != current_map)
        {
            return false;
        }

        const unsigned int keypln_idx = keyfrm->add_keyplane(keypln);

        // 当前帧中的keypln的世界坐标系坐标
        Vec3_t n_w;
        double d_w = 0.0;
        Vec3_t centroid_w = Vec3_t::Zero();
        bool has_centroid_w = false;
        {
            const Eigen::Matrix4d Tcw = keyfrm->GetPose().matrix().cast<double>();
            const Eigen::Matrix3d R_cw = Tcw.block<3, 3>(0, 0);
            const Eigen::Vector3d t_cw = Tcw.block<3, 1>(0, 3);
            const Vec3_t n_c = keypln->get_normal();
            const double d_c = keypln->get_offset();
            n_w = R_cw.transpose() * n_c;
            d_w = d_c + n_c.dot(t_cw);

            const Vec3_t centroid_c = keypln->getCentroid();
            if (centroid_c.allFinite() && centroid_c.squaredNorm() > 1e-8)
            {
                centroid_w = R_cw.transpose() * (centroid_c - t_cw);
                has_centroid_w = centroid_w.allFinite();
            }
        }
        std::cout<<"归一化前的n_w: "<<n_w<<" d_w: "<<d_w<<std::endl;

        // 归一化 + 符号规范化
        double mod = n_w.norm();
        if (mod > 1e-12) {
            n_w /= mod;
            d_w /= mod;
        }
        if (d_w < 0.0) {
            n_w = -n_w;
            d_w = -d_w;
        } else if (std::abs(d_w) < 1e-8) {
            int idx = 0;
            double maxc = std::abs(n_w(0));
            if (std::abs(n_w(1)) > maxc) { maxc = std::abs(n_w(1)); idx = 1; }
            if (std::abs(n_w(2)) > maxc) { idx = 2; }
            if (n_w(idx) < 0.0) { n_w = -n_w; }
        }
        std::cout<<"归一化后的n_w: "<<n_w<<" d_w: "<<d_w<<std::endl;


        // 将当前帧观测到的keypln_w与地图中现有的plane_lm匹配。
        // 如果匹配成功，matched_plane指向匹配到的那个平面。如果匹配失败，matched_plane为空。
        Plane *matched_plane = nullptr;
        std::vector<Plane *> plane_lms = current_map->get_all_landmark_planes();
        for (int i = 0; i < plane_lms.size(); ++i)
        {
            Plane *plane_lm = plane_lms[i];
            if (!plane_lm)
            {
                continue;
            }
            if (!plane_lm->is_valid() || plane_lm->GetMap() != current_map)
            {
                continue;
            }
            if (checkPlaneMatch(n_w, d_w, centroid_w, has_centroid_w, plane_lm))
            {
                matched_plane = plane_lm;
                break;
            }
        }
        // 如果和matched_plane匹配到了的话，直接与其关联，然后return
        if (matched_plane)
        {
            if (has_centroid_w && matched_plane->get_num_map_points() == 0)
            {
                matched_plane->setCentroid(centroid_w);
            }
            matched_plane->add_observation(keyfrm, keypln_idx);
            keyfrm->add_observed_plane(matched_plane, keypln_idx);
            AttachNearbyMapPoints(matched_plane, keyfrm);
            // mPlaneCandidates.erase(
            //     std::remove_if(mPlaneCandidates.begin(), mPlaneCandidates.end(),
            //                    [&](const PlaneCandidateState& candidate)
            //                    {
            //                        return checkPlaneMatch(n_w, d_w, candidate);
            //                    }),
            //     mPlaneCandidates.end());

            std::cout << "将以下的keyplane与3d plane landmark关联: " << std::endl
                        << "equation: " << keypln->get_normal() << std::endl
                        << keypln->get_offset() << std::endl;

            return true;
        }

        // 没有匹配到已有plane_lm，将其作为候选平面加入mPlaneCandidates池，并使用连续观测策略筛选候选平面（多候选并行）
        // 它可能是：
        // 新的候选平面
        // 已在mPlaneCandidates中的，但还没有连续观测3帧的平面


        // 在候选池中寻找plane，看是否存在和keypln相匹配的
        int candidate_idx = -1;
        for (int i = 0; i < mPlaneCandidates.size(); ++i)
        {
            if (checkPlaneMatch(n_w, d_w, centroid_w, has_centroid_w, mPlaneCandidates[i]))
            {
                candidate_idx = static_cast<int>(i);
                break;
            }
        }

        // 如果没找到，则新建一个候选平面加入池子，返回false
        if (candidate_idx < 0)
        {
            StartPlaneCandidate(n_w, d_w, centroid_w, has_centroid_w, keyfrm, keypln_idx);
            return false;
        }

        // 找到了匹配。更新这个候选平面
        PlaneCandidateState& candidate = mPlaneCandidates[candidate_idx];
        // 正常情况：上一关键帧连续观测到当前关键帧
        if (candidate.last_kf_id + 1 == keyfrm->mnId)
        {
            candidate.normal = n_w;
            candidate.offset = d_w;
            candidate.centroid = centroid_w;
            candidate.has_centroid = has_centroid_w && has_valid_centroid(centroid_w);
            candidate.consecutive += 1;
            candidate.last_kf_id = keyfrm->mnId;
            candidate.observations.emplace_back(keyfrm, keypln_idx);
        }
        // 意外情况，当前帧有两个属于相同平面的keypln。前面做了聚类，所以理应不应出现这种情况。
        else if (candidate.last_kf_id == keyfrm->mnId){
            const bool has_same_kf_observation =
                std::any_of(candidate.observations.begin(), candidate.observations.end(),
                            [&](const std::pair<KeyFrame*, unsigned int>& obs)
                            {
                                return obs.first == keyfrm;
                            });
            if (!has_same_kf_observation)
            {
                candidate.observations.emplace_back(keyfrm, keypln_idx);
            }
        }
        else
        {
            StartPlaneCandidate(n_w, d_w, centroid_w, has_centroid_w, keyfrm, keypln_idx);
            return false;
        }


        if (candidate.consecutive >= mConfig.min_plane_consecutive_obs)
        {
            Plane *new_plane = new Plane(keyfrm, keypln, current_map);
            if (has_centroid_w)
            {
                new_plane->setCentroid(centroid_w);
            }
            new_plane->set_valid();
            current_map->add_landmark_plane(new_plane);

            for (const auto& obs : candidate.observations)
            {
                KeyFrame* kf = obs.first;
                const unsigned int obs_keypln_idx = obs.second;
                if (!kf || kf->isBad() || kf->GetMap() != current_map)
                {
                    continue;
                }
                if (obs_keypln_idx >= kf->keyplns_.size())
                {
                    continue;
                }
                new_plane->add_observation(kf, obs_keypln_idx);
                kf->add_observed_plane(new_plane, obs_keypln_idx);
                AttachNearbyMapPoints(new_plane, kf);
            }

            std::cout << "往地图中添加了3d plane landmark: " << std::endl
                        << "equation: " << new_plane->get_normal() << std::endl
                        << new_plane->get_offset() << std::endl;

            mPlaneCandidates.erase(mPlaneCandidates.begin() + candidate_idx);
            return true;
        }

        return false;

    }


    //返回valid_point_count，有效玻璃点云数量。通过参数返回cloud，当前帧有效玻璃部分的点云。
    //将玻璃看作一个整体版本
    int glass_pointcloud_generate(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, KeyFrame* keyfrm, int glass_mask_threshold){
        cv::Mat rgb_img=keyfrm->get_img_rgb();
        cv::Mat mask_GS=keyfrm->get_mask_GS();
        cv::Mat depth_GS_img=keyfrm->get_depth_GS_fused_img();
        if (rgb_img.empty() || mask_GS.empty() || depth_GS_img.empty())
        {
            cloud->clear();
            return 0;
        }

        const int rows = depth_GS_img.rows;
        const int cols = depth_GS_img.cols;
        cloud->width = static_cast<uint32_t>(cols);
        cloud->height = static_cast<uint32_t>(rows);
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);

        cloud->is_dense = false;
        int valid_point_count=0;
        std::vector<Vec3_t> pos_to_be_proj;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const int idx = y * cols + x;
                //如果该处是玻璃，使用depth_GS的深度。否则0
                if(mask_GS.at<uint8_t>(y,x) > glass_mask_threshold){
                    const float depth_m = read_depth_in_meters(depth_GS_img, y, x);
                    if (depth_m <= 0.0f)
                    {
                        cloud->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                        cloud->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                        cloud->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                        continue;
                    }

                    Vec3_t pos_c=coortrans_pix2cam(y,x,depth_m,keyfrm->mK_);

                    pcl::PointXYZRGBA& pt = cloud->points[idx];
                    pt.z = pos_c(2);
                    pt.x = pos_c(0);
                    pt.y = pos_c(1);
                    cv::Vec3b color_px = rgb_img.at<cv::Vec3b>(y,x);
                    pt.r = color_px[2];
                    pt.g = color_px[1];
                    pt.b = color_px[0];
                    pt.a = 255;
                    valid_point_count+=1;
                    if(y%60==0 && x%80==0){
                        pos_to_be_proj.push_back(pos_c);
                    }
                }
                else{
                    cloud->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                    cloud->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                    cloud->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
            }
        }
        // cout<<"valid_point_count: "<<valid_point_count<<endl;
        return valid_point_count;
    }

    int glass_block_pointcloud_generate(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, KeyFrame* keyfrm, cv::Mat depth_GS_img, int glass_mask_threshold){
        cloud->clear();
        cloud->is_dense = false;
        int valid_point_count=0;
        cv::Mat rgb_img=keyfrm->get_img_rgb();
        cv::Mat mask_GS=keyfrm->get_mask_GS();
        if (depth_GS_img.empty() || rgb_img.empty() || mask_GS.empty())
        {
            return 0;
        }

        cloud->points.reserve(static_cast<size_t>(cv::countNonZero(depth_GS_img)));
        for (int y = 0; y < depth_GS_img.rows; ++y) {
            for (int x = 0; x < depth_GS_img.cols; ++x) {
                // 仅保留有效点，避免用大量 NaN 稀释 RANSAC 采样命中率
                if(mask_GS.at<uint8_t>(y,x) <= glass_mask_threshold){
                    continue;
                }

                const float depth_m = read_depth_in_meters(depth_GS_img, y, x);
                if (depth_m <= 0.0f)
                {
                    continue;
                }

                const Vec3_t pos_c=coortrans_pix2cam(y,x,depth_m,keyfrm->mK_);

                pcl::PointXYZRGBA pt;
                pt.z = pos_c(2);
                pt.x = pos_c(0);
                pt.y = pos_c(1);
                cv::Vec3b color_px = rgb_img.at<cv::Vec3b>(y,x);
                pt.r = color_px[2];
                pt.g = color_px[1];
                pt.b = color_px[0];
                pt.a = 255;

                cloud->points.push_back(pt);
                valid_point_count+=1;
            }
        }

        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        return valid_point_count;
    }

    Vec3_t coortrans_pix2cam(int y, int x, float depth, Eigen::Matrix3f K){
        float fx, fy, cx, cy;
        fx = K(0,0);
        fy = K(1,1);
        cx = K(0,2);
        cy = K(1,2);
        
        if (0.0 < depth)
        {
            const float unproj_x = (x - cx) * depth / fx;
            const float unproj_y = (y - cy) * depth / fy;
            const Vec3_t pos_c{unproj_x, unproj_y, depth};
            return pos_c;
        }
        else
        {
            if(depth<0) std::cerr<<"error: coortrans_pix2cam: glass depth<0"<<std::endl;
            return Vec3_t::Zero();
        }
    }
}
