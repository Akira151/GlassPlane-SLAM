#ifndef GLASSPLANE_MAPPING_CONFIG_H
#define GLASSPLANE_MAPPING_CONFIG_H

namespace ORB_SLAM3
{

struct GlassPlaneMappingConfig
{
    int frame_mask_white_threshold = 240;
    int pointcloud_mask_threshold = 128;
    int min_plane_consecutive_obs = 10;

    float min_frame_glass_ratio = 0.20f;
    float min_block_glass_ratio = 0.05f;
    float ransac_distance_threshold = 0.05f;
    float min_ransac_inlier_ratio = 0.40f;
    float max_ransac_mean_distance = 0.02f;
    float plane_attach_dist = 0.30f;
    float plane_match_thr_ang = 0.96f;
    float plane_match_thr_d = 0.20f;
};

} // namespace ORB_SLAM3

#endif // GLASSPLANE_MAPPING_CONFIG_H
