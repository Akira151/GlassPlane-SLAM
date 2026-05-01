
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>   // 必须
#include <pcl/filters/extract_indices.h>         // 如果要剔除平面点
#include <iostream>

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
typedef Eigen::Vector3d Vec3_t;
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
//返回valid_point_count，有效玻璃点云数量
int glass_point_generate(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, cv::Mat rgb_img, cv::Mat mask_GS, cv::Mat depth_GS_img){

    Eigen::Matrix3f K;
    float fx=615.81;
    float fy=615.75;
    float cx=316.68;
    float cy=237.42;
    K<<fx,0,cx,0,fy,cy,0,0,1;

    cout<<"K:"<<K<<endl;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

    cloud->width = 640;
    cloud->height = 480;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    cloud->is_dense = false;
    int valid_point_count=0;
    uint8_t glass_mask_threshold=128;
    //生成稠密点云的lmks。在未对mask进行instance化之前，只有一个平面。即一组点被放进字典里面。


    // Eigen::Matrix4d cam_pose_wc=keyfrm->GetPoseInverse().matrix().cast<double>();
    std::vector<Vec3_t> pos_to_be_proj;
    for (int y = 0; y < 480; ++y) {
        for (int x = 0; x < 640; ++x) {
            //如果该处是玻璃，使用depth_GS的深度。否则使用depth的深度
            uint16_t depth_value;
            int idx = y * 640 + x;

            if(mask_GS.at<uint8_t>(y,x)>glass_mask_threshold){
                depth_value = depth_GS_img.at<uint16_t>(y,x);
            }
            else{

                // depth_value = depth.at<uint16_t>(y,x); // m -> mm
                cloud->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                cloud->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                cloud->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // depth_value = depth.at<uint16_t>(y,x); // m -> mm
            // if(mask.at<uint8_t>(y,x)>glass_mask_threshold && (depth_value>10000||depth_value==0)){
            //     depth_value = depth_GS.at<uint16_t>(y,x);
            // }


            float depth_m = static_cast<float>(depth_value) / 1000.0f; //单位 m
            // Vec3_t pos_w=coortrans_pix2world(y,x,depth_m,keyfrm->camera_,cam_pose_wc);
            Vec3_t pos_c=coortrans_pix2cam(y,x,depth_m,K);



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
                // plane->add_landmark(lm);
                pos_to_be_proj.push_back(pos_c);
            }
            
        }
    }
    
    //点云可视化



}



int main(){
    //viewer初始化
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("3D Segmentation")
        );

    viewer->setBackgroundColor(0,0,0);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();
    viewer->setCameraPosition(
        0, 0, -1,   // 相机位置 (x, y, z)
        0, 0, 0,    // 注视点 (look at)
        0, -1, 0    // 相机向上方向
    );

    string dataset_path="/home/wzq/code/SLAM_methods/ORB_SLAM3_detailed_comments/test/data/";
    cv::Mat rgb_img=imread(dataset_path+"rgb.png",cv::IMREAD_UNCHANGED);
    cv::Mat mask_GS=imread(dataset_path+"mask.png",cv::IMREAD_UNCHANGED);
    cv::Mat depth_GS_img=imread(dataset_path+"depth_fitted.png",cv::IMREAD_UNCHANGED);
    cv::Mat depth_GS_vis=imread(dataset_path+"depth_fitted_vis.png",cv::IMREAD_UNCHANGED);
    //cloud初始化
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    glass_point_generate(cloud,rgb_img,mask_GS,depth_GS_img);


    // 显示
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    viewer->spinOnce(30);
    cv::imshow("depth_NN",depth_GS_vis);
    cv::waitKey(1);

    return 0;
    
}