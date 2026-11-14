
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>   // 必须
#include <pcl/filters/extract_indices.h>         // 如果要剔除平面点
#include <iostream>

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    cv::Mat mask = cv::imread("/home/wzq/code/SLAM_methods/ORB_SLAM3_detailed_comments/test/data/mask.png", cv::IMREAD_UNCHANGED);

    //筛除那些玻璃占比不超过10%的帧
    cv::Mat whiteMask = (mask >= 254);
    int white_count = cv::countNonZero(whiteMask);
    double ratio = static_cast<double>(white_count) / mask.total();
    std::cout<<"white_count: "<<white_count<<" mask_GS.total():"<<mask.total()<<std::endl;
    std::cout<<"当前帧玻璃区域占比为: "<<ratio<<std::endl;
    cv::imshow("whiteMask",whiteMask);
    cv::waitKey(0);
}