/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif
#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <pcl/io/pcd_io.h>
#include "System.h"

#include <boost/make_shared.hpp>
#include <cstdlib>
#include <string>
namespace ORB_SLAM3{
    
PointCloudMapping::PointCloudMapping(double resolution_)
{
    cout<<"PointCloudMapping start, resolution="<<resolution_<<endl;
    this->resolution = resolution_;
    // globalMap = boost::make_shared< PointCloud >( );

    cloud_voxel = std::make_shared<PointCloud>(); 
    globalMap = std::make_shared< PointCloud >( );

    const char* disableViewerEnv = std::getenv("ORB3_DISABLE_PCL_VIEWER");
    if(disableViewerEnv && std::string(disableViewerEnv) == "1")
    {
        mbEnableViewer = false;
        cout << "PointCloudMapping viewer disabled by ORB3_DISABLE_PCL_VIEWER=1" << endl;
    }
    else
    {
        viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
    }
    cout<<"PointCloudMapping end"<<endl;
}

PointCloudMapping::~PointCloudMapping()
{
    shutdown();
}

void PointCloudMapping::shutdown()
{
    unique_lock<mutex> joinLock(shutdownJoinMutex);
    if(!mbEnableViewer || !viewerThread || !viewerThread->joinable())
        return;

    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
    }
    keyFrameUpdated.notify_all();
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    if(!mbEnableViewer)
        return;
    
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
    cout<<"insertKeyFrame end"<<endl;
}

void PointCloudMapping::insertGlassPlaneCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, Eigen::Matrix4f T)
{
    if(!mbEnableViewer)
        return;
    
    cout<<"receive a glass plane pointcloud"<<endl;
    unique_lock<mutex> lck(glassPlaneMutex);
    PointCloud::Ptr cloud_transformed(new PointCloud);
    // pcl::transformPointCloud(*cloud, *cloud_transformed, T);
    glassPlaneClouds.push_back(cloud);

    keyFrameUpdated.notify_one();
    cout<<"insertGlassPlanecloud end"<<endl;
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"generatePointCloud start"<<endl;
    PointCloud::Ptr tmp( new PointCloud() );
    PointCloud::Ptr cloud(new PointCloud);
    // point cloud is null ptr
    int valid_count=0;
    int zero_count = 0;
    if(depth.type()!=CV_32F){
        cout<<"generat poinitcloud error! depth_float type not float"<<endl;
        return cloud;
    }
    for ( int m=0; m<depth.rows; m+=10 )
    {
        for ( int n=0; n<depth.cols; n+=10 )
        {
            
            float d = depth.ptr<float>(m)[n];
            

            if (!std::isfinite(d) || d < 0.01 || d > 4.0){
                zero_count++;
                continue;
            }
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
            p.a = 255;

            tmp->points.push_back(p);
            valid_count++;
        }
    }

    tmp->width = tmp->points.size();
    tmp->height = 1;
    tmp->is_dense = false;

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat( kf->GetPose() );
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    std::cout << "depth sample: " << depth.at<float>(240,320) << " m" << std::endl;

    cout<<"local pointcloud: valid_count:"<<valid_count<<", zero_count="<<zero_count<<endl;
    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    cout<<"viewer start"<<endl;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
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

    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        bool updated=false;
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            // keyFrameUpdated.wait( lck_keyframeUpdated );
            updated=(keyFrameUpdated.wait_for(lck_keyframeUpdated, std::chrono::milliseconds(30)) == std::cv_status::no_timeout);
        }

        PointCloud::Ptr plane_to_add;
        size_t N_Plane=0;
        bool show_plane_each_frame=false;
        if(updated){
            size_t N=0;
            {
                unique_lock<mutex> lck( keyframeMutex );
                N = keyframes.size();
            }
            {
                unique_lock<mutex> lck(globalMapMutex);
                for(size_t i = lastKeyframeSize; i < N; i++)
                {
                    PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
                    *globalMap += *p;
                    globalMap->width = globalMap->points.size();
                    globalMap->height = 1;
                    globalMap->is_dense = false;
                }

                if(!globalMap->empty())
                {
                    PointCloud::Ptr filtered(new PointCloud());
                    pcl::VoxelGrid<PointT> voxel;
                    voxel.setLeafSize(
                        static_cast<float>(resolution),
                        static_cast<float>(resolution),
                        static_cast<float>(resolution));
                    voxel.setInputCloud(globalMap);
                    voxel.filter(*filtered);
                    filtered->width = filtered->points.size();
                    filtered->height = 1;
                    filtered->is_dense = false;
                    globalMap = filtered;
                }
            }

            lastKeyframeSize = N;
            // 提取出来的平面的显示。只显示最后一个玻璃平面。
            
            if(show_plane_each_frame)
            {
                
                unique_lock<mutex> lck( glassPlaneMutex );
                N_Plane=glassPlaneClouds.size();
                if(!glassPlaneClouds.empty() && N_Plane > lastPlaneIdx){
                    plane_to_add = glassPlaneClouds.back();
                }
            }
        }

        // // 在滤波之后的global点云的基础上加上每帧的玻璃平面点云
        PointCloud::Ptr display(new PointCloud);
        {
            unique_lock<mutex> lck(globalMapMutex);
            *display += *globalMap;
        }
        if (plane_to_add)
        {
            *display += *plane_to_add;
        }
        display->width = display->points.size();
        display->height = 1;
        display->is_dense = false;
        // viewer.showCloud( display );
        // cout << "show global map, size=" << display->points.size() << endl;

        // 点云渐近递增
        // if (plane_to_add)
        // {
        //     *globalMap += *plane_to_add;
        // }
    

        // 显示
        viewer->removeAllPointClouds();
        viewer->addPointCloud<pcl::PointXYZRGBA>(display, "cloud");
        viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

        cout << "show global map, size=" << display->points.size() << endl;
        // cout << "Showing frame " << i+1 << "/" << n << endl;
        viewer->spinOnce(30);


        // viewer.showCloud( globalMap );


        lastPlaneIdx=N_Plane;
    }
}


}
