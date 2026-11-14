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
#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <pcl/io/pcd_io.h>
#include "System.h"

#include <boost/make_shared.hpp>
namespace ORB_SLAM3{
    
PointCloudMapping::PointCloudMapping(double resolution_)
{
    cout<<"PointCloudMapping start, resolution="<<resolution_<<endl;
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    // globalMap = boost::make_shared< PointCloud >( );
    globalMap = std::make_shared< PointCloud >( );
    
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
    cout<<"PointCloudMapping end"<<endl;
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    
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
    
    cout<<"receive a glass plane pointcloud"<<endl;
    unique_lock<mutex> lck(glassPlaneMutex);
    PointCloud::Ptr cloud_transformed(new PointCloud);
    pcl::transformPointCloud(*cloud, *cloud_transformed, T);
    glassPlaneClouds.push_back(cloud_transformed);

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
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            
            float d = depth.ptr<float>(m)[n];
            

            if (d < 0.01 ){
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

            tmp->points.push_back(p);
            valid_count++;
        }
    }

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat( kf->GetPose() );
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    std::cout << "depth sample: " << depth.at<float>(240,320) << " m" << std::endl;

    cout<<"local pointcloud: valid_count:"<<valid_count<<", zero_count="<<zero_count<<endl;
    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    cout<<"viewer start"<<endl;
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        // for ( size_t i=lastKeyframeSize; i<N ; i++ )
        // {
        //     PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
        //     *globalMap += *p;
        // }

        // 提取出来的平面的显示。只显示最后一个玻璃平面。
        size_t N_Plane=0;
        PointCloud::Ptr plane_to_add;
        {
            unique_lock<mutex> lck( glassPlaneMutex );
            N_Plane=glassPlaneClouds.size();
            if(!glassPlaneClouds.empty() && N_Plane > lastPlaneIdx){
                plane_to_add = glassPlaneClouds.back();
            }
        }


        // 对global点云进行滤波
        // PointCloud::Ptr tmp(new PointCloud());
        // voxel.setInputCloud( globalMap );
        // voxel.filter( *tmp );
        // globalMap->swap( *tmp );

        // // 在滤波之后的global点云的基础上加上每帧的玻璃平面点云
        // PointCloud::Ptr display(new PointCloud);
        // *display += *globalMap;
        // if (plane_to_add)
        // {
        //     *display += *plane_to_add;
        // }
        // viewer.showCloud( display );
        // cout << "show global map, size=" << display->points.size() << endl;

        // 点云渐近递增
        if (plane_to_add)
        {
            *globalMap += *plane_to_add;
        }
        viewer.showCloud( globalMap );
        cout << "show global map, size=" << globalMap->points.size() << endl;


        lastKeyframeSize = N;
        lastPlaneIdx=N_Plane;
    }
}


}