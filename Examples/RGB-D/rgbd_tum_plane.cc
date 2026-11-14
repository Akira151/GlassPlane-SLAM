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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<cstdlib>
#include<experimental/filesystem>
#include<sstream>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;
namespace fs = std::experimental::filesystem;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<string> &vstrImageFilenamesMask, vector<string> &vstrImageFilenamesDepthGSFused,
                vector<double> &vTimestamps);
bool LoadDepthGSBlocksFromFolder(const string &folderPath, vector<cv::Mat> &vDepthGSBlocks);

int main(int argc, char **argv)
{
    if(argc != 4 && argc!=5 )
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence. associations.txt is not need but you can still add it" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<string> vstrImageFilenamesMask;
    vector<string> vstrImageFilenamesDepthGSFused;
    vector<double> vTimestamps;
    string strDatasetPath = string(argv[3]);
    LoadImages(strDatasetPath, vstrImageFilenamesRGB, vstrImageFilenamesD,vstrImageFilenamesMask,vstrImageFilenamesDepthGSFused, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesMask.size())
    {
        cerr << endl << "Different number of images for mask and depth." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesDepthGSFused.size())
    {
        cerr << endl << "Different number of images for depthGS and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    bool bUseViewer = true;
    const char* disableViewerEnv = std::getenv("ORB3_DISABLE_VIEWER");
    if(disableViewerEnv && std::string(disableViewerEnv) == "1")
        bUseViewer = false;
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,bUseViewer,0,std::string(),true,false);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD, imMask, imDepthGSFused;
    vector<cv::Mat> vImDepthGS;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imMask=cv::imread(string(argv[3])+"/"+vstrImageFilenamesMask[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imDepthGSFused = cv::imread(string(argv[3])+"/"+vstrImageFilenamesDepthGSFused[ni],cv::IMREAD_UNCHANGED);
        
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            SLAM.Shutdown();
            return 1;
        }
        if(imD.empty())
        {
            cerr << endl << "Failed to load Depth at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesD[ni] << endl;
            SLAM.Shutdown();
            return 1;
        }
        if(imMask.empty())
        {
            cerr << endl << "Failed to load Mask at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesMask[ni] << endl;
            SLAM.Shutdown();
            return 1;
        }
        if(imDepthGSFused.empty())
        {
            cerr << endl << "Failed to load Depth_fitted_fused at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesDepthGSFused[ni] << endl;
            SLAM.Shutdown();
            return 1;
        }

        // 读取单帧的玻璃表面深度检测结果
        const string frameTimestamp = fs::path(vstrImageFilenamesRGB[ni]).stem().string();
        const string blockDepthFolder = string(argv[3]) + "/depth_fitted/" + frameTimestamp;
        if(!LoadDepthGSBlocksFromFolder(blockDepthFolder, vImDepthGS))
        {
            cerr << endl << "Warning: failed to load block depths at: " << blockDepthFolder
                 << ". Continue with empty block depths for this frame." << endl;
            vImDepthGS.clear();
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
            cv::resize(imDepthGSFused, imDepthGSFused, cv::Size(width, height));
            for(size_t i = 0; i < vImDepthGS.size(); ++i)
                cv::resize(vImDepthGS[i], vImDepthGS[i], cv::Size(width, height));
        }

#ifdef COMPILEDWITHC17
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD_glassplane(imRGB,imD,imMask,vImDepthGS,imDepthGSFused,tframe);

#ifdef COMPILEDWITHC17
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strDatasetPath, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<string> &vstrImageFilenamesMask, vector<string> &vstrImageFilenamesDepthGSFused,
                vector<double> &vTimestamps)
{
    string strAssociationFilename=strDatasetPath+"/associations.txt";
    string strMaskFilename=strDatasetPath+"/mask.txt";
    string strDepthGSFusedFilename=strDatasetPath+"/depth_fitted_fused.txt";
    if(!fs::exists(fs::path(strAssociationFilename))){
        cerr << endl << "{dataset}/associations.txt do not exist!" << endl;
        return;
    }
    if(!fs::exists(fs::path(strMaskFilename))){
        cerr << endl << "{dataset}/mask.txt do not exist!" << endl;
        return;
    }
    if(!fs::exists(fs::path(strDepthGSFusedFilename))){
        cerr << endl << "{dataset}/depth_fitted_fused.txt do not exist!" << endl;
        return;
    }

    // association
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    if (!fAssociation.is_open())
    {
        cerr << endl << "Failed to open association file: " << strAssociationFilename << endl;
        return;
    }
    string s;
    while(getline(fAssociation, s))
    {
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
    fAssociation.close();
    // mask
    ifstream fmask;
    fmask.open(strMaskFilename.c_str());
    if (!fmask.is_open())
    {
        cerr << endl << "Failed to open mask index file: " << strMaskFilename << endl;
        return;
    }
    while(getline(fmask, s)){
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            string sMask;
            ss >> t >> sMask;
            vstrImageFilenamesMask.push_back(sMask);
        }
    }
    fmask.close();
    // fused depthGS
    ifstream fdepth_fitted_fuse;
    fdepth_fitted_fuse.open(strDepthGSFusedFilename.c_str());
    if (!fdepth_fitted_fuse.is_open())
    {
        cerr << endl << "Failed to open fused depth index file: " << strDepthGSFusedFilename << endl;
        return;
    }
    while(getline(fdepth_fitted_fuse, s)){
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            string sDepth_fitted_fuse;
            ss >> t >> sDepth_fitted_fuse;
            vstrImageFilenamesDepthGSFused.push_back(sDepth_fitted_fuse);
        }
    }
    fdepth_fitted_fuse.close();
}

// 读取单帧的玻璃表面深度结果。这个函数读取一个以当前帧时间戳为名的文件夹下面的1.png, 10.png, 2.png等，返回到一个每次都被初始化的vDepthGSBlocks中。
bool LoadDepthGSBlocksFromFolder(const string &folderPath, vector<cv::Mat> &vDepthGSBlocks)
{
    vDepthGSBlocks.clear();
    fs::path folder(folderPath);
    if(!fs::exists(folder) || !fs::is_directory(folder))
    {
        return false;
    }

    vector<fs::path> vBlockFiles;
    // 遍历folder下的子文件。
    for(const auto &entry : fs::directory_iterator(folder))
    {
        if(fs::is_regular_file(entry.path()))
        {
            vBlockFiles.push_back(entry.path());
        }
    }
    sort(vBlockFiles.begin(), vBlockFiles.end());

    for(const fs::path &filePath : vBlockFiles)
    {
        cv::Mat depthBlock = cv::imread(filePath.string(), cv::IMREAD_UNCHANGED);
        if(depthBlock.empty())
            continue;
        vDepthGSBlocks.push_back(depthBlock);
    }

    return !vDepthGSBlocks.empty();
}
