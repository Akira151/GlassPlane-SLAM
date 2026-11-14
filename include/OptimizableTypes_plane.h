/**
 * This file is part of Structure PLP-SLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Developed by Fangwen Shu <Fangwen.Shu@dfki.de>
 *
 * If you use this code, please cite the respective publications as
 * listed on the github repository.
 *
 * Structure PLP-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Structure PLP-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Structure PLP-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef ORB_SLAM3_OPTIMIZABLETYPES_PLANE_H
#define ORB_SLAM3_OPTIMIZABLETYPES_PLANE_H

#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/stuff/misc.h"

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/sim3.h"

#include <Eigen/Geometry>
#include <include/CameraModels/GeometricCamera.h>


// #include <g2o/types/slam3d/parameter_se3_offset.h>
// #include <g2o/types/slam3d/isometry3d_mappings.h>
#include <math.h>

#include "Plane3D.h"


namespace ORB_SLAM3
{

class landmark_vertex_plane
    //优化变量维度3 | 顶点的数据类型Plane3D
    //优化变量必须是一个向量。无论顶点类型是什么。
        : public ::g2o::BaseVertex<3, Plane3D>
{
    // A vertex for 3D plane landmark:
    // The DOF of variables to be optimized is 3 -> (phi,psi,d), the datatype is Plane3D
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    landmark_vertex_plane();

    bool read(std::istream &is) override;
    bool write(std::ostream &os) const override;

    // reset function
    void setToOriginImpl() override
    {
        _estimate = Plane3D(); // initialize the variables to be optimized
    }

    // update function (important), this is the ⊞ operater in mainfold
    void oplusImpl(const double *update_) override
    {
        Eigen::Map<const Eigen::Vector3d> update(update_);
        if (!update.allFinite())
        {
            return;
        }
        // Vec4_t est_before(_estimate.normal()(0),_estimate.normal()(1),_estimate.normal()(2),_estimate.distance());
        _estimate.oplus(Eigen::Vector3d(update));
        // Vec4_t est_after(_estimate.normal()(0),_estimate.normal()(1),_estimate.normal()(2),_estimate.distance());
        // if(est_before==est_after) std::cout<<"plane vtx oplusimpl finished, but not updating plane vtx"<<std::endl;
    }

    // bool setEstimateDataImpl(const double *est) override
    // {
    //     Eigen::Map<const Eigen::Vector4d> _est(est);
    //     _estimate.fromVector(_est);
    //     return true;
    // }

    // bool getEstimateData(double *est) const override
    // {
    //     Eigen::Map<Eigen::Vector4d> _est(est);
    //     _est = _estimate.toVector( );
    //     return true;
    // }

    // int estimateDimension() const override
    // {
    //     return 4;
    // }

    // bool setEstimateDataImpl(const double *est) override
    // {
    //     Eigen::Map<const Eigen::Vector4d> _est(est);
    //     _estimate.fromVector(_est);
    //     return true;
    // }

    // bool getEstimateData(double *est) const override
    // {
    //     Eigen::Map<Eigen::Vector4d> _est(est);
    //     _est = _estimate.toVector( );
    //     return true;
    // }

    // int estimateDimension() const override
    // {
    //     return 4;
    // }
};


// FW:
// error function is the 2D-3D line reprojection error
// Plücker coordinates, and orthonormal representation
class reproj_edge_plane3d
// 这里的四个模板参数分别表示：误差维度是3，观测量是4维向量，两个顶点的类型
//                             _error, _measurement
    : public ::g2o::BaseBinaryEdge<3, Eigen::Vector4d, g2o::VertexSE3Expmap, landmark_vertex_plane>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    reproj_edge_plane3d();

    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    // 四元数方法的尺度和oplus方法的尺度不同，导致误差e的切空间和oplus的切空间不一致。即并不是单纯的“尺度不同”，而是连空间都不同，J自然方向不对，步子会发散。
    // 解决办法：使用和oplus尺度相同的误差函数 sensors 2019中提出的方法
    // _error=
    // [ R1T*n2 ]
    // [ d1-d2 ]
    // 其中R1是推测值est，即planelm的法向量所代表的空间变换。它可以将(1,0,0)变换到n1的位置，即n1=R1*(1,0,0)。
    // R1T*n2表示“将n2进行R1的逆变换，如果n2=n1，则得到的欧拉角为(0,0)。但是实际有偏差，所以对应了两个很小的夹角。将这个作为误差。这实际上是在流形上的误差求法。
    // 将d作为线性空间的量，进行差运算。
    void computeError() override
    {
        const g2o::VertexSE3Expmap *se3Vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const landmark_vertex_plane *planeVertex = static_cast<const landmark_vertex_plane *>(_vertices[1]);
        // vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));est是Tcw
        auto Tcw_SE3Quat = se3Vertex->estimate();
        // std::cout<<"current pose est:"<<pose_keyframe_SE3Quat<<std::endl;

        const Plane3D &planelm_w = planeVertex->estimate();
        const Eigen::Vector4d plane_coeffs_w = planelm_w.toVector();
        if (!plane_coeffs_w.allFinite() || plane_coeffs_w.head<3>().norm() < 1e-9)
        {
            _error.setZero();
            return;
        }

        Eigen::Vector4d measurement=_measurement;
        if (!measurement.allFinite() || measurement.head<3>().norm() < 1e-9)
        {
            _error.setZero();
            return;
        }
        const Plane3D keypln_c(measurement);
        const Vec4_t projected_plane = cam_project_plane(Tcw_SE3Quat, keypln_c);
        if (!projected_plane.allFinite() || projected_plane.head<3>().norm() < 1e-9)
        {
            _error.setZero();
            return;
        }
        const Plane3D keypln_w(projected_plane); // re-projected line function (ax+by+c=0)
        //est-measurement
        Eigen::Vector3d error_vec=planelm_w.ominus(keypln_w);
        if (!error_vec.allFinite())
        {
            _error.setZero();
            return;
        }

        _error(0) = error_vec(0); 
        _error(1) = error_vec(1);
        _error(2) = error_vec(2);
        // std::cout<<"顶点"<<se3Vertex->id()<<"的面投影误差："<<_error.transpose()<<std::endl;
    }


    // 平面的坐标系变换方程
    // pi_c = Tcw-T * pi_w
    // pi_w = Twc-T * pi_c (本函数中使用)
    // Tcw-T = TwcT
    //==================
    // 下面两个矩阵实际上是同一个。本函数中实际使用的形式:右边的TcwT
    //      Twc-T               TcwT
    // [  Rwc       0 ]  =  [ RcwT   0]
    // [ -twcT*Rwc  1 ]     [ tcwT   1]
    inline Vec4_t cam_project_plane(const g2o::SE3Quat &Tcw_SE3Quat, const Plane3D &plane_c) const
    {
        Eigen::Matrix4d Tcw = Tcw_SE3Quat.to_homogeneous_matrix();
        if (!Tcw.allFinite())
        {
            return Vec4_t::Zero();
        }
        return Tcw.transpose() * plane_c.toVector();
    }

    // 下面的代码是为了四元数形式优化变量准备的。由于使用了另外一种(n d)形式的流形作为优化变量，所以没有被使用
    // ========================================================================================
    // //基于四元数定义的误差计算函数 IRCA 2015 CMU提出的方法
    // void computeError() override
    // {
    //     const g2o::VertexSE3Expmap *se3Vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
    //     const landmark_vertex_plane *planeVertex = static_cast<const landmark_vertex_plane *>(_vertices[1]);
    //     auto pose_keyframe_SE3Quat = se3Vertex->estimate();
    //     // std::cout<<"current pose est:"<<pose_keyframe_SE3Quat<<std::endl;

    //     const Plane3D &plane_w = planeVertex->estimate();
    //     Vec4_t est=normalize(plane_w.toVector());

    //     Eigen::Vector4d measurement=_measurement;
    //     const Plane3D &plane_c(measurement);
    //     Vec4_t proj = normalize(cam_project_plane(pose_keyframe_SE3Quat, plane_c)); // re-projected line function (ax+by+c=0)
    //     //w, x, y, z
    //     Eigen::Quaterniond est_q(est(3),est(0),est(1),est(2));
    //     Eigen::Quaterniond proj_q(proj(3),proj(0),proj(1),proj(2));
    //     Eigen::Quaterniond error_q=proj_q.conjugate()*est_q;
    //     Vec3_t error_vec=log_plane(Vec4_t(error_q.x(),error_q.y(),error_q.z(),error_q.w()));

    //     _error(0) = error_vec(0);
    //     _error(1) = error_vec(1);
    //     _error(2) = error_vec(2);
    //     // std::cout<<"顶点"<<se3Vertex->id()<<"的面投影误差："<<_error.transpose()<<std::endl;
    // }

    // //将平面参数转化成四元数形式，即a, b, c, d的模长=1
    // Vec4_t normalize(Vec4_t coeffs){
    //     double n = coeffs.head<4>().norm();
    //     coeffs = coeffs * (1. / n);
    //     return coeffs;
    // }

    // //关于四元数的元素位置关系要尤其注意。在Eigen::Q库里面是w,x,y,z，而在手写的方法里面全部是x,y,z,w，包括平面的方程也是x,y,z,d。
    // Vec3_t log_plane(Vec4_t coeffs){
    //     double x=coeffs(0);
    //     double y=coeffs(1);
    //     double z=coeffs(2);
    //     double w=coeffs(3);
    //     if (w>1.0 || w<-1.0) std::cout << "error: w>1.0! || w<-1.0 w:"<<w<<std::endl;
    //     if (w > 1.0) w = 1.0;
    //     if (w < -1.0) w = -1.0;
    //     double vnorm = std::sqrt(x*x + y*y + z*z);
    //     //如果输入的值非常接近1,0,0,0(w=1)，说明pi1和pi2已经非常接近了。那么他们的差直接设成0就好。否则分母过小可能出问题。
    //     if (vnorm < 1e-12) {
    //         std::cout<<"set log to 0 due to pi1 and pi2 are very close."<<std::endl;
    //         return Vec3_t::Zero();
    //     }
    //     Vec3_t result=(2*std::acos(w)/vnorm)*Vec3_t(x,y,z);
    //     return result;
    // }
    // ========================================================================================


    double _fx, _fy, _cx, _cy;
    Eigen::Matrix3d _K;
    Eigen::Matrix3d _cam_matrix;
};
}
#endif
