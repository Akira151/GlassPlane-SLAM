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

#ifndef PLANE_3D_H
#define PLANE_3D_H
#include "landmark_plane.h"
#include <opencv2/core/core.hpp>
#include <cmath>
#include <iostream>
#include <limits>

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/stuff/misc.h"
#include "Thirdparty/g2o/g2o/types/se3quat.h"

namespace ORB_SLAM3
{

    class Plane3D
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // FW:
        //  Minimal Parameterization of a 3D Plane:
        //       for applying graph optimization, we need a minimal parameterization of the plane:
        //       π = (azimuth, elevation, d), or (psi, phi, d)
        //       phi and psi should be restricted between (-π, π] to avoid singularity issue

        friend Plane3D operator*(const g2o::SE3Quat &t, const Plane3D &plane);

        Plane3D()
        {
            _coeffs << 1.0, 0.0, 0.0, 1.0;
        }

        Plane3D(const Eigen::Vector4d &v) : Plane3D()
        {
            fromVector(v);
        }

        Plane3D(const Plane *plane) : Plane3D()
        {
            double a,b,c,d=0;
            if(plane!=nullptr){
                plane->get_equation(a,b,c,d);
                if(a==0&&b==0&&c==0&&d==0){
                    std::cout<<"error! plane is nullptr when constructing Plane3D! "<<std::endl;
                }
                fromVector(Eigen::Vector4d(a,b,c,d));
            }
            else{
                std::cout<<"error! plane is nullptr when constructing Plane3D! "<<std::endl;
            }
        }

        inline Eigen::Vector4d toVector() const
        {
            return _coeffs;
        }

        inline const Eigen::Vector4d &coeffs() const
        {
            return _coeffs;
        }

        inline void fromVector(const Eigen::Vector4d &coeffs_)
        {
            if (!coeffs_.allFinite())
            {
                return;
            }
            const double n = coeffs_.head<3>().norm();
            if (!std::isfinite(n) || n <= std::numeric_limits<double>::epsilon())
            {
                return;
            }
            _coeffs = coeffs_;
            normalize(_coeffs);
        }

        // return  psi (azimuth) 用static表示其可以被Plane3d::azimuth()的方法调用，不依赖实例
        static inline double azimuth(const Eigen::Vector3d &v)
        {
            return std::atan2(v(1), v(0)); // arctan(y/x)
        }

        // return phi (elevation)
        static inline double elevation(const Eigen::Vector3d &v)
        {
            return std::atan2(v(2), v.head<2>().norm()); // arctan(z/(sqrt(x^2 + y^2)))
        }

        // return normalized distance
        // inline double distance() const
        // {
        //     return -_coeffs(3);
        // }
        inline double offset() const
        {
            return _coeffs(3);
        }

        // return nomalized normal
        inline Eigen::Vector3d normal() const
        {
            return _coeffs.head<3>();
        }

        // return rotation
        static inline Eigen::Matrix3d rotation(const Eigen::Vector3d &v)
        {
            double _azimuth = azimuth(v);
            double _elevation = elevation(v);

            // !fix the error: Eigen::AngleAxis() not declared in the scope
            // return (AngleAxis(_azimuth, Vector3::UnitZ()) * AngleAxis(-_elevation, Vector3::UnitY())).toRotationMatrix();
            Eigen::AngleAxisd azimuth_v(_azimuth, Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd elevation_v(-_elevation, Eigen::Vector3d::UnitY());
            return (azimuth_v * elevation_v).toRotationMatrix();
        }

        // called in plane vertex -> update function, which is the ⊞ operator in manifold
        inline void oplus(const Eigen::Vector3d &v)
        {
            if (!v.allFinite() || !_coeffs.allFinite())
            {
                return;
            }
            // notice here the input v: (azimuth, elevation, d)
            // construct a normal from azimuth and elevation; 
            // psi:水平偏转角 phi:竖直偏转角。
            // 用极坐标形式表示的平面最小参数化方程，即用ψ, φ,即psi和phi这两个角来表示一个法向量n=(x,y,z)。因为存在约束|n|=1，削减了一个自由度。
            // x = cos(psi)cos(phi)
            // y = sin(psi)cos(phi)
            // z = sin(phi)
            double _azimuth = v[0];
            double _elevation = v[1];
            double s = std::sin(_elevation), c = std::cos(_elevation);
            Eigen::Vector3d n(c * std::cos(_azimuth), c * std::sin(_azimuth), s);

            // rotate the normal
            Eigen::Matrix3d R = rotation(normal());
            double d = offset() + v[2];
            if (!R.allFinite() || !std::isfinite(d))
            {
                return;
            }
            _coeffs.head<3>() = R * n;
            _coeffs(3) = d;
            normalize(_coeffs);
        }

        inline Eigen::Vector3d ominus(const Plane3D &plane) const
        {
            if (!_coeffs.allFinite() || !plane.toVector().allFinite())
            {
                return Eigen::Vector3d::Zero();
            }
            // construct the rotation that would bring the plane normal in (1 0 0)
            Eigen::Matrix3d R = rotation(normal()).transpose();
            Eigen::Vector3d n = R * plane.normal();
            double d = offset() - plane.offset();
            if (!R.allFinite() || !n.allFinite() || !std::isfinite(d))
            {
                return Eigen::Vector3d::Zero();
            }
            return Eigen::Vector3d(azimuth(n), elevation(n), d);
        }

        // // for planes are perpendicular (see SP-SLAM)
        // inline ::g2o::Vector2 ominus_ver(const Plane3D &plane)
        // {
        //     // construct the rotation that would bring the plane normal in (1 0 0)
        //     Eigen::Vector3d v = normal().cross(plane.normal());
        //     Eigen::AngleAxisd ver(M_PI / 2, v / v.norm());
        //     Eigen::Vector3d b = ver * normal();

        //     Eigen::Matrix3d R = rotation(b).transpose();
        //     Eigen::Vector3d n = R * plane.normal();
        //     return ::g2o::Vector2(azimuth(n), elevation(n));
        // }

        // // for planes are parallel (see SP-SLAM)
        // inline ::g2o::Vector2 ominus_par(const Plane3D &plane)
        // {
        //     // construct the rotation that would bring the plane normal in (1 0 0)
        //     Eigen::Vector3d nor = normal();
        //     if (plane.normal().dot(nor) < 0)
        //         nor = -nor;
        //     Eigen::Matrix3d R = rotation(nor).transpose();
        //     Eigen::Vector3d n = R * plane.normal();

        //     return ::g2o::Vector2(azimuth(n), elevation(n));
        // }

    protected:
    //返回一个单位化的coeffs，即a b c d除以法向量的模长。
    // 更改：由于在优化器中，平面是作为四元数参加优化的。所以为了方便直接在这里把平面转化成（伪）四元数形式，即除掉n d的模长。
        static inline void normalize(Eigen::Vector4d &coeffs)
        {
            double n = coeffs.head<3>().norm();
            if (!coeffs.allFinite() || !std::isfinite(n) || n <= std::numeric_limits<double>::epsilon())
            {
                return;
            }
            // double n = coeffs.norm();
            coeffs = coeffs * (1. / n);

            // FW: why?
            if (coeffs(3) < 0.0)
                coeffs = -coeffs;
        }

        // normalized plane parameters
        Eigen::Vector4d _coeffs;
    };

    // overload the mathmatical operation *
    // 用法：
    // g2o::SE3Quat T_cw
    // Plane3D plane_w
    // plane_c = T_cw * plane_w
    inline Plane3D operator*(const g2o::SE3Quat &t, const Plane3D &plane)
    {
        Eigen::Vector4d v = plane._coeffs;
        Eigen::Vector4d v2;
        Eigen::Matrix3d R = t.rotation().toRotationMatrix();
        v2.head<3>() = R * v.head<3>();
        v2(3) = v(3) - t.translation().dot(v2.head<3>());
        return Plane3D(v2);
    }

} // namespace orslam3

#endif
