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

#include "Map.h"
#include "KeyFrame.h"
#include "keyplane.h"
#include <cmath>

namespace ORB_SLAM3
{

    keyplane::keyplane(KeyFrame *ref_keyfrm, Map *map_db)
        : 
            _n({1.0, 0.0, 0.0}),
            _d(-1.0),
            _abs_n(1.0),
            _plane_centroid({0.0, 0.0, 0.0}),
            _ref_keyfrm(ref_keyfrm),
            _map_db(map_db),
            _valid(false),
            _needs_refinement(true)
    {
        _best_error = std::numeric_limits<double>::max();
    }

    keyplane::keyplane(KeyFrame *ref_keyfrm)
        : 
            _n({1.0, 0.0, 0.0}),
            _d(-1.0),
            _abs_n(1.0),
            _plane_centroid({0.0, 0.0, 0.0}),
            _ref_keyfrm(ref_keyfrm),
            _valid(false),
            _needs_refinement(true)
    {
        _best_error = std::numeric_limits<double>::max();
    }

    void keyplane::fromVector(const Vec3_t& n, const double d)
    {
        if (!n.allFinite() || !std::isfinite(d))
        {
            return;
        }
        _n = n;
        _d = d;
        normalize(_n, _d);
        _abs_n = _n.norm();
    }

    void keyplane::normalize(Vec3_t &n, double &d)
    {
        double mod = n.norm();
        if (!n.allFinite() || !std::isfinite(d) || !std::isfinite(mod) || mod <= std::numeric_limits<double>::epsilon())
        {
            return;
        }

        n = n * (1. / mod);
        d = d * (1. / mod);

        // Canonicalize sign to remove ambiguity between (n,d) and (-n,-d).
        if (d < 0.0)
        {
            n = -n;
            d = -d;
        }
        else if (std::abs(d) < 1e-8)
        {
            int idx = 0;
            double maxc = std::abs(n(0));
            if (std::abs(n(1)) > maxc)
            {
                maxc = std::abs(n(1));
                idx = 1;
            }
            if (std::abs(n(2)) > maxc)
            {
                idx = 2;
            }
            if (n(idx) < 0.0)
            {
                n = -n;
            }
        }
    }

    void keyplane::normalize(Vec4_t &coeffs)
    {
        double mod = coeffs.head<3>().norm();
        if (!coeffs.allFinite() || !std::isfinite(mod) || mod <= std::numeric_limits<double>::epsilon())
        {
            return;
        }
        coeffs = coeffs * (1. / mod);

    }

    void keyplane::set_equation(double a, double b, double c, double d)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        const Vec3_t candidate_n(a, b, c);
        if (!candidate_n.allFinite() || !std::isfinite(d))
        {
            return;
        }
        this->fromVector(candidate_n, d);
        std::cout<<"=========== <keyplane.cc> keyplane set==========="<<std::endl;
        std::cout<<"set normalized equation: "<<std::endl<<
        "_n: "<<_n<<std::endl<<
        "_d: "<<_d<<std::endl<<
        "_abs_n, should equals n : "<<_abs_n<<std::endl;
        std::cout<<"=========== <keyplane.cc> keyplane set end==========="<<std::endl;
        this->_valid=true;

    }

    void keyplane::set_equation(Vec3_t n, double d)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        if (!n.allFinite() || !std::isfinite(d))
        {
            return;
        }
        this->fromVector(n, d);
        std::cout<<"=========== <keyplane.cc> keyplane set==========="<<std::endl;
        std::cout<<"set normalized equation: "<<std::endl<<
        "_n: "<<_n<<std::endl<<
        "_d: "<<_d<<std::endl<<
        "_abs_n, should equals n : "<<_abs_n<<std::endl;
        std::cout<<"=========== <keyplane.cc> keyplane set end==========="<<std::endl;
        this->_valid=true;

    }

    void keyplane::get_equation(double &a, double &b, double &c, double &d) const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        a = _n(0);
        b = _n(1);
        c = _n(2);
        d = _d;
    }

    void keyplane::get_equation(Eigen::Vector4d& coeffs) const //const表示该函数不能修改类成员
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        coeffs << _n(0), _n(1), _n(2), _d;
    }

    double keyplane::calculate_distance(const Vec3_t &pos_w) const
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        double dist = (_n.dot(pos_w) + _d) / _abs_n;
        return std::fabs(dist);
    }   

    bool keyplane::is_valid() const
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        return _valid;
    }

    void keyplane::set_valid()
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        _valid = true;
    }

    void keyplane::set_invalid()
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        _valid = false;
    }



    Vec3_t keyplane::get_normal() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _n;
    }

    double keyplane::get_normal_norm() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _abs_n;
    }

    double keyplane::get_offset() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _d;
    }


    void keyplane::setCentroid(Vec3_t &centroid)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        _plane_centroid = centroid;
    }

    Vec3_t keyplane::getCentroid() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _plane_centroid;
    }


} // namespace PLPSLAM
