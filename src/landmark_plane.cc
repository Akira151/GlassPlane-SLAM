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

#include "KeyFrame.h"
#include <cmath>

#include "keyplane.h"
#include "landmark_plane.h"

namespace ORB_SLAM3
{

    std::atomic<unsigned int> Plane::_next_id{0};

    Plane::Plane(KeyFrame *ref_keyfrm ,keyplane *keypln, Map *pmap)
        : 
            _id(_next_id++),
            _n({1.0, 0.0, 0.0}),
            _d(-1.0),
            _abs_n(1.0),
            _plane_centroid({0.0, 0.0, 0.0}),
            _ref_keyfrm(ref_keyfrm),
            _valid(false),
            _needs_refinement(true),
            _mpMap(pmap)
    {
        _best_error = std::numeric_limits<double>::max();
        
        // keyplane is in camera frame; convert to world plane using inverse-transpose
        const Eigen::Matrix4d Tcw = ref_keyfrm->GetPose().matrix().cast<double>();
        const Eigen::Matrix3d R_cw = Tcw.block<3, 3>(0, 0);
        const Eigen::Vector3d t_cw = Tcw.block<3, 1>(0, 3);
        const Vec3_t n_c = keypln->get_normal();
        const double d_c = keypln->get_offset();

        const Vec3_t n_w = R_cw.transpose() * n_c;
        const double d_w = d_c + n_c.dot(t_cw);
        this->fromVector(n_w, d_w);
    }

    void Plane::fromVector(const Vec3_t& n, const double d)
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

    void Plane::normalize(Vec3_t &n, double &d)
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

    void Plane::normalize(Vec4_t &coeffs)
    {
        double mod = coeffs.head<3>().norm();
        if (!coeffs.allFinite() || !std::isfinite(mod) || mod <= std::numeric_limits<double>::epsilon())
        {
            return;
        }
        coeffs = coeffs * (1. / mod);

    }

    void Plane::set_equation(double a, double b, double c, double d)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        const Vec3_t candidate_n(a, b, c);
        if (!candidate_n.allFinite() || !std::isfinite(d))
        {
            return;
        }
        this->fromVector(candidate_n, d);
        std::cout<<"=========== <Plane.cc> Plane set==========="<<std::endl;
        std::cout<<"set normalized equation: "<<std::endl<<
        "_n: "<<_n<<std::endl<<
        "_d: "<<_d<<std::endl<<
        "_abs_n, should equals n : "<<_abs_n<<std::endl;
        std::cout<<"=========== <Plane.cc> Plane set end==========="<<std::endl;
        this->_valid=true;

    }

    void Plane::set_equation(Vec3_t n, double d)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        if (!n.allFinite() || !std::isfinite(d))
        {
            return;
        }
        this->fromVector(n, d);
        std::cout<<"=========== <Plane.cc> Plane set==========="<<std::endl;
        std::cout<<"set normalized equation: "<<std::endl<<
        "_n: "<<_n<<std::endl<<
        "_d: "<<_d<<std::endl<<
        "_abs_n, should equals n : "<<_abs_n<<std::endl;
        std::cout<<"=========== <Plane.cc> Plane set end==========="<<std::endl;
        this->_valid=true;

    }

    void Plane::get_equation(double &a, double &b, double &c, double &d) const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        a = _n(0);
        b = _n(1);
        c = _n(2);
        d = _d;
    }

    double Plane::calculate_distance(const Vec3_t &pos_w) const
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        double dist = (_n.dot(pos_w) + _d) / _abs_n;
        return std::fabs(dist);
    }   

    bool Plane::is_valid() const
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        return _valid;
    }

    void Plane::set_valid()
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        _valid = true;
    }

    void Plane::set_invalid()
    {
        std::lock_guard<std::mutex> lock1(_mtx_observations);
        std::lock_guard<std::mutex> lock2(_mtx_position);
        _valid = false;
    }



    Vec3_t Plane::get_normal() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _n;
    }

    double Plane::get_normal_norm() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _abs_n;
    }

    double Plane::get_offset() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _d;
    }


    void Plane::setCentroid(Vec3_t &centroid)
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        _plane_centroid = centroid;
    }

    Vec3_t Plane::getCentroid() const
    {
        std::lock_guard<std::mutex> lock(_mtx_position);
        return _plane_centroid;
    }

    void Plane::add_observation(KeyFrame *keyfrm, unsigned int keypln_idx)
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        if (_observations.count(keyfrm))
        {
            return;
        }
        _observations[keyfrm] = keypln_idx;

        // monocular/RGB-D, not for stereo settings for now
        _num_observations += 1;
    }

    void Plane::EraseObservation(KeyFrame *keyfrm)
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        auto it = _observations.find(keyfrm);
        if (it == _observations.end())
        {
            return;
        }
        _observations.erase(it);
        _num_observations = static_cast<unsigned int>(_observations.size());
    }

    std::map<KeyFrame *, unsigned int> Plane::get_observations() const
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        return _observations;
    }

    size_t Plane::ObservationCount() const
    {
        std::lock_guard<std::mutex> lock(_mtx_observations);
        return _observations.size();
    }

    void Plane::add_map_point(MapPoint *mp)
    {
        if (!mp || mp->isBad())
        {
            return;
        }
        std::lock_guard<std::mutex> lock_pts(_mtx_landmarks);
        const unsigned int id = mp->mnId;
        if (_landmarks.count(id))
        {
            return;
        }
        _landmarks[id] = mp;

        const Eigen::Vector3f pos_f = mp->GetWorldPos();
        const Eigen::Vector3d pos = pos_f.cast<double>();
        const size_t n = _landmarks.size();
        {
            std::lock_guard<std::mutex> lock_pos(_mtx_position);
            if (n == 1)
            {
                _plane_centroid = pos;
            }
            else
            {
                _plane_centroid = (_plane_centroid * static_cast<double>(n - 1) + pos) / static_cast<double>(n);
            }
        }
    }

    std::vector<MapPoint *> Plane::get_map_points() const
    {
        std::lock_guard<std::mutex> lock(_mtx_landmarks);
        std::vector<MapPoint *> points;
        points.reserve(_landmarks.size());
        for (const auto &kv : _landmarks)
        {
            points.push_back(kv.second);
        }
        return points;
    }

    size_t Plane::get_num_map_points() const
    {
        std::lock_guard<std::mutex> lock(_mtx_landmarks);
        return _landmarks.size();
    }

    Map* Plane::GetMap()
    {
        unique_lock<mutex> lock(_mMutexMap);
        return _mpMap;
    }

    void Plane::UpdateMap(Map* pMap)
    {
        unique_lock<mutex> lock(_mMutexMap);
        _mpMap = pMap;
    }

} // namespace PLPSLAM
