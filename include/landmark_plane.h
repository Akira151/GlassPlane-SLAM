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

#ifndef KEYPLANE_H
#define KEYPLANE_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include "keyplane.h"
#include <unordered_map>
#include <mutex>
#include <map>
#include <vector>
namespace ORB_SLAM3
{
    class KeyFrame;
    class Map;
    class MapPoint;
    typedef Eigen::Vector3d Vec3_t;
    typedef Eigen::Vector4d Vec4_t;
    // FW: 3D keyplane structure fitted from 3D sparse point cloud
    class Plane
    {
        
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructors
        // [1] a 3D keyplane instance is initialized with given instance segmentation, labeled as *invalid first
        // [2] only after successfully fitting (RANSAC) from point cloud, it will be labeled as *valid
        // [3] if a keyplane is merged, it will be labeled as *invalid again

        Plane(KeyFrame *ref_keyfrm, keyplane *keyplane, Map *pmap);

        void add_observation(KeyFrame *keyfrm, unsigned int keypln_idx);
        void EraseObservation(KeyFrame *keyfrm);
        std::map<KeyFrame *, unsigned int> get_observations() const;
        size_t ObservationCount() const;

        // 被多少关键帧看到过
        unsigned int _num_observations = 0;

        // observations (keyframe and corresponding keyline index in this keyframe)
        // <是哪个关键帧看到的这个面，是这一帧中的第几个面>
        std::map<KeyFrame *, unsigned int> _observations;
        // Setter and Getter
        void add_map_point(MapPoint *mp);
        std::vector<MapPoint *> get_map_points() const;
        size_t get_num_map_points() const;





        void set_equation(double a, double b, double c, double d);           // set the keyplane parameters
        void set_equation(Vec3_t n, double d);
        void get_equation(double &a, double &b, double &c, double &d) const; // get the keyplane parameters

        bool is_valid() const; // indicates if a keyplane is valid
        void set_valid();      // a keyplane is valid after fitting from point cloud
        void set_invalid();    // a keyplane is invalid if it is merged as a candidate

        Vec3_t get_normal () const;      // get the normal vector
        double get_normal_norm() const; // the the norm of normal
        double get_offset() const;



        double calculate_distance(const Vec3_t &pos_w) const; // calculate point-keyplane distance

        unsigned int _id;
        static std::atomic<unsigned int> _next_id;


        // Color for visualization
        bool _has_color = false;
        double _r;
        double _g;
        double _b;

        // keyplane centroid
        void setCentroid(Vec3_t &centroid);
        Vec3_t getCentroid() const;

        // visualization
        // Vec3_t _base_vector1{0, 0, 0};
        // Vec3_t _base_vector2{0, 0, 0};
        // void update_base_vectors();

        // 用于orbslam3多地图管理。存储这个plane_landmark属于哪个地图。用getmap和updatemap来存取。
        Map* _mpMap; 

        Map* GetMap();
        void UpdateMap(Map* pMap);
        std::mutex _mMutexMap;

        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;


    private:
        // The classic Hessian form: π = (n^T, d)^T
        // !Notice here, normal n is not normalized
        Vec3_t _n;                                               // n = (a, b, c)^T, not normalized
        double _d;                                               // the distance from origin
        double _abs_n;                                           // |n|
        std::unordered_map<unsigned int, MapPoint *> _landmarks; // associated map points
        Vec3_t _plane_centroid;                                  // centroid 3D point which is calculated by the average of all map_points' coordinates.

        //! reference keyframe
        KeyFrame *_ref_keyfrm;

        //! map database
        Map *_map_db;

        // Flags
        bool _valid;
        bool _needs_refinement;

        // Statistics
        double _best_error; // best error estimated from SVD -> residual

        mutable std::mutex _mtx_position;
        mutable std::mutex _mtx_observations;
        mutable std::mutex _mtx_landmarks;

        void fromVector(const Vec3_t& n, const double d);
        void normalize(Vec3_t &n, double &d);
        void normalize(Vec4_t &coeffs);
    };

} // namespace PLPSLAM

#endif
