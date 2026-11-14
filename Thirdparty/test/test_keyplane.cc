#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv/cv.hpp>
#include "keyplane.h"
#include "KeyFrame.h"
#include "Map.h"

using namespace ORB_SLAM3;
using namespace std;

int main() {
    std::cout<<"yes!"<<std::endl;

    typedef Eigen::Vector3d Vec3;
    Vec3 n1(9,0,0);
    double d1=9;
    Vec3 n2(1,2,3);
    double d2=2;

    KeyFrame* kf;
    Map* map_db;

    keyplane kp1(kf,map_db);
    kp1.set_equation(n1,d1);
    double a = 0, b = 0, c = 0, d = 0;
    cout<<"a,b,c,d orin: "<<a<<" "<<b<<" "<<c<<" "<<d<<endl;
    kp1.get_equation(a,b,c,d);
    cout<<"kp1 equation: "<<a<<" "<<b<<" "<<c<<" "<<d<<endl;

    keyplane kp2(kf,map_db);
    kp2.set_equation(n2,d2);
    kp2.get_equation(a,b,c,d);
    cout<<"kp2 equation: "<<a<<" "<<b<<" "<<c<<" "<<d<<endl;


    return 0;
}
