#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cmath>
#include<iostream>

//Autonomous Data
using namespace cv;
using namespace std;

int main()
{

    Mat leftImage, rightImage;
    leftImage = imread("/lhome/luqman/Work/stereo_tutorial/images/leftImg.png");
    rightImage = imread("/lhome/luqman/Work/stereo_tutorial/images/rightImg.png");
    Mat leftGrayImg, rightGrayImg;

    //To convert from channel 3 to channel 1
    cvtColor(leftImage,leftGrayImg,COLOR_BGR2GRAY);
    cvtColor(rightImage,rightGrayImg,COLOR_BGR2GRAY);

    // Left camera intrinsics
    cv::Mat intrinLeft = cv::Mat(3,3,CV_64F);
    intrinLeft.at<double>(0,0) = 643.6696894899887;
    intrinLeft.at<double>(0,1) = 0.0;
    intrinLeft.at<double>(0,2) = 362.55693461986806;
    intrinLeft.at<double>(1,0) = 0;
    intrinLeft.at<double>(1,1) = 643.195958095438;
    intrinLeft.at<double>(1,2) = 235.97439375017078;
    intrinLeft.at<double>(2,0) = 0.0;
    intrinLeft.at<double>(2,1) = 0.0;
    intrinLeft.at<double>(2,2) = 1.0;

    // Right camera intrinsics
    cv::Mat intrinRight = cv::Mat(3,3,CV_64F);
    intrinRight.at<double>(0,0) = 646.5493349750983;
    intrinRight.at<double>(0,1) = 0;
    intrinRight.at<double>(0,2) = 371.2506584573806;
    intrinRight.at<double>(1,0) = 0;
    intrinRight.at<double>(1,1) = 646.114994072424;
    intrinRight.at<double>(1,2) = 240.7298493693128;
    intrinRight.at<double>(2,0) = 0;
    intrinRight.at<double>(2,1) = 0;
    intrinRight.at<double>(2,2) = 1;

    //Left camera distortions parameters
    cv::Mat distorLeft = cv::Mat(5,1,CV_64F);
    distorLeft.at<double>(0,0) = -0.40857872522130634;
    distorLeft.at<double>(1,0) = 0.17153250043015486;
    distorLeft.at<double>(2,0) = -0.00017086167957252753;
    distorLeft.at<double>(3,0) = -0.0001514669827181509;
    distorLeft.at<double>(4,0) = 0.0;

    //Right camera distortions parameters
    cv::Mat distorRight = cv::Mat(5,1,CV_64F);
    distorRight.at<double>(0,0) = -0.4059598651487747;
    distorRight.at<double>(1,0) = 0.17085484976716034;
    distorRight.at<double>(2,0) = -0.0008339149470988998;
    distorRight.at<double>(3,0) = 0.0003865393687439495;
    distorRight.at<double>(4,0) = 0.0;

    cv::Mat rotMat = cv::Mat(3,3,CV_64F);
    /*
    double roll = -1.639, pitch = -0.005, yaw = -1.597;
    cv::Mat rollMat = cv::Mat(3,3,CV_64F);
    rollMat.at<double>(0,0) = cos(roll);
    rollMat.at<double>(0,1) = sin(roll);
    rollMat.at<double>(0,2) = 0;
    rollMat.at<double>(1,0) = -sin(roll);
    rollMat.at<double>(1,1) = cos(roll);
    rollMat.at<double>(1,2) = 0;
    rollMat.at<double>(2,0) = 0.0;
    rollMat.at<double>(2,1) = 0.0;
    rollMat.at<double>(2,2) = 1.0;

    cv::Mat yawMat = cv::Mat(3,3,CV_64F);
    yawMat.at<double>(0,0) = cos(yaw);
    yawMat.at<double>(0,1) = 0;
    yawMat.at<double>(0,2) = -sin(yaw);
    yawMat.at<double>(1,0) = 0;
    yawMat.at<double>(1,1) = 1.0;
    yawMat.at<double>(1,2) = 0;
    yawMat.at<double>(2,0) = sin(yaw);
    yawMat.at<double>(2,1) = 0.0;
    yawMat.at<double>(2,2) = cos(yaw);

    cv::Mat pitchMat = cv::Mat(3,3,CV_64F);
    pitchMat.at<double>(0,0) = 1.0;
    pitchMat.at<double>(0,1) = 0.0;
    pitchMat.at<double>(0,2) = 0.0;
    pitchMat.at<double>(1,0) = 0.0;
    pitchMat.at<double>(1,1) = cos(pitch);
    pitchMat.at<double>(1,2) = sin(pitch);
    pitchMat.at<double>(2,0) = 0.0;
    pitchMat.at<double>(2,1) = -sin(pitch);
    pitchMat.at<double>(2,2) = cos(pitch);
    */

    rotMat.at<double>(0,0) = 1.00;
    rotMat.at<double>(0,1) = 0.0;
    rotMat.at<double>(0,2) = 0.0;
    rotMat.at<double>(1,0) = 0.0;
    rotMat.at<double>(1,1) = 1.0;
    rotMat.at<double>(1,2) = 0.0;
    rotMat.at<double>(2,0) = 0.0;
    rotMat.at<double>(2,1) = 0.0;
    rotMat.at<double>(2,2) = 1.0;
    //rotMat = rollMat * yawMat * pitchMat;

    cv::Mat translationMat = cv::Mat(3,1,CV_64F);
    translationMat.at<double>(0,0) = 153.50167013008075/646.5493349750983; // -1.380//
    translationMat.at<double>(1,0) = 0.00;
    translationMat.at<double>(2,0) = 0.00;

    cv::Size imgSize = leftImage.size();
    cv::Mat oRotleft, oRotRight, projLeft, projRight, Q;

    cv::stereoRectify(intrinLeft, distorLeft, intrinRight, distorRight,imgSize, rotMat,
                      translationMat, oRotleft, oRotRight, projLeft, projRight, Q, 0, -1,imgSize);

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(intrinLeft, distorLeft, oRotleft, projLeft,
                                imgSize, CV_16SC2, map11, map12);

    cv::initUndistortRectifyMap(intrinRight, distorRight, oRotRight, projRight,
                                imgSize, CV_16SC2, map21, map22);

    cv::Mat img1r, img2r;
    cv::remap(leftImage, img1r, map11, map12, cv::INTER_LINEAR);
    cv::remap(rightImage, img2r, map21, map22, cv::INTER_LINEAR);

    imshow("point_cloud_filename.png", img2r);
    cvWaitKey();

    StereoBM bm(StereoBM::BASIC_PRESET, 16, 9);

    bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
    bm.state->preFilterCap = 63;
    bm.state->SADWindowSize = 9;
    bm.state->minDisparity = -50;
    bm.state->numberOfDisparities = 112;
    bm.state->textureThreshold = 56;
    bm.state->uniquenessRatio = 2;
    bm.state->speckleWindowSize = 10;
    bm.state->speckleRange = 16;
    bm.state->disp12MaxDiff = 1;

    cv::Mat disparityMat;

//    bm(img1r, img2r, disparityMat);

//    imshow("point_cloud_filename.png", disparityMat);
//    cvWaitKey();

//    Mat xyz;
//    reprojectImageTo3D(disparityMat, xyz, Q, false, CV_32F);

}
