#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cmath>
#include<iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
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
    intrinLeft.at<double>(0,0) = 721.54;
    intrinLeft.at<double>(0,1) = 0.0;
    intrinLeft.at<double>(0,2) = 609.56;
    intrinLeft.at<double>(1,0) = 0;
    intrinLeft.at<double>(1,1) = 721.54;
    intrinLeft.at<double>(1,2) = 172.85;
    intrinLeft.at<double>(2,0) = 0.0;
    intrinLeft.at<double>(2,1) = 0.0;
    intrinLeft.at<double>(2,2) = 1.0;

    // Right camera intrinsics
    cv::Mat intrinRight = cv::Mat(3,3,CV_64F);
    intrinRight.at<double>(0,0) = 721.54;
    intrinRight.at<double>(0,1) = 0;
    intrinRight.at<double>(0,2) = 609.56;
    intrinRight.at<double>(1,0) = 0;
    intrinRight.at<double>(1,1) = 721.54;
    intrinRight.at<double>(1,2) = 172.85;
    intrinRight.at<double>(2,0) = 0;
    intrinRight.at<double>(2,1) = 0;
    intrinRight.at<double>(2,2) = 1;

    cv::Mat rotMat = cv::Mat(3,3,CV_64F);
    rotMat.at<double>(0,0) = 1.00;
    rotMat.at<double>(0,1) = 0.0;
    rotMat.at<double>(0,2) = 0.0;
    rotMat.at<double>(1,0) = 0.0;
    rotMat.at<double>(1,1) = 1.0;
    rotMat.at<double>(1,2) = 0.0;
    rotMat.at<double>(2,0) = 0.0;
    rotMat.at<double>(2,1) = 0.0;
    rotMat.at<double>(2,2) = 1.0;

    cv::Mat translationMat = cv::Mat(3,1,CV_64F);
    translationMat.at<double>(0,0) = 387.57/721.54;
    translationMat.at<double>(1,0) = 0.0;
    translationMat.at<double>(2,0) = 0.0;

    cv::Size imgSize = leftImage.size();

    int ndisparities = 16*5;
    int SADWindowSize = 21;
    cv::Mat disparityMat, disparityMat8;
    //CLass for computing stereo correspondence by Block Matching Algorithm

    StereoBM bm(StereoBM::BASIC_PRESET, ndisparities, SADWindowSize);
    bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
    bm.state->preFilterCap = 63;
    bm.state->SADWindowSize = 9;
    bm.state->minDisparity = -50;
    bm.state->numberOfDisparities = 192;
    bm.state->textureThreshold = 56;
    bm.state->uniquenessRatio = 2;
    bm.state->speckleWindowSize = 10;
    bm.state->speckleRange = 16;
    bm.state->disp12MaxDiff = 1;

    bm(leftGrayImg, rightGrayImg, disparityMat);
    normalize(disparityMat, disparityMat8, 0, 255, CV_MINMAX, CV_8U);
    imshow("point_cloud_filename.png", disparityMat8);
    cvWaitKey();

    //**CLass for computing stereo correspondence by Semi Block Matching Algorithm
/*
    StereoBM sbm;
    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -39;
    sbm.state->textureThreshold = 507;
    sbm.state->uniquenessRatio = 0;
    sbm.state->speckleWindowSize = 0;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 1;
    sbm(leftGrayImg, rightGrayImg, disparityMat);
    normalize(disparityMat, disparityMat8, 0, 255, CV_MINMAX, CV_8U);
    imshow("point_cloud_filename.png", disparityMat8);
    cvWaitKey();
*/
//    Mat xyz;
//    reprojectImageTo3D(disparityMat, xyz, Q, false, CV_32F);

}
