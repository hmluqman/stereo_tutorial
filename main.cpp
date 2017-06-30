#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cmath>
#include<iostream>

using namespace cv;
using namespace std;

cv::Mat disparityMat, disparityMat8;
Mat threeDEnvironment;

void disparityEvent(int evt, int x, int y, int flags, void*)
{
    cout<<"X cordinate: "<< x << " Y Cordinate " << y << std::endl;
    cout<<"Disparity Value: "<<disparityMat.at<short>(y,x)/16<<std::endl;

    //cout<<"Disparity Value: "<<disparityMat.at<float>(y,x)<<std::endl;
}

void environmentEvent(int evt, int x, int y, int flags, void*)
{
    cout<<"X cordinate: "<< x << " Y Cordinate " << y << std::endl;
    cout<<"Environment Value: "<<threeDEnvironment.at<cv::Vec3f>(y,x)<<std::endl;
}

int main(int argc, char *argv[])
{
    Mat leftImage, rightImage;
    leftImage = imread("/lhome/luqman/Work/stereo_tutorial/images/leftImg.png");
    rightImage = imread("/lhome/luqman/Work/stereo_tutorial/images/rightImg.png");
    Mat leftGrayImg, rightGrayImg;

    //To convert from channel 3 to channel 1
    cvtColor(leftImage,leftGrayImg,COLOR_BGR2GRAY);
    cvtColor(rightImage,rightGrayImg,COLOR_BGR2GRAY);
    cout<<"Size of Image "<<leftGrayImg.size()<<endl;
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

    // Left camera calibration matrix
    cv::Mat cameraMatrix1 = cv::Mat(3,3,CV_64F);
    cameraMatrix1.at<double>(0,0) = 9.842439e+02;
    cameraMatrix1.at<double>(0,1) = 0.000000e+00;
    cameraMatrix1.at<double>(0,2) = 6.900000e+02;
    cameraMatrix1.at<double>(1,0) = 0.000000e+00;
    cameraMatrix1.at<double>(1,1) = 9.808141e+02;
    cameraMatrix1.at<double>(1,2) = 2.331966e+02;
    cameraMatrix1.at<double>(2,0) = 0.000000e+00;
    cameraMatrix1.at<double>(2,1) = 0.000000e+00;
    cameraMatrix1.at<double>(2,2) = 1.000000e+00;

    // Right camera calibration matrix
    cv::Mat cameraMatrix2 = cv::Mat(3,3,CV_64F);
    cameraMatrix2.at<double>(0,0) = 9.895267e+02;
    cameraMatrix2.at<double>(0,1) = 0.000000e+00;
    cameraMatrix2.at<double>(0,2) = 7.020000e+02;
    cameraMatrix2.at<double>(1,0) = 0.000000e+00;
    cameraMatrix2.at<double>(1,1) = 9.878386e+02;
    cameraMatrix2.at<double>(1,2) = 2.455590e+02;
    cameraMatrix2.at<double>(2,0) = 0.000000e+00;
    cameraMatrix2.at<double>(2,1) = 0.000000e+00;
    cameraMatrix2.at<double>(2,2) = 1.000000e+00;

    cv::Mat rotMat = cv::Mat::eye(3,3,CV_64F);

    cv::Mat translationMat = cv::Mat(3,1,CV_64F);
    translationMat.at<double>(0,0) = 387.57/721.54;
    translationMat.at<double>(1,0) = 0.0;
    translationMat.at<double>(2,0) = 0.0;

    //Left camera distortions parameters
    cv::Mat distorLeft = cv::Mat(5,1,CV_64F);
    distorLeft.at<double>(0,0) = -0.37288;
    distorLeft.at<double>(1,0) = 0.20373;
    distorLeft.at<double>(2,0) = 0.0022190;
    distorLeft.at<double>(3,0) = 0.0013837;
    distorLeft.at<double>(4,0) = -0.072337;

    //Right camera distortions parameters
    cv::Mat distorRight = cv::Mat(5,1,CV_64F);
    distorRight.at<double>(0,0) = -0.36447;
    distorRight.at<double>(1,0) = 0.17900;
    distorRight.at<double>(2,0) = 0.0011481;
    distorRight.at<double>(3,0) = -6.298563e-04;
    distorRight.at<double>(4,0) = -5.314062e-02;

    cv::Size imgSize = leftImage.size();

    cv::Mat R1, R2, P1, P2, Q;
    // Calculation of stereo rectify transform
    cv::stereoRectify(cameraMatrix1, distorLeft, cameraMatrix2, distorRight, imgSize, rotMat,
                      translationMat, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY);

    int ndisparities = 16*5;
    int SADWindowSize = 21;

    //CLass for computing stereo correspondence by Block Matching Algorithm

    StereoBM bm(StereoBM::BASIC_PRESET, ndisparities, SADWindowSize);
    bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
    bm.state->preFilterCap = 32;
    bm.state->SADWindowSize = 15;// Could be 5x5.....21x21
    bm.state->minDisparity = -29;
    bm.state->numberOfDisparities = 96;
    bm.state->textureThreshold = 12;//Default value is 12
    bm.state->uniquenessRatio = 2;
    bm.state->speckleWindowSize = 10;
    bm.state->speckleRange = 12;
    bm.state->disp12MaxDiff = 1;
    bm(leftGrayImg, rightGrayImg, disparityMat);
    normalize(disparityMat, disparityMat8, 0, 255, CV_MINMAX, CV_8U);
    imshow("point_cloud", disparityMat);

    /*
    StereoBM sbm;
    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -29;
    sbm.state->textureThreshold = 95;
    sbm.state->uniquenessRatio = 1;
    sbm.state->speckleWindowSize = 15;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 1;
    sbm(leftGrayImg, rightGrayImg, disparityMat);
    normalize(disparityMat, disparityMat8, 0, 255, CV_MINMAX, CV_8U);
    imshow("point_cloud", disparityMat);
    */
    cvSetMouseCallback("point_cloud", disparityEvent, 0);
    cvWaitKey();

    reprojectImageTo3D(disparityMat, threeDEnvironment, Q, true, CV_32F);
    imshow("ThreeDEnvironement", threeDEnvironment);
    cvSetMouseCallback("ThreeDEnvironement", environmentEvent, 0);
    cvWaitKey();

}
