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

cv::Mat disparityMat, disparityMat8;
Mat threeDEnvironment;

void disparityEvent(int evt, int x, int y, int flags, void*)
{
   // evt==CV_EVENT_MOUSEMOVE;
//    cout<<"Disparity Value: "<<disparityMat.at<short>(y,x)/16<<std::endl;
    cout<<"Disparity Value: "<<disparityMat.at<float>(y,x)<<std::endl;
}

void environmentEvent(int evt, int x, int y, int flags, void*)
{
   // evt==CV_EVENT_MOUSEMOVE;
    cout<<"Environment Value: "<<threeDEnvironment.at<cv::Vec3f>(y,x)<<std::endl;
}

void blockMatching(const cv::Mat& leftImage, const cv::Mat& rightImage, cv::Mat& disparity)
{
//    StereoBM bm(StereoBM::BASIC_PRESET, 16, 9);

    StereoBM bm(StereoBM::BASIC_PRESET, 16, 9);
    bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
    //bm.state->preFilterType = CV_STEREO_BM_NORMALIZED_RESPONSE;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = 11;
    bm.state->minDisparity = -10;       //default is zero
    bm.state->numberOfDisparities = 16*8;//default is 64
    bm.state->textureThreshold = 6;
    bm.state->uniquenessRatio = 3;
    bm.state->speckleWindowSize = 8;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;
    bm(leftImage, rightImage, disparity, CV_32F);
}

int main()
{
    bool isBlockmatching = true;
    Mat leftImage, rightImage;
    leftImage = imread("/lhome/luqman/Work/stereo_tutorial/images/leftImg.png");
    rightImage = imread("/lhome/luqman/Work/stereo_tutorial/images/rightImg.png");

    // Left camera intrinsics
    cv::Mat leftCameraMat = cv::Mat(3,3,CV_64F);
    leftCameraMat.at<double>(0,0) = 643.6696894899887;
    leftCameraMat.at<double>(0,1) = 0.0;
    leftCameraMat.at<double>(0,2) = 362.55693461986806;
    leftCameraMat.at<double>(1,0) = 0;
    leftCameraMat.at<double>(1,1) = 643.195958095438;
    leftCameraMat.at<double>(1,2) = 235.97439375017078;
    leftCameraMat.at<double>(2,0) = 0.0;
    leftCameraMat.at<double>(2,1) = 0.0;
    leftCameraMat.at<double>(2,2) = 1.0;

    // Right camera intrinsics
    cv::Mat rightCameraMat = cv::Mat(3,3,CV_64F);
    rightCameraMat.at<double>(0,0) = 646.5493349750983;
    rightCameraMat.at<double>(0,1) = 0;
    rightCameraMat.at<double>(0,2) = 371.2506584573806;
    rightCameraMat.at<double>(1,0) = 0;
    rightCameraMat.at<double>(1,1) = 646.114994072424;
    rightCameraMat.at<double>(1,2) = 240.7298493693128;
    rightCameraMat.at<double>(2,0) = 0;
    rightCameraMat.at<double>(2,1) = 0;
    rightCameraMat.at<double>(2,2) = 1;

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

    cv::Mat rotMat = cv::Mat::eye(3,3,CV_64F);

    cv::Mat translationMat = cv::Mat(3,1,CV_64F);
    translationMat.at<double>(0,0) = 153.50167013008075/646.5493349750983; // 0.237
    translationMat.at<double>(1,0) = 0.00;
    translationMat.at<double>(2,0) = 0.00;

    cv::Size imgSize = leftImage.size();
    cv::Mat RectTransleft, RectTransRight, RectProjMatLeft, RectProjMatRight;
    cv::Mat DispToDepthMapp;

    cv::stereoRectify(leftCameraMat, distorLeft, rightCameraMat, distorRight,imgSize, rotMat,
                      translationMat, RectTransleft, RectTransRight, RectProjMatLeft,
                      RectProjMatRight, DispToDepthMapp, CV_CALIB_ZERO_DISPARITY);

    cv::Mat mapLeft1, mapLeft2, mapRight1, mapRight2;
    cv::initUndistortRectifyMap(leftCameraMat, distorLeft, RectTransleft, RectProjMatLeft,
                                imgSize, CV_16SC2, mapLeft1, mapLeft2);

    cv::initUndistortRectifyMap(rightCameraMat, distorRight, RectTransRight, RectProjMatRight,
                                imgSize, CV_16SC2, mapRight1, mapRight2);

    cv::Mat leftImgRectified, rightImgRectified;
    cv::remap(leftImage, leftImgRectified, mapLeft1, mapLeft2, cv::INTER_LINEAR);
    cv::remap(rightImage, rightImgRectified, mapRight1, mapRight2, cv::INTER_LINEAR);

    cv::Mat leftGrayRectImg, rightGrayRectImg;
    //To convert from channel 3 to channel 1
    cvtColor(leftImgRectified,leftGrayRectImg,COLOR_BGR2GRAY);
    cvtColor(rightImgRectified,rightGrayRectImg,COLOR_BGR2GRAY);
    //cv::Mat disparityMat, disparityMat8;

    blockMatching(leftGrayRectImg, rightGrayRectImg, disparityMat);
    imshow("Block_Matching", disparityMat);
    cvSetMouseCallback("Block_Matching",  disparityEvent, 0);
    cvWaitKey(0);

    normalize(disparityMat, disparityMat8, 0, 255, CV_MINMAX, CV_8U);
    imshow("Normalized Disparity", disparityMat8);
   // cvSetMouseCallback("Normalized Disparity",  mouseEvent, 0);
    cvWaitKey(0);

    reprojectImageTo3D(disparityMat, threeDEnvironment, DispToDepthMapp, true, CV_32F);
    imshow("point_cloud", threeDEnvironment);
  //  cvSetMouseCallback("point_cloud", environmentEvent, 0);
    cvWaitKey();
    std::cout << "Size of 3D vironment "<<threeDEnvironment.at<cv::Vec3f>(240,325);

}
