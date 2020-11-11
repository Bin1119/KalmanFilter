#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;

const int winHeight = 700;
const int winWidth = 1200;

KalmanFilter CreateKF(double sys_noise);
Point Kalman_Predict(KalmanFilter KF);
Point mousePosition = Point(winWidth >> 1, winHeight >> 1);
bool Match_straegy_bin(Point pridict_1, Point pridict_2, Mat measurement_A, Mat measurement_B);

//mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
    if (event == CV_EVENT_MOUSEMOVE) {
        mousePosition = Point(x, y);
    }
}

int main(void)
{
    int i = 0;
    const int measureNum = 2;
    Point track[2][5000];
    Point predict_pt, predict_pt_1;	// 预测用的点

    KalmanFilter KF = CreateKF(1e-5);	// 创建一个Kalman Filter
    KalmanFilter KF_1 = CreateKF(1e-6);

    // 生成假数据
    Point data[2][800];
    for (int i = 0; i < 800; i++)
        data[0][i] = Point(i + 50, 200);
    for (int j = 0; j < 800; j++)
        data[1][j] = Point(j + 50, 200);

    //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义
    Mat measurement_A = Mat::zeros(measureNum, 1, CV_32F);
    Mat measurement_B = Mat::zeros(measureNum, 1, CV_32F);

    // 新建一个窗口界面
    namedWindow("kalman");
    setMouseCallback("kalman", mouseEvent);
    Mat image(winHeight, winWidth, CV_8UC3, Scalar(0));

    for(int k=100;k<500;k++)
    {
        //1.kalman prediction
        predict_pt = Kalman_Predict(KF);
        predict_pt_1 = Kalman_Predict(KF_1);

        //2.update measurement
        //顺序版本的数据
        measurement_A.at<float>(0) = data[0][k].x;
        measurement_A.at<float>(1) = data[0][k].y;
        //鼠标+规整数据
        measurement_B.at<float>(0) = (float)mousePosition.x;
        measurement_B.at<float>(1) = (float)mousePosition.y;

        //使用匹配策略
        bool match_1_A = Match_straegy_bin(predict_pt, predict_pt_1, measurement_A, measurement_B);
        if (match_1_A)
        {
            KF.correct(measurement_A);
            KF_1.correct(measurement_B);
        }
        else
        {
            KF.correct(measurement_B);
            KF_1.correct(measurement_A);
        }

        //draw
        image.setTo(Scalar(255, 255, 255, 0));
        circle(image, predict_pt, 5, Scalar(0, 255, 0), 3);    //predicted point with green
        circle(image, predict_pt_1, 5, Scalar(0, 0, 255), 3);    //predicted point with green
        circle(image, mousePosition, 5, Scalar(255, 0, 0), 3); //current position with red

        track[0][i] = predict_pt;
        track[1][i] = predict_pt_1;


        // 绘制轨迹
        for (int j = 0; j < i; j++)
        {
            circle(image, track[0][j], 3, Scalar(0, 0, 0), 1);
            circle(image, track[1][j], 3, Scalar(225, 225, 0), 1);
        }

        i = i + 1;

        char buf[256];
        snprintf(buf, 256, "predicted position:(%3d,%3d)", predict_pt.x, predict_pt.y);
        putText(image, buf, Point(10, 30), CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(0, 0, 0), 1, 8);

        snprintf(buf, 256, "current position :(%3d,%3d)", mousePosition.x, mousePosition.y);
        putText(image, buf, cvPoint(10, 60), CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(0, 0, 0), 1, 8);

        imshow("kalman", image);
        int key = waitKey(3);
        if (key == 27) {//esc
            break;
        }
    }
    return 0;
}

KalmanFilter CreateKF(double sys_noise)		// 创建一个Kalman滤波器
{
    RNG rng;
    //1.kalman filter setup
    const int stateNum = 4;                                      //状态值4×1向量(x,y,△x,△y)
    const int measureNum = 2;                                    //测量值2×1向量(x,y)
    KalmanFilter KF(stateNum, measureNum, 0);	// 构建卡尔曼滤波器模型

    KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //转移矩阵A
    setIdentity(KF.measurementMatrix);                                             //测量矩阵H
    setIdentity(KF.processNoiseCov, Scalar::all(sys_noise));                            //系统噪声方差矩阵Q 1e-5
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
    setIdentity(KF.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
    rng.fill(KF.statePost, RNG::UNIFORM, 0, winHeight > winWidth ? winWidth : winHeight);   //初始状态值x(0)
    return KF;
}

Point Kalman_Predict(KalmanFilter KF)
{
    Mat prediction = KF.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));   //预测值(x',y')
    return predict_pt;
}

bool Match_straegy_bin(Point pridict_1, Point pridict_2, Mat measurement_A, Mat measurement_B)
{
    //检验预测1和A是否匹配
    double delt_1, delt_2;
    delt_1 = sqrt(pow((measurement_A.at<float>(0) - pridict_1.x), 2) + pow((measurement_A.at<float>(1) - pridict_1.y), 2));
    delt_2 = sqrt(pow((measurement_B.at<float>(0) - pridict_1.x), 2) + pow((measurement_B.at<float>(1) - pridict_1.y), 2));
    if (delt_1 < delt_2)
        return true;
    else
        return false;
}
