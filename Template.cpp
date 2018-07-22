 #include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <libgen.h>
#include <signal.h>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mjpg_streamer.h"

#include "LinuxDARwIn.h"
#include "StatusCheck.h"


#include "Follower.h"

#include "GenericVision.h"

#ifdef MX28_1024
#define MOTION_FILE_PATH    "../../../Data/motion_1024.bin"
#else
#define MOTION_FILE_PATH    "../../../Data/motion_4096.bin"
#endif

#define INI_FILE_PATH      "config_forward.ini"
#define INI_FILE_PATH2      "config_backward.ini"

#define U2D_DEV_NAME0       "/dev/ttyUSB0"
#define U2D_DEV_NAME1       "/dev/ttyUSB1"

#define SCRIPT_FILE_PATH    "script.asc"
#define PI 3.1415

LinuxCM730 linux_cm730(U2D_DEV_NAME0);
CM730 cm730(&linux_cm730);

using namespace std;
using namespace cv;

int button = 1;
bool pressed = false;
bool ball_found;
bool step1=1; 
bool step2=0;
bool step3=0;
int step_cycle=0;
bool stage1 = 1; 
bool stage2 = 0; 
bool stage3 = 0; 
Point3i marker1;
Point3i marker2;
Point3i marker3;

bool DEBUG = false; 

Image *rgb_output; 
minIni* ini_f;
minIni* ini_b;

Point2D* object;

GenericVision vision;
Mat hsvFrame; 

BallTracker tracker; 
Follower follower;

vector<int> region;    // 0 for semicircle, 1 for inner rect
bool noObstacles=true;
int thresholdForGreen=39;


bool tuning_objects=true;
Mat field;
int balltuning=0,objecttuning=0,opponenttuning=0,teammembertuning=0;
Vec3i gainHSV=Vec3i(11,11,11);
Point2i ball_tuner=Point2i(0,0);
int orientation=60;
Vec2f robot_position=Vec2f(397,515);
vector<Vec2f> obstacles_position_right;
vector<Vec2f> opponent_position_right;
vector<Vec2f> teammember_position_right;
Vec2f ball_position_right;
vector<Vec2f> obstacles_position;
vector<Vec2f> opponent_position;
vector<Vec2f> teammember_position;
Vec2f ball_position;
Vec3i ballHSV=Vec3i(-30,-30,-30);
Vec3i obstaclesHSV=Vec3i(-30,-30,-30);
Vec3i opponentHSV=Vec3i(-30,-30,-30);
Vec3i teammemberHSV=Vec3i(-30,-30,-30);

                timeval a, b;
                int difference = 0;

///////////////////////////////////////////////////////
///////////////// Functions ///////////////////////////
///////////////////////////////////////////////////////

void regionsDet(Mat image)
{
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    namedWindow("Thresholded using POG", WINDOW_NORMAL);
    int nRows=image.rows;
    int nCols=image.cols;
    Mat thresholded=Mat::zeros(nRows, nCols, CV_8UC1);
    int rw,cl;
    uchar *p;
    createTrackbar("thresholdForGreen", "Thresholded using POG", &thresholdForGreen, 255, NULL);
    for(rw=0;rw<nRows;rw++){
        p = thresholded.ptr<uchar>(rw);
        for(cl=0;cl<nCols;cl++){
            Vec3b intensities = image.at<Vec3b>(rw,cl);
            float percentOfGreen = (float) intensities.val[1]/(intensities.val[0]+intensities.val[1]+intensities.val[2]);
            if((percentOfGreen*100)>thresholdForGreen){
                p[cl]=0;
            }
            else
            {
                p[cl]=255;
            }
        }
    }
    imshow("Thresholded using POG",thresholded);

    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
    vector<Vec4f> lines_std;
    vector<Vec4f> lines_std_joined;
    ls->detect(thresholded, lines_std);
    Mat seg=Mat::zeros(image.rows, image.cols, CV_8UC1);;
    ls->drawSegments(seg, lines_std);

    vector<float> pos_slope;
    vector<float> neg_slope;
    lines_std_joined.push_back(lines_std[0]);
    float slope_prev=(lines_std[0][0] - lines_std[0][2]) / (lines_std[0][1] - lines_std[0][3]);
    int lines_std_joined_index=0;
    vector<Vec4f> lines_std_pos;
    vector<Vec4f> lines_std_neg;
    for(int i=0;i<lines_std.size();i++)
    {
        float slope=(lines_std[i][0]-lines_std[i][2])/(lines_std[i][1]-lines_std[i][3]);
        //ROS_INFO("slope = %f",(float)(lines_std[i][0]-lines_std[i][2])/(lines_std[i][1]-lines_std[i][3]));
        //int angle=atan (slope) * 180 / PI;
        if(slope<0.0) {
            neg_slope.push_back(slope);
            //line(image,Point(lines_std[i][0],lines_std[i][1]),Point(lines_std[i][2],lines_std[i][3]),Scalar(255, 0, 0),1,8,0);
            lines_std_neg.push_back(lines_std[i]);
        }
        else {
            pos_slope.push_back(slope);
            //line(image,Point(lines_std[i][0],lines_std[i][1]),Point(lines_std[i][2],lines_std[i][3]),Scalar(0, 255, 255),1,8,0);
            lines_std_pos.push_back(lines_std[i]);
        }
    }
    vector<Vec4f> lines_std_neg_comm;
    vector<Vec4f> lines_std_pos_comm;
    int neg_count=0,pos_count=0,count=0;
    float comm_neg_slope=neg_slope[0],comm_pos_slope=pos_slope[0];
    for(int i=0;i<neg_slope.size()-1;i++) {
        for(int j=i+1;j<neg_slope.size();j++)
            if (abs(neg_slope[i]-neg_slope[j])<=0.5)
                    count++;
        if(count>=1) {
            comm_neg_slope=neg_slope[i];
            neg_count=count;
            lines_std_neg_comm.push_back(lines_std_neg[i]);
        }
        count=0;
    }
    //cout<<"before:"<<lines_std_neg_comm.size()<<endl;
    for(int i=0;i<lines_std_neg_comm.size()-1;i++)
        for(int k=i+1;k<lines_std_neg_comm.size();k++) {
            if ((abs(round(lines_std_neg_comm[i][0]) - round(lines_std_neg_comm[k][0])) <= 1 &&
                 abs(round(lines_std_neg_comm[i][1]) - round(lines_std_neg_comm[k][1])) <= 1) ||
                (abs(round(lines_std_neg_comm[i][2]) - round(lines_std_neg_comm[k][0])) <= 1 &&
                 abs(round(lines_std_neg_comm[i][3]) - round(lines_std_neg_comm[k][1])) <= 1) ||
                (abs(round(lines_std_neg_comm[i][0]) - round(lines_std_neg_comm[k][2])) <= 1 &&
                 abs(round(lines_std_neg_comm[i][1]) - round(lines_std_neg_comm[k][3])) <= 1) ||
                (abs(round(lines_std_neg_comm[i][2]) - round(lines_std_neg_comm[k][2])) <= 1 &&
                 abs(round(lines_std_neg_comm[i][3]) - round(lines_std_neg_comm[k][3])) <= 1)) {
                //line(image,Point(lines_std_neg_comm[i][0],lines_std_neg_comm[i][1]),Point(lines_std_neg_comm[i][2],lines_std_neg_comm[i][3]),Scalar(0, 255, 255),1,8,0);
                //line(image,Point(lines_std_neg_comm[k][0],lines_std_neg_comm[k][1]),Point(lines_std_neg_comm[k][2],lines_std_neg_comm[k][3]),Scalar(0, 255, 255),1,8,0);
                lines_std_neg_comm.erase(lines_std_neg_comm.begin() + i);
                lines_std_neg_comm.erase(lines_std_neg_comm.begin() + k-1);
                i--;
            }
        }
    //cout<<"After:"<<lines_std_neg_comm.size()<<endl;
    for(int i=0;i<lines_std_neg_comm.size();i++)
        line(image,Point(lines_std_neg_comm[i][0],lines_std_neg_comm[i][1]),Point(lines_std_neg_comm[i][2],lines_std_neg_comm[i][3]),Scalar(0, 0, 255),1,8,0);
    //cout<<"-ve slope = "<<comm_neg_slope<<" count = "<<neg_count<<endl;
    count=0;
    for(int i=0;i<pos_slope.size()-1;i++) {
        for(int j=i+1;j<pos_slope.size();j++)
            if (abs(pos_slope[i]-pos_slope[j])<=0.5)
                count++;
        if(count>=1) {
            comm_pos_slope=pos_slope[i];
            pos_count=count;
            lines_std_pos_comm.push_back(lines_std_pos[i]);
        }
        count=0;
    }
    for(int i=0;i<lines_std_pos_comm.size()-1;i++)
        for(int k=i+1;k<lines_std_pos_comm.size();k++) {
            if ((abs(round(lines_std_pos_comm[i][0]) - round(lines_std_pos_comm[k][0])) <= 1 &&
                 abs(round(lines_std_pos_comm[i][1]) - round(lines_std_pos_comm[k][1])) <= 1) ||
                (abs(round(lines_std_pos_comm[i][2]) - round(lines_std_pos_comm[k][0])) <= 1 &&
                 abs(round(lines_std_pos_comm[i][3]) - round(lines_std_pos_comm[k][1])) <= 1) ||
                (abs(round(lines_std_pos_comm[i][0]) - round(lines_std_pos_comm[k][2])) <= 1 &&
                 abs(round(lines_std_pos_comm[i][1]) - round(lines_std_pos_comm[k][3])) <= 1) ||
                (abs(round(lines_std_pos_comm[i][2]) - round(lines_std_pos_comm[k][2])) <= 1 &&
                 abs(round(lines_std_pos_comm[i][3]) - round(lines_std_pos_comm[k][3])) <= 1)) {
                //line(image,Point(lines_std_neg_comm[i][0],lines_std_neg_comm[i][1]),Point(lines_std_neg_comm[i][2],lines_std_neg_comm[i][3]),Scalar(0, 255, 255),1,8,0);
                //line(image,Point(lines_std_neg_comm[k][0],lines_std_neg_comm[k][1]),Point(lines_std_neg_comm[k][2],lines_std_neg_comm[k][3]),Scalar(0, 255, 255),1,8,0);
                lines_std_pos_comm.erase(lines_std_pos_comm.begin() + i);
                lines_std_pos_comm.erase(lines_std_pos_comm.begin() + k-1);
                i--;
            }
        }
    //cout<<"After:"<<lines_std_neg_comm.size()<<endl;
    for(int i=0;i<lines_std_pos_comm.size();i++)
        line(image,Point(lines_std_pos_comm[i][0],lines_std_pos_comm[i][1]),Point(lines_std_pos_comm[i][2],lines_std_pos_comm[i][3]),Scalar(255, 0, 0),1,8,0);
    //cout<<"+ve slope = "<<comm_pos_slope<<" count = "<<pos_count<<endl;
    namedWindow("regions img", WINDOW_NORMAL);
    imshow("regions img",image);
}

void callBackFunc(int event, int x, int y, int flags, void *userdata)
{
    Mat *HSVFrame = (Mat *) userdata;
    Point2f mousePt1,mousePt2;
    /// Left button
    if (event == EVENT_LBUTTONDOWN)
    {
        mousePt1 = Point2f(x, y);
        ball_tuner=Point2f(x, y);
    }
    else if (event == EVENT_LBUTTONUP)
    {
        mousePt2 = Point2f(x, y);
    }
}

void findBallPosition(float dist,float deg)
{
    int x=round(dist*cos(deg * PI / 180));
    int y=round(dist*sin(deg * PI / 180));
    ball_position[0]=robot_position[0]+((-1)*y);
    ball_position[1]=robot_position[1]+x;
}

void findOpponentPosition(float dist,float deg)
{
    int x=round(dist*cos(deg * PI / 180));
    int y=round(dist*sin(deg * PI / 180));
    Vec2f pos;
    pos[0]=robot_position[0]+((-1)*y);
    pos[1]=robot_position[1]+x;
     
    opponent_position.push_back(pos);
}

void findobstaclesPosition(float dist,float deg)
{
    int x=round(dist*cos(deg * PI / 180));
    int y=round(dist*sin(deg * PI / 180));
    Vec2f pos;
    pos[0]=robot_position[0]+((-1)*y);
    pos[1]=robot_position[1]+x;
     
    obstacles_position.push_back(pos);
}

void findteammemberPosition(float dist,float deg)
{
    int x=round(dist*cos(deg * PI / 180));
    int y=round(dist*sin(deg * PI / 180));
    Vec2f pos;
    pos[0]=robot_position[0]+((-1)*y);
    pos[1]=robot_position[1]+x;
     
    teammember_position.push_back(pos);
}

void detectBall(Mat image){
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    Mat hsvFrame;
    cvtColor(image, hsvFrame, CV_BGR2HSV);

        Mat thresholdFrame;
        namedWindow("ballHSV threshold", WINDOW_NORMAL);
        if(ballHSV[0]>=0){
        inRange(hsvFrame, ballHSV-gainHSV, ballHSV+gainHSV, thresholdFrame);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        morphologyEx(thresholdFrame, thresholdFrame, 3, element);
        imshow("ballHSV threshold", thresholdFrame);
        Canny(thresholdFrame, thresholdFrame, 100, 255, 3);
        findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

            Point2f xy;
            float radius=0.0;
            minEnclosingCircle(contours[0],xy,radius);
            //circle(image, xy, radius,  Scalar(0, 0, 255), 1, 8, 0);

            float distanceInCms=(float)(39.1*17.1426)/radius;
            float distanceInPixel=distanceInCms/0.4;

            float relative_angle=(float)((image.cols-xy.x)/image.cols)*80.0<40.0?orientation-(40.0-(((float)(image.cols-xy.x)/image.cols)*80.0)):orientation+((((float)(image.cols-xy.x)/image.cols)*80.0)-40.0);

            findBallPosition(distanceInPixel,relative_angle);
        }
}

void detectOpponent(Mat image){
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    Mat hsvFrame;
    cvtColor(image, hsvFrame, CV_BGR2HSV);

        Mat thresholdFrame;
        namedWindow("opponentHSV threshold", WINDOW_NORMAL);
        if(opponentHSV[0]>=0){
        inRange(hsvFrame, opponentHSV-gainHSV, opponentHSV+gainHSV, thresholdFrame);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        morphologyEx(thresholdFrame, thresholdFrame, 3, element);
        imshow("opponentHSV threshold", thresholdFrame);
        Canny(thresholdFrame, thresholdFrame, 100, 255, 3);
        findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            for( int i = 0; i< contours.size(); i=hierarchy[i][0] )
            {
                Rect r= boundingRect(contours[i]);
                if(hierarchy[i][2]>0) { //Check if there is a child contour
                    //drawContours(image, contours, i, Scalar(0,0,255), 1, 8, hierarchy, 0, Point());
                    rectangle(image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);

                    int largeSide=r.width>r.height?r.width:r.height;
                    float distanceInCms=(float)(70.9*43)/largeSide;
                    float distanceInPixel=distanceInCms/0.4;

                    //circle(image, Point(r.x+r.width/2,r.y), 2,  Scalar(0, 0, 255), 1, 8, 0);
                    float relative_angle=((float)((image.cols-(r.x+r.width/2))/image.cols)*80.0)<40.0?orientation-(40.0-(((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)):orientation+((((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)-40.0);
                    findOpponentPosition(distanceInPixel,relative_angle);
                }
            }
     }
}

void detectTeammember(Mat image){
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    Mat hsvFrame;
    cvtColor(image, hsvFrame, CV_BGR2HSV);

        Mat thresholdFrame;
        namedWindow("teammemberHSV threshold", WINDOW_NORMAL);
        if(teammemberHSV[0]>=0){
        inRange(hsvFrame, teammemberHSV-gainHSV, teammemberHSV+gainHSV, thresholdFrame);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        morphologyEx(thresholdFrame, thresholdFrame, 3, element);
        imshow("teammemberHSV threshold", thresholdFrame);
        Canny(thresholdFrame, thresholdFrame, 100, 255, 3);
        findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            for( int i = 0; i< contours.size(); i=hierarchy[i][0] )
            {
                Rect r= boundingRect(contours[i]);
                if(hierarchy[i][2]>0) { //Check if there is a child contour
                    //drawContours(image, contours, i, Scalar(0,0,255), 1, 8, hierarchy, 0, Point());
                    rectangle(image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);

                    int largeSide=r.width>r.height?r.width:r.height;
                    float distanceInCms=(float)(70.9*43)/largeSide;
                    float distanceInPixel=distanceInCms/0.4;

                    //circle(image, Point(r.x+r.width/2,r.y), 2,  Scalar(0, 0, 255), 1, 8, 0);
                    float relative_angle=((float)((image.cols-(r.x+r.width/2))/image.cols)*80.0)<40.0?orientation-(40.0-(((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)):orientation+((((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)-40.0);
                    findteammemberPosition(distanceInPixel,relative_angle);
                }
            }
     }
}

void detectObstacles(Mat image){
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    Mat hsvFrame;
    cvtColor(image, hsvFrame, CV_BGR2HSV);

        Mat thresholdFrame;
        namedWindow("obstaclesHSV threshold", WINDOW_NORMAL);
        if(obstaclesHSV[0]>=0){
        inRange(hsvFrame, obstaclesHSV-gainHSV, obstaclesHSV+gainHSV, thresholdFrame);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        morphologyEx(thresholdFrame, thresholdFrame, 3, element);
        imshow("obstaclesHSV threshold", thresholdFrame);
        Canny(thresholdFrame, thresholdFrame, 100, 255, 3);
        findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            for( int i = 0; i< contours.size(); i=hierarchy[i][0] )
            {
                Rect r= boundingRect(contours[i]);
                if(hierarchy[i][2]>0) { //Check if there is a child contour
                    if(r.width*r.height>100){
                    //cout<<r.width<<"--"<<r.height<<endl;
                    rectangle(image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);

                    int largeSide=r.width>r.height?r.width:r.height;
                    float distanceInCms=(float)(70.9*112)/largeSide;
                    float distanceInPixel=distanceInCms/0.4;

                    float relative_angle=((float)((image.cols-(r.x+r.width/2))/image.cols)*80.0)<40.0?orientation-(40.0-(((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)):orientation+((((float)(image.cols-(r.x+r.width/2))/image.cols)*80.0)-40.0);
                    findobstaclesPosition(distanceInPixel,relative_angle);
				}
                }
            }
     }
}

void updateMap(){
      //if(opponent_position_right[i][1]>=6 && opponent_position_right[i][1]<=444 && opponent_position_right[i][0]>=6 && opponent_position_right[i][0]<=694)
      for(int i=0;i<opponent_position_right.size();i++)
         circle(field, Point((opponent_position_right[i][1]),(opponent_position_right[i][0])), 5, Scalar(193, 182, 255), 15, 8, 0);

      for(int i=0;i<teammember_position_right.size();i++)
         circle(field, Point((teammember_position_right[i][1]),(teammember_position_right[i][0])), 5, Scalar(0, 165, 255), 15, 8, 0);

      for(int i=0;i<obstacles_position_right.size();i++)
         rectangle(field, Point((obstacles_position_right[i][1]-12),(obstacles_position_right[i][0]-12)),Point((obstacles_position_right[i][1]+12),(obstacles_position_right[i][0])+12), Scalar(0, 0, 255), -2, 8, 0);

	
      if(ball_position[1]!=0.0)
         circle(field, Point((ball_position[1]),(ball_position[0])), 5, Scalar(0, 255, 255), 15, 8, 0);

      for(int i=0;i<opponent_position.size();i++)
         circle(field, Point((opponent_position[i][1]),(opponent_position[i][0])), 5, Scalar(193, 182, 255), 15, 8, 0);

      for(int i=0;i<teammember_position.size();i++)
         circle(field, Point((teammember_position[i][1]),(teammember_position[i][0])), 5, Scalar(0, 165, 255), 15, 8, 0);

      for(int i=0;i<obstacles_position.size();i++)
         rectangle(field, Point((obstacles_position[i][1]-12),(obstacles_position[i][0]-12)),Point((obstacles_position[i][1]+12),(obstacles_position[i][0])+12), Scalar(0, 0, 255), -2, 8, 0);

    circle(field, Point((robot_position[1]),(robot_position[0])), 5, Scalar(255, 0, 0), 15, 8, 0);
    int x=round(25*cos(orientation * PI / 180));
    int y=round(25*sin(orientation * PI / 180));
    line(field,Point((robot_position[1]),(robot_position[0])),Point(robot_position[1]+x,robot_position[0]+((-1)*y)),Scalar(255, 0, 0), 2, 8, 0);
}

void tuning(Mat image){
    int downsampled=0;
    while(image.rows>400 && image.cols>400){
        downsampled++;
        //ROS_INFO("Downsample count = %d",downsampled);
        GaussianBlur( image, image, Size( 3, 3 ), 0, 0);
        pyrDown( image, image, Size( image.cols/2, image.rows/2 ) );
    }

    namedWindow("Tuning Window");
    createTrackbar("Tuning ball", "Tuning Window", &balltuning, 1, NULL);
    createTrackbar("Tuning object", "Tuning Window", &objecttuning, 1, NULL);
    createTrackbar("Tuning opponents", "Tuning Window", &opponenttuning, 1, NULL);
    createTrackbar("Tuning teammembers", "Tuning Window", &teammembertuning, 1, NULL);

    Mat hsvFrame;
    cvtColor(image, hsvFrame, CV_BGR2HSV);

    setMouseCallback("Tuning Window", callBackFunc, &hsvFrame);

    if(ball_tuner.x!=0.0){
        Mat selection = hsvFrame(Range(ball_tuner.y,ball_tuner.y+5.0), Range(ball_tuner.x,ball_tuner.x+5.0));
        Scalar averageHSV = mean(selection); /// get the average HSV in the selection area
        Vec3i avgHSV = Vec3i(int(averageHSV.val[0]), int(averageHSV.val[1]),int(averageHSV.val[2])); /// convert scalar to vec3i
        Mat thresholdFrame;
        namedWindow("HSV threshold", WINDOW_NORMAL);
        inRange(hsvFrame, avgHSV-gainHSV, avgHSV+gainHSV, thresholdFrame);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        morphologyEx(thresholdFrame, thresholdFrame, 3, element);
        imshow("HSV threshold", thresholdFrame);
        Canny(thresholdFrame, thresholdFrame, 100, 255, 3);
        findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        if(balltuning==1){
            ballHSV=avgHSV;
        }else if(objecttuning==1){
            obstaclesHSV=avgHSV;
        }else if(opponenttuning==1){
            opponentHSV=avgHSV;
        }else if(teammembertuning==1){
            teammemberHSV=avgHSV;
        }

        ball_tuner.x=0.0;
        ball_tuner.y=0.0;
    }

    imshow("Tuning Window", image);
}

int main(void) 
{
	
	bool ball_found; 
	float avg=0.0;
	int count=0;
	
	//bool recalibrate = 1;  

	//vision.initGUI();

	follower = Follower(); 

	int userInput = 0; 

    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGQUIT, &sighandler);
    signal(SIGINT, &sighandler);

    change_current_dir();

    ini_f = new minIni(INI_FILE_PATH);
    ini_b = new minIni(INI_FILE_PATH2);

    rgb_output = new Image(Camera::WIDTH, Camera::HEIGHT, Image::RGB_PIXEL_SIZE);

    LinuxCamera::GetInstance()->Initialize(0);
    LinuxCamera::GetInstance()->SetCameraSettings(CameraSettings());    // set default camera setting

    
    //////////////////// Framework Initialize ////////////////////////////
    if (MotionManager::GetInstance()->Initialize(&cm730) == false) {
        linux_cm730.SetPortName(U2D_DEV_NAME1);
        if (MotionManager::GetInstance()->Initialize(&cm730) == false) {
            printf("Fail to initialize Motion Manager!\n");
            return 0;
        }
    }

    MotionManager::GetInstance()->AddModule((MotionModule *) Action::GetInstance());
    MotionManager::GetInstance()->AddModule((MotionModule *) Head::GetInstance());
    MotionManager::GetInstance()->AddModule((MotionModule *) Walking::GetInstance());

    LinuxMotionTimer *motion_timer = new LinuxMotionTimer(MotionManager::GetInstance());
    motion_timer->Start();


    int firm_ver = 0;
    if (cm730.ReadByte(JointData::ID_HEAD_PAN, MX28::P_VERSION, &firm_ver, 0) != CM730::SUCCESS) {
        fprintf(stderr, "Can't read firmware version from Dynamixel ID %d!! \n\n", JointData::ID_HEAD_PAN);
        exit(0);
    }
    Action::GetInstance()->LoadFile((char *) MOTION_FILE_PATH);
    Action::GetInstance()->m_Joint.SetEnableBody(true, true);
    MotionManager::GetInstance()->SetEnable(true);

    cm730.WriteByte(CM730::P_LED_PANNEL, 0x01 | 0x02 | 0x04, NULL);
    
    Action::GetInstance()->Start(1);
    while (Action::GetInstance()->IsRunning()) usleep(8 * 1000);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// END OF INITIALIZING ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// STATUS BUTTON LOOP ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	while(1) {

		StatusCheck::Check(cm730);

		LinuxCamera::GetInstance()->CaptureFrame();
		memcpy(rgb_output->m_ImageData, LinuxCamera::GetInstance()->fbuffer->m_RGBFrame->m_ImageData,
		LinuxCamera::GetInstance()->fbuffer->m_RGBFrame->m_ImageSize);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		vision.rawFrame = cv::Mat(rgb_output->m_Height, rgb_output->m_Width, CV_8UC3, rgb_output->m_ImageData);
        Mat org=vision.rawFrame.clone();
        field=imread("SoccerField.png");
        
		//------------------- Algorithm for detecting regions
        //regionsDet(org);
        
		tuning(org);
	    updateMap();
        namedWindow("Field", WINDOW_NORMAL);
        imshow("Field", field);
		
		if(waitKey(30) == 27){
				
			destroyAllWindows();
			Walking::GetInstance()->Stop(); 
			break;
		}
		
	
	///////////////////////////// ACTION ////////////////////////////////////////////////////////////////////////

		if( StatusCheck::m_cur_mode == START )
		{
			while (button)
			{           
 
				button = 0;

				/// Load in Page number that is same as walking Stance
				/// To avoid sudden jerks after tuning walk.

				Action::GetInstance()->Start(9); /// Basketball Walk Ready Page
				while (Action::GetInstance()->IsRunning()) usleep(8 * 1000);

				/// Re-Initialize Head / Walking before able to start walk
				Head::GetInstance()->m_Joint.SetEnableHeadOnly(true, true);
				Walking::GetInstance()->m_Joint.SetEnableBodyWithoutHead(true, true);
				MotionManager::GetInstance()->SetEnable(true);
            
				Head::GetInstance()->MoveByAngle(-30,30); 
				Head::GetInstance()->m_LeftLimit = 25;
				Head::GetInstance()->m_RightLimit = -25; 
				Head::GetInstance()->m_TopLimit = 40; 
				Head::GetInstance()->m_BottomLimit = -5;
            
				cout << "Initializing body complete" << endl;
			}
		
			/// Start button pressed
			if( StatusCheck::m_old_btn == 2 ){
                //cout << "START PROGRAM -- button pressed" << endl;

                gettimeofday(&a, 0);
				pressed = true;
				usleep(8 * 1000);
			}

			
			//If started
            //if(noObstacles) 
			if( pressed  )
			{
				
				//ball_found = tracker.ball_position.X != 0 && tracker.ball_position.Y != 0;
                
				Walking::GetInstance()->BALANCE_ENABLE = true;
                if(step_cycle<1){
				if(step1){
	                opponent_position_right.clear();
	                obstacles_position_right.clear();
	                teammember_position_right.clear();

	                detectBall(org);
				    detectOpponent(org);
				    detectTeammember(org);
				    detectObstacles(org);
				    for(int i=0;i<opponent_position.size();i++)
				         opponent_position_right.push_back(opponent_position[i]);
				
				    for(int i=0;i<teammember_position.size();i++)
				         teammember_position_right.push_back(teammember_position[i]);
				         
				    for(int i=0;i<obstacles_position.size();i++)
				         obstacles_position_right.push_back(obstacles_position[i]);
	                
	                obstacles_position.clear();
	                opponent_position.clear();
	                teammember_position.clear();
	
			        gettimeofday(&b, 0);
                    difference = b.tv_sec - a.tv_sec;
                    if(difference>3){
               
			           step1=0;
			           step2=1;
			           Head::GetInstance()->MoveByAngle(30, 30);
	                   orientation+=60;
					   gettimeofday(&a, 0);
				    }
				}
				if(step2){
					teammember_position.clear();
			        opponent_position.clear();
			        obstacles_position.clear();
			        
			        detectBall(org);
			        detectOpponent(org);
			        detectTeammember(org);
			        detectObstacles(org);
			        
			        gettimeofday(&b, 0);
                    difference = b.tv_sec - a.tv_sec;
                    if(difference>3){
					   step2=0;
					   step3=1;
					   gettimeofday(&a, 0);
				    }
				}
				if(step3){
					gettimeofday(&b, 0);
                    difference = b.tv_sec - a.tv_sec;
                    Walking::GetInstance()->LoadINISettings(ini_f);
					MotionManager::GetInstance()->LoadINISettings(ini_f);
				    //Walking::GetInstance()->X_MOVE_AMPLITUDE=5;
					Walking::GetInstance()->Start();
					if(difference>3){
						Walking::GetInstance()->Stop();
						step3=0;
						step1=1;
                                                step_cycle++;
						Head::GetInstance()->MoveByAngle(-30, 30);
	                    orientation-=60;
                            robot_position[0]-=25;
	                    gettimeofday(&a, 0);
					}
				}
                }
				
				if (!ball_found){
						
					if(DEBUG)
						cout << "lose target" <<endl;

					//Walking::GetInstance()->A_MOVE_AMPLITUDE = 0; 
				
				}
			}//pressed
		}
  }//end while Status Check for buttons
}//end main
