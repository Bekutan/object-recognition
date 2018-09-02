#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <vector>
#include <string>
#include <chrono>
/// K-means
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

//K=1 SIFT
//K=2 SURF
//K=3 ORB
//K=4 Blob+SURF
const int K=4;
int spasiti=0;
std::vector<Point2d> pts_src;
std::vector<Point2d> pts_dst;
cv::Mat h;
std::vector<KeyPoint> keypoints_1, keypoints_2; 
ofstream myfile;
cv::Mat img_1, img_2;
SimpleBlobDetector::Params params;
vector<vector<Point> > pom_konture;
std::vector<Point2f> centri;

bool spojiti=false;

class tacka
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	tacka(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		
  for(int i = 0; i < total_values; i++)
    this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<tacka> points;

public:
	Cluster(int id_cluster, tacka point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	void addPoint(tacka point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	tacka getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(tacka point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(std::vector<tacka> & points)
	{
		if(K > total_points)
			return;

		std::vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = std::rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}

		int iter = 1;
		while(true)
		{
			bool done = true;

			// associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}

			// recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			if(done == true || iter >= max_iterations)
			{
				std::cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}

		// shows elements of clusters
    keypoints_1.resize(K);
		for(int i = 0; i < K; i++)
		{
			/*
      int total_points_cluster =  clusters[i].getTotalPoints();

			std::cout << "Cluster " << clusters[i].getID() + 1 << std::endl;
			for(int j = 0; j < total_points_cluster; j++)
			{
				std::cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
					std::cout << clusters[i].getPoint(j).getValue(p) << " ";

				std::string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					std::cout << "- " << point_name;

				std::cout << endl;
			}

			std::cout << "Cluster values: ";
      */
			
			keypoints_1[i].pt.x = clusters[i].getCentralValue(0);
      		keypoints_1[i].pt.y = clusters[i].getCentralValue(1);

		}
	}
};

class ImageConverter
{
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::NodeHandle nh_2;
  image_transport::ImageTransport it_2;
  image_transport::Subscriber image_sub_2;
  //cv::Mat img_1, img_2;
  int brojac1 = 0, brojac2 = 0, brojac = 0;//zadnji za orijentaciju brojac
  std::vector<double> DetCom1;//prosjek vremena
  std::vector<double> DetCom2;
  std::vector<double> bfm;
  int svaka_dvadeseta = 0;//da mi malo isfiltrira upis hu momenata
  double hu[7];
  std::string orijentacija = "";
  std::chrono::high_resolution_clock::time_point vrijeme = std::chrono::high_resolution_clock::now();//za upis vremea u csv
  bool inicijalizacija = true; // za dodjeljivanje prvih orijentacija
  std::vector<int> stanje = {1,1,1,1,1,1,1,1,1,1};
  std::vector<int> stanje0 = {0,0,0,0,0,0,0,0,0,0};
  int broj_kontura = 0;
public:
  ImageConverter()
    : it_(nh_), it_2(nh_2)
  {
    if (!brojac1){
		homografija();
        myfile.open ("example.csv");
	} 
    image_sub_ = it_.subscribe("/camera/image_raw", 1, &ImageConverter::imageCb, this);
    image_sub_2 = it_2.subscribe("/camera/thermal_image_view", 1, &ImageConverter::imageCb, this);
    
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
	myfile.close();
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    std::cout << msg->encoding << std::endl;
    cv_bridge::CvImageConstPtr cv_ptr1, cv_ptr2;
    try
    {
      if (msg->encoding == "bgr8"){
        cv_ptr1 = cv_bridge::toCvCopy(msg, enc::BGR8);

        //if (spasiti==100) imwrite("/home/edina/cpp_test/bazna1.jpg", img_1);
        //else spasiti++;
        img_1 = (cv_ptr1->image).clone();
		imshow("prva", img_1);
		waitKey(3);
//---------------------
		imwrite("/home/edina/Pictures/mix/rgb_org.jpg", img_1);
        ROS_INFO("%d %d PRVA", (cv_ptr1->image).rows, (cv_ptr1->image).cols); 

        if (brojac1 & brojac2) detekcija();
        else brojac1++;

      }
      else{
        cv_ptr2 = cv_bridge::toCvCopy(msg, enc::RGB8);
        //if (spasiti==100) imwrite("/home/edina/cpp_test/bazna_term.jpg", img_2);
        //else spasiti++;
        ROS_INFO("%d %d druga", (cv_ptr2->image).rows, (cv_ptr2->image).cols);
        img_2 = (cv_ptr2->image).clone(); 
		//im_out = (cv_ptr2->image).clone(); 
        //if ( spasiti == 100 ) imwrite("/home/edina/cpp_test/bazna2.jpg", img_2);

  
        if (brojac1 & brojac2) detekcija();
        else brojac2++;

      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("OVDJE cv_bridge exception: %s", e.what());
      return;
    }



  }

  void parametri()
  {

  

      params.filterByColor = true;
	  params.blobColor = 255;


	  // Change thresholds
	  params.minThreshold = 30;
	  params.maxThreshold = 250;

	  // Filter by Area.
	  params.filterByArea = false;
	  //params.minArea = 1500;

	  // Filter by Circularity
	  params.filterByCircularity = true;
	  params.minCircularity = 0.02;

	  // Filter by Convexity
	  params.filterByConvexity = true;
	  params.minConvexity = 0.4;

	  // Filter by Inertia
	  params.filterByInertia = true;
	  params.minInertiaRatio = 0.01;
    
  }
  
  void det(Mat &img_1, int a=0) //za homografiju
  {

      SimpleBlobDetector::Params params;
	
	  params.filterByColor = true;
	  if(a)  params.blobColor = 255;
	  else  params.blobColor = 0;
	 


	  // Change thresholds
	  params.minThreshold = 30;
	  params.maxThreshold = 250;

	  // Filter by Area.
	  params.filterByArea = true;
	  params.minArea = 6;

	  // Filter by Circularity
	  params.filterByCircularity = true;
	  params.minCircularity = 0.02;

	  // Filter by Convexity
	  params.filterByConvexity = true;
	  params.minConvexity = 0.4;

	  // Filter by Inertia
	  params.filterByInertia = true;
	  params.minInertiaRatio = 0.01;


	  // Storage for blobs
	  std::vector<KeyPoint> keypoints, novi;

	  // Set up detector with params
	  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   
	  // Detect blobs
	  detector->detect( img_1, keypoints);
	  if(a){

	      pts_src.push_back(keypoints[1].pt);
    	  pts_src.push_back(keypoints[3].pt);
    	  pts_src.push_back(keypoints[2].pt);
    	  pts_src.push_back(keypoints[5].pt);
	      pts_src.push_back(keypoints[15].pt);
    	  pts_src.push_back(keypoints[20].pt);
    	  pts_src.push_back(keypoints[19].pt);
		  pts_src.push_back(keypoints[8].pt);
    	  pts_src.push_back(keypoints[30].pt);
    	  pts_src.push_back(keypoints[27].pt);
		  novi.push_back(keypoints[0]);	
		  novi.push_back(keypoints[1]);
		  novi.push_back(keypoints[3]);
		  novi.push_back(keypoints[2]);
		  novi.push_back(keypoints[5]);
		  novi.push_back(keypoints[15]);
		  novi.push_back(keypoints[20]);
		  novi.push_back(keypoints[19]);
		  novi.push_back(keypoints[8]);
	      novi.push_back(keypoints[30]);
		  novi.push_back(keypoints[27]);

	  }else{

	      pts_dst.push_back(keypoints[1].pt);
          pts_dst.push_back(keypoints[2].pt);
          pts_dst.push_back(keypoints[3].pt);
      	  pts_dst.push_back(keypoints[5].pt);
          pts_dst.push_back(keypoints[9].pt);
          pts_dst.push_back(keypoints[15].pt);
          pts_dst.push_back(keypoints[14].pt);
		  pts_dst.push_back(keypoints[20].pt);
          pts_dst.push_back(keypoints[18].pt);
          pts_dst.push_back(keypoints[19].pt);
		  novi.push_back(keypoints[0]);		
		  novi.push_back(keypoints[1]);
		  novi.push_back(keypoints[2]);
		  novi.push_back(keypoints[3]);
		  novi.push_back(keypoints[5]);
		  novi.push_back(keypoints[9]);
		  novi.push_back(keypoints[15]);
		  novi.push_back(keypoints[14]);
		  novi.push_back(keypoints[20]);
          novi.push_back(keypoints[18]);
		  novi.push_back(keypoints[19]);
	  }
	  Mat im_with_keypoints;
	  drawKeypoints( img_1, novi, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	  //drawKeypoints( img_1, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	  //if(a)imshow("sliba1", im_with_keypoints); 
	  //else imshow("sliba2", im_with_keypoints);
	  //waitKey(3);
  }

  void homografija()
  {
	Mat im_out, merge;//za prikaz
    Mat im_dst = imread("/home/edina/catkin_ws/src/test_pkg/src/tresholding.jpg");
    Mat im_src = imread("/home/edina/catkin_ws/src/test_pkg/src/termalna.png");  
	//hconcat(im_dst, im_src, merge);

    det(im_src,1);
	det(im_dst);

    h = findHomography(pts_src, pts_dst);
    
    warpPerspective(im_src, im_out, h, im_dst.size());
	
    //imshow("Warped Source Image", im_out);
	//imshow("sliba", merge);
    //waitKey(3);
  }

  void filter(std::vector<KeyPoint> &rgb, std::vector<KeyPoint> &termalna)
  {
    std::vector<KeyPoint> pom;
    for(int i=0; i<termalna.size(); i++){
      for(int j=0; j<rgb.size(); j++){
        if(rgb[j].pt.x > termalna[i].pt.x-10 & rgb[j].pt.x < termalna[i].pt.x + 10 &rgb[j].pt.y > termalna[i].pt.y-10 & rgb[j].pt.y < termalna[i].pt.y + 10)
          pom.push_back(rgb[j]);
      }
    }
    rgb = pom;
  }

  vector<vector<Point> > konture()
  {
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();  
    int thresh = 150;
    RNG rng(12345);
    Mat tresholded = img_1.clone();
	Mat obrada = img_2.clone();
	warpPerspective(obrada, obrada, h, img_1.size());
	Mat temperatura = obrada, obicna = obrada;

	cvtColor(tresholded, tresholded, CV_RGB2GRAY);
	cvtColor(obrada, obrada, CV_RGB2GRAY);
	cvtColor(obicna, obicna, CV_RGB2GRAY);
	cvtColor(temperatura, temperatura, CV_RGB2GRAY);
	//imshow("pretvorena", tresholded);
	//waitKey(3);
	GaussianBlur(tresholded, tresholded, Size(3, 3), 0, 0);
    threshold( tresholded, tresholded, 238, 255, 0 );
    erode(tresholded, tresholded, Mat() );
    dilate(tresholded, tresholded, Mat());

	threshold( obrada, obrada, 17, 255, 0 );
	threshold( obicna, obicna, 100, 255, 0 );
//---------------
	imshow("obicna", obicna);
	imwrite("/home/edina/Pictures/mix/obicna.jpg", obicna);
	waitKey(3);

    Mat canny_output;
    vector<vector<Point> > contours, contours2, obicna_kont;
    vector<Vec4i> hierarchy;  
    Canny( tresholded, canny_output, thresh, thresh*3, 3 );
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //findContours( tresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Canny( obrada, obrada, thresh, thresh*3, 3 );
    findContours( obrada, contours2, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );
	cv::drawContours(obrada, contours2, -1, cv::Scalar(250,250,0), -1);
//---------------------
	imshow("obrada", obrada);
	waitKey(3);
	imwrite("/home/edina/Pictures/mix/obrada.jpg", obrada);

	Canny( obicna, obicna, thresh, thresh*3, 3 );
    findContours( obicna, obicna_kont, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );
	// Get the moments and mass centers
    std::vector<Moments> mu(contours.size());
	std::vector<Point2f> mc(contours.size());

	std::vector<Moments> mu2(contours2.size());
	std::vector<Point2f> mc2(contours2.size());

	std::vector<Moments> obicna_momenti(obicna_kont.size());
	std::vector<Point2f> obicna_centar(obicna_kont.size());

	std::vector<double> poluprecnik(contours2.size());
	std::vector<Point2d> ispis(contours2.size());

	std::vector<double> obicna_pol(obicna_kont.size());


	//ISPROBATI I ZA FALSE KOD MOMENATA
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], true);
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
	    for (int i = 0; i < contours2.size(); i++)
    {
        mu2[i] = moments(contours2[i], true);
		mc2[i] = Point2f(mu2[i].m10 / mu2[i].m00, mu2[i].m01 / mu2[i].m00);

		poluprecnik[i] = std::sqrt(contourArea(contours2[i])/3.14193);
    }

		    for (int i = 0; i < obicna_kont.size(); i++)
    {
        obicna_momenti[i] = moments(obicna_kont[i], true);
		obicna_centar[i] = Point2f(obicna_momenti[i].m10 / obicna_momenti[i].m00, obicna_momenti[i].m01 / obicna_momenti[i].m00);

		obicna_pol[i] = std::sqrt(contourArea(obicna_kont[i])/3.14193);
    }

	std::vector<float> cont_avgs(contours2.size(), 0.f);
	std::vector<int> counts(contours2.size());
	Mat pomoc(120,160, CV_8UC1, Scalar(0));
	//Mat pomoc = cv::Mat::zeros(img_1.size(), CV_8UC3);

	for(int i = 0; i < contours2.size(); i++){
		for(int j = -(int)poluprecnik[i]-10; j < (int)poluprecnik[i]+10; j++){
			if(mc2[i].y + j > 0 & mc2[i].y + j < 120) {
				for(int k = -(int)poluprecnik[i]-10; k < (int)poluprecnik[i]+10; k++){
					if(mc2[i].x + k > 0 & mc2[i].x + k < 160  ){
						uchar value = temperatura.data[((int)mc2[i].y + j)*160 +(int) mc2[i].x + k ];
						
						cont_avgs[i] += int(value);

						if ((int)value > 3 ) counts[i]++;


					}
				 
				}

			}

		}

		//odavde zapisati cont_avgs
		cont_avgs[i] /= counts[i];

	}

	std::vector<float> cont_avgs1(obicna_kont.size(), 0.f);
	std::vector<int> counts1(obicna_kont.size());

	std::vector<std::vector<int>> matrica(100, std::vector<int>(100));
	for(int i = 0; i < obicna_kont.size(); i++){
		for(int j = -(int)obicna_pol[i]-10; j < (int)obicna_pol[i]+10; j++){
			if(obicna_centar[i].y + j > 0 & obicna_centar[i].y + j < 120) {
				for(int k = -(int)obicna_pol[i]-10; k < (int)obicna_pol[i]+10; k++){
					if(obicna_centar[i].x + k > 0 & obicna_centar[i].x + k < 160  ){
						uchar value = temperatura.data[((int)obicna_centar[i].y + j)*160 +(int)obicna_centar[i].x + k ];
						pomoc.data[((int)obicna_centar[i].y + j)*160 +(int)obicna_centar[i].x + k ] = value;
						if ((int)value > 140 ){
								counts1[i]++;
								cont_avgs1[i] += int(value);
						} 
					}
				 
				}

			}

		}
		cont_avgs1[i] /= counts1[i];
		for(int j=0; j<contours2.size(); j++){

			if(obicna_centar[i].x > mc2[j].x - 10 & obicna_centar[i].x < mc2[j].x + 10 & obicna_centar[i].y > mc2[j].y - 10 & obicna_centar[i].y < mc2[j].y + 10  ){
				cont_avgs[j] = cont_avgs1[i];
			}

			
			if( i == obicna_kont.size()-1)
				std::cout << 40./255*cont_avgs[j]+20 << "*.*.*.*.";

		}
	}
	std::cout<<std::endl;



	    //******* filter kontura
		// mc[j].x > keypoints_2[i].pt.x-20 & mc[j].x < keypoints_2[i].pt.x + 20 & mc[j].y > keypoints_2[i].pt.y-20 & mc[j].y < keypoints_2[i].pt.y + 20
	
	///*** ovdje ide sad poluprečnik konture
	
	vector<vector<Point> > pom, pom2;
    for(int i = 0; i < mc2.size(); i++){
		//poluprecnik[i] = std::sqrt(contourArea(contours2[i])/3.14193);
      for(int j = 0; j < contours.size(); j++){
        if(!spojiti & mc[j].x > mc2[i].x-20 & mc[j].x < mc2[i].x + 20 & mc[j].y > mc2[i].y-20 & mc[j].y < mc2[i].y + 20){
          spojiti=true;
		  pom.push_back( contours[j] );
		  pom2.push_back( contours2[i]);
		  centri.push_back(mc2[i]);
		  }
      }
	  spojiti=false;
    }

	//SORTIRANE KONTURE PREMA KONTURAMA TERMALNE
    contours = pom;
	contours2 = pom2;
	//***************

	broj_kontura = contours.size();
	cv::Mat labels = cv::Mat::zeros(img_1.size(), CV_8UC3); 
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i < contours.size(); i++ )
       {

		HuMoments(mu[i], hu);
		
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
        drawContours( drawing, contours, (int)i, color, -1, 4, hierarchy, 0, Point() ); //bilo 3 maxlevel
	    //drawContours( img_1, contours, (int)i, Scalar(), 1, 4); //bilo 3 maxlevel

		//if(mc[i].x > keypoints_2[0].pt.x-10 & mc[i].x < keypoints_2[0].pt.x + 10 & mc[i].y > keypoints_2[0].pt.y-10 & mc[i].y < keypoints_2[0].pt.y + 10)
		
		
		//Point pCoordinates(mc[i].x-int(mc[i].x)%50-3, mc[i].y - int(mc[i].y)%50-3);
		//Point pCoordinates(50, 50);
		
		//putText(img_1, orijentacija, pCoordinates, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(), 1, 4); // Write object number
		//UMJESTO -1 IDE 2
		cv::drawContours( img_1, contours, (int)i, color, -1, 4, hierarchy, 0, Point() ); //bilo 3 maxlevel
		if(40./255*cont_avgs[i]>20) putText(img_1, "O", mc2[i], CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0,0), 1, 4);
		else putText(img_1, "L", mc2[i], CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0,0), 1, 4);
		//circle(img_1, mc[i], 2, color, -1, 8, 0);
       
	   }
	
		cv::drawContours(labels, contours2, -1, cv::Scalar(255,0,0), -1);

		
	//pom_konture=contours;
    //namedWindow( "Contours", WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
	waitKey(3);
//-----------------------
	imwrite("/home/edina/Pictures/mix/konture.jpg", drawing);
  	return contours;	
  }
  void spajanje(vector<vector<Point> > &contours)
  {
	Mat im_out, merge;
    warpPerspective(img_2, im_out, h, img_1.size());
	hconcat(img_1, im_out, merge);
	//************
	//imshow("sliba", merge);
    //waitKey(3);
	std::vector<Moments> mu(contours.size());
	std::vector<Point2f> mc(contours.size());;


	for( int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		
			
				line(merge, mc[i], Point2f(centri[i].x+160, centri[i].y), Scalar(150, 10, 0), 2);
				//line(merge, mc[i], Point2f(mc[i].x+155, mc[i].y-((-1)^i)*i*2), Scalar(0, 150, 0), 2);
			
		

		

	}
//----------------
	imshow("MyMatcher", merge );
	imwrite("/home/edina/Pictures/mix/spojeno.jpg", merge);
	waitKey(3);


  }
  void detekcija(){
    
    //std::vector<KeyPoint> keypoints_1, keypoints_2; odavde uzela
    Mat descriptors_1, descriptors_2;
    cv::Ptr<Feature2D> f2d;
    Mat im_out;
    warpPerspective(img_2, im_out, h, img_1.size());
	
    cv::imshow("slika2", img_2 );
//----------2
	imwrite("/home/edina/Pictures/mix/img_2.jpg", img_2);
	//cv::imshow( "nest", img_1 );
    cv::imshow("slika2pom", im_out );
	imwrite("/home/edina/Pictures/mix/termalna.jpg", im_out);
    cv::waitKey(3);
    if( K == 4 ) parametri();
    if( K == 1 ) f2d = xfeatures2d::SIFT::create();
    else if ( K == 2 ) f2d = xfeatures2d::SURF::create();
    else if ( K == 3 ) f2d = ORB::create();
    else if ( K == 4 ) f2d = SimpleBlobDetector::create(params);

    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //-- Step 1: Detect the keypoints:
    //-- Step 2: Calculate descriptors (feature vectors)     
    f2d->detect( img_1, keypoints_1 );
    if( K == 4 ) f2d = xfeatures2d::SURF::create(); 
  
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    if (K==4) f2d = SimpleBlobDetector::create(params);
    f2d->detect( im_out, keypoints_2 );

	vector<vector<Point> > contours = konture();

	//std::cout<<keypoints_1.size()<<"**********"<<std::endl;

    //*********************************
    filter(keypoints_1, keypoints_2);
    //*********************************
	//std::cout<<keypoints_1.size()<<"**********"<<std::endl;

	

    std::vector<tacka> tacke_za_kmeans;
    
    for(int i = 0; i < keypoints_1.size(); i++){
      std::vector<double> values;
      values.push_back(keypoints_1[i].pt.x);
      values.push_back(keypoints_1[i].pt.y); 
      tacka p(i, values);
      tacke_za_kmeans.push_back(p);
    }
	if (broj_kontura){
		std::cout<<"if if if "<<broj_kontura<<std::endl;
    	KMeans kmeans(broj_kontura, tacke_za_kmeans.size(), 2, 100);
    	kmeans.run(tacke_za_kmeans);
	}


	
  
    if(K == 4) f2d = xfeatures2d::SURF::create(); 
    f2d->compute( img_1, keypoints_1, descriptors_1 );
    f2d->compute( im_out, keypoints_2, descriptors_2 );
    std::cout<<keypoints_1.size()<<" <-prva    druga-> "<<keypoints_2.size()<<std::endl;
	//vratiti
    //Mat im_with_keypoints, im_with_keypoints_2;
    //drawKeypoints( img_1, keypoints_1, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	//drawKeypoints( im_out, keypoints_2, im_with_keypoints_2, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //imshow("keypoints1", im_with_keypoints );
	  //waitKey(30);
    //imshow("keypoints2", im_with_keypoints_2 );
	  //waitKey(30);

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
  
    double dif1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    double dif2 = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
    double dif3 = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();

    //-- Draw matches
    Mat img_matches;
    drawMatches( img_1, keypoints_1, im_out, keypoints_2, matches, img_matches );
    std::cout<< std::endl << "T1=" << dif1 << " T2= " << dif2 << " BFMatcher  T=" << dif3 << std::endl;
    //-- Show detected matches
    cv::imshow("Matches", img_matches );

    cv::waitKey(30);

    DetCom1.push_back(dif1);
    DetCom2.push_back(dif2);
    bfm.push_back(dif3);
	imwrite("/home/edina/Pictures/matches7.jpg", img_matches);
	spajanje (contours);

  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub");
  ImageConverter ic;
  ros::spin();
  return 0;
}