#include <iostream>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <CGAL/Point_set_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3/IO.h>

#include "common.h"
#include "features.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel;

boost::filesystem::path root("D:/DATASET/ApolloCarInstance/dynamic_sample/");
cv::Mat K = (cv::Mat_<float>(3, 3) << 576.136966392455, 0, 421.559469032005, 0, 576.4689170155, 338.7462160994775, 0, 0, 1);

const int pixel_threshold = 2000;

struct Frame
{
	std::string img_file;
	std::vector<std::string> mask_files;

	cv::Mat img;
	std::vector<cv::Mat> masks;
	cv::Mat background_mask;
	std::vector<int> mask_id_in_previous_frame;
};

void track_frame(Frame& v_frame1,Frame& v_frame2)
{
	std::vector<cv::KeyPoint> bg_keypoints1, bg_keypoints2;
	cv::Mat bg_descriptor1, bg_descriptor2;
	std::vector<cv::DMatch> bg_good_matches;
	std::vector<cv::Point3f> bg_points;
	cv::Mat bg_R, bg_t;
	reconstruct_points(v_frame1.img, v_frame2.img, v_frame1.background_mask, v_frame2.background_mask,
		K, bg_keypoints1, bg_keypoints2,
		bg_descriptor1, bg_descriptor2,
		bg_good_matches, bg_points, bg_R, bg_t
	);

	for(int id_mask2=0;id_mask2<v_frame2.masks.size();++id_mask2)
	{
		if(v_frame2.mask_id_in_previous_frame[id_mask2]==-1)
			continue;

		std::vector<cv::KeyPoint> item_keypoints1, item_keypoints2;
		cv::Mat item_descriptor1, item_descriptor2;
		std::vector<cv::DMatch> item_good_matches;
		std::vector<cv::Point3f> item_points;
		cv::Mat item_R, item_t;
		reconstruct_points(v_frame1.img, v_frame2.img, 
			v_frame1.masks[v_frame2.mask_id_in_previous_frame[id_mask2]], v_frame2.masks[id_mask2],
			K, item_keypoints1, item_keypoints2,
			item_descriptor1, item_descriptor2,
			item_good_matches, item_points, item_R, item_t
		);

		
		CGAL::Point_set_3<CGAL::Point_3<kernel>> point_set_3;
		for(auto& p: item_points)
			point_set_3.insert(CGAL::Point_3<kernel>(p.x, p.y, p.z));
		CGAL::write_ply_point_set(std::ofstream("temp.ply"), point_set_3);

		// -- Debug
		cv::Mat show(v_frame1.img.rows, v_frame1.img.cols,CV_8UC3,cv::Scalar(0,0,0));

		std::vector<cv::Point2f> points;
		for(auto item:item_points)
		{
			item /= item.z;
			cv::Mat pt = K * (cv::Mat_<float>(3, 1) << item.x, item.y, item.z);
			pt /= pt.at<float>(2, 0);
			cv::circle(show, cv::Point2f(pt.at<float>(0, 0), pt.at<float>(1, 0)), 1, cv::Scalar(0, 0, 255));
		}
		
		//cv::drawMatches(v_frame1.img, item_keypoints1, v_frame2.img, item_keypoints2, item_good_matches, show);
		debug_img({ v_frame1.img,v_frame2.img ,v_frame2.masks[id_mask2],show });
		continue;
		// -- Debug
	}
	
	
}

int main() {
	cv::Mat img1 = cv::imread((root / "180116_064519300_Camera_6.jpg").string(), cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread((root / "180116_064519424_Camera_6.jpg").string(), cv::IMREAD_GRAYSCALE);

	std::vector<Frame> frames;
	
	for (boost::filesystem::directory_iterator it(root.string());it != boost::filesystem::directory_iterator();++it)
	{
		//std::cout << it->path() << std::endl;
		std::string filename = it->path().filename().string();
		if(filename.find("mask")==std::string::npos)
		{
			frames.push_back(Frame());
			frames[frames.size() - 1].img_file=it->path().string();
		}
		else
		{
			frames[frames.size() - 1].mask_files.push_back(it->path().string());
		}
	}

	for(Frame& frame:frames)
	{
		frame.img = cv::imread(frame.img_file, cv::IMREAD_GRAYSCALE);
		frame.background_mask = cv::Mat::zeros(frame.img.rows, frame.img.cols, CV_8UC1);

		for(std::string mask_path:frame.mask_files)
		{
			frame.masks.push_back(cv::imread(mask_path, cv::IMREAD_UNCHANGED));
			cv::bitwise_or(frame.masks[frame.masks.size() - 1], frame.background_mask, frame.background_mask);

		}
		cv::bitwise_not(frame.background_mask, frame.background_mask);
		//debug_img({ frame.background_mask ,frame.img });
	}


	//reconstruct_points(img1, img2, img_mask1, img_mask2, K);
	Frame& frame1 = frames[0], frame2 = frames[1];

	std::vector<cv::KeyPoint> total_keypoints1, total_keypoints2;
	cv::Mat total_descriptor1, total_descriptor2;
	std::vector<cv::DMatch> total_good_matches;
	detect_keypoints(
		frame1.img, frame2.img,
		cv::Mat::ones(frame1.img.rows, frame1.img.cols, CV_8UC1), cv::Mat::ones(frame1.img.rows, frame1.img.cols, CV_8UC1),
		total_keypoints1, total_keypoints2, total_descriptor1, total_descriptor2, total_good_matches
	);
	for(int idx_mask2=0;idx_mask2< frame2.masks.size();idx_mask2++)
	{
		const cv::Mat& mask_img2 = frame2.masks[idx_mask2];
		if(cv::countNonZero(mask_img2)< pixel_threshold)
		{
			frame2.mask_id_in_previous_frame.push_back(-1);
			continue;
		}
		
		std::vector<int> point_id_in_previous_img;
		for(const cv::DMatch& match: total_good_matches)
		{
			if (mask_img2.at<cv::uint8_t>(total_keypoints2[match.trainIdx].pt.y, total_keypoints2[match.trainIdx].pt.x) != 0)
				point_id_in_previous_img.push_back(match.queryIdx);
		}
		std::vector<int> points_contain(frame1.masks.size(),0);
		for (int idx_mask1 = 0;idx_mask1 < frame1.masks.size();idx_mask1++)
		{
			for(int match_id:point_id_in_previous_img)
				if (frame1.masks[idx_mask1].at<cv::uint8_t>(total_keypoints1[match_id].pt.y, total_keypoints1[match_id].pt.x) != 0)
					points_contain[idx_mask1] += 1;
		}
		int mask_id_with_max_point = (std::max_element(points_contain.begin(), points_contain.end()) - points_contain.begin());
		if(points_contain[mask_id_with_max_point] > 3 && cv::countNonZero(frame1.masks[mask_id_with_max_point]) > pixel_threshold)
			frame2.mask_id_in_previous_frame.push_back(mask_id_with_max_point);
		else
			frame2.mask_id_in_previous_frame.push_back(-1);

		//std::cout << points_contain[mask_id_with_max_point] << std::endl;
		//debug_img({ frame1.masks[mask_id_with_max_point],frame2.masks[idx_mask2] });
	}

	// -- Debug
	//int id_current = -1;
	//for(int id_previous:frame2.mask_id_in_previous_frame)
	//{
	//	id_current += 1;
	//	if(id_previous==-1)
	//		continue;
	//	debug_img({ frame1.masks[id_previous],frame2.masks[id_current] });
	//}
	// -- Debug

	track_frame(frame1, frame2);
	
	//reconstruct_points(img1, img2, 
	//	img_mask1, img_mask2,
	//	//cv::Mat(img1.rows, img1.cols, CV_8UC1, 1), cv::Mat(img2.rows, img2.cols, CV_8UC1, 1), 
	//	K,keypoints1, keypoints2, descriptor1, descriptor2, good_matches, points,R,t
	//	);


	//cv::Mat show;
	//cv::drawKeypoints(img1, keypoints1, show);
	//cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, show);

	//cv::namedWindow("test", cv::WINDOW_NORMAL);
	//cv::imshow("test", show);
	//cv::waitKey();
	
	return 0;
}