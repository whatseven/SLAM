#pragma once

#include <opencv2/opencv.hpp>

int ratio_test(std::vector<std::vector<cv::DMatch> >& v_matches)
{
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch> >::iterator
		it = v_matches.begin(); it != v_matches.end(); ++it)
	{
		// if 2 NN has been identified
		if (it->size() > 1)
		{
			// check distance ratio
			if ((*it)[0].distance / (*it)[1].distance > .8f)
			{
				it->clear(); // remove match
				removed++;
			}
		}
		else
		{ // does not have 2 neighbours
			it->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

void symmetry_test(const std::vector<std::vector<cv::DMatch> >& v_matches1,
	const std::vector<std::vector<cv::DMatch> >& v_matches2,
	std::vector<cv::DMatch>& o_matches)
{
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch> >::const_iterator
		it1 = v_matches1.begin(); it1 != v_matches1.end(); ++it1)
	{
		// ignore deleted matches
		if (it1->empty() || it1->size() < 2)
			continue;

		// for all matches image 2 -> image 1
		for (std::vector<std::vector<cv::DMatch> >::const_iterator
			it2 = v_matches2.begin(); it2 != v_matches2.end(); ++it2)
		{
			// ignore deleted matches
			if (it2->empty() || it2->size() < 2)
				continue;

			// Match symmetry test
			if ((*it1)[0].queryIdx == (*it2)[0].trainIdx &&
				(*it2)[0].queryIdx == (*it1)[0].trainIdx)
			{
				// add symmetrical match
				o_matches.push_back(cv::DMatch((*it1)[0].queryIdx,
					(*it1)[0].trainIdx, (*it1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

void detect_keypoints(const cv::Mat& v_img_1, const cv::Mat& v_img_2, const cv::Mat& v_mask_1, const cv::Mat& v_mask_2,
	std::vector<cv::KeyPoint>& o_keypoints1, std::vector<cv::KeyPoint>& o_keypoints2,
	cv::Mat& o_descriptor1, cv::Mat& o_descriptor2,
	std::vector<cv::DMatch>& o_good_matches
)
{
	cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create(1000);
	//cv::Ptr<cv::Feature2D> extractor = cv::ORB::create(2000);
	
	extractor->detectAndCompute(v_img_1, v_mask_1, o_keypoints1, o_descriptor1);
	extractor->detectAndCompute(v_img_2, v_mask_2, o_keypoints2, o_descriptor2);

	std::vector<std::vector<cv::DMatch>> matches1, matches2;
	//cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();
	cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
	matcher->knnMatch(o_descriptor1, o_descriptor2, matches1, 2);
	matcher->knnMatch(o_descriptor2, o_descriptor1, matches2, 2);

	int removed1 = ratio_test(matches1);
	// clean image 2 -> image 1 matches
	int removed2 = ratio_test(matches2);
	// 4. Remove non-symmetrical matches
	symmetry_test(matches1, matches2, o_good_matches);
}


cv::Point2f pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
	return cv::Point2f
	(
		(p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
		(p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
	);
}

void triangulation(
	const std::vector<cv::KeyPoint>& keypoint_1,
	const std::vector<cv::KeyPoint>& keypoint_2,
	const std::vector<cv::DMatch>& matches,
	const cv::Mat& R, const cv::Mat& t,
	std::vector< cv::Point3f >& points, const cv::Mat& K)
{
	cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);

	std::vector<cv::Point2f> pts_1, pts_2;
	for (cv::DMatch m : matches)
	{
		pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
		pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
	}

	cv::Mat pts_4d;
	cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

	for (int i = 0; i < pts_4d.cols; i++)
	{
		cv::Mat x = pts_4d.col(i);
		x /= x.at<float>(3, 0); // πÈ“ªªØ
		cv::Point3f p(
			x.at<float>(0, 0),
			x.at<float>(1, 0),
			x.at<float>(2, 0)
		);
		points.push_back(p);
	}

}


void reconstruct_points(const cv::Mat& v_img1, const cv::Mat& v_img2, const cv::Mat& v_img_mask1, const cv::Mat& v_img_mask2, const cv::Mat& K,
	std::vector<cv::KeyPoint>& o_keypoints1, std::vector<cv::KeyPoint>& o_keypoints2,
	cv::Mat& o_descriptor1, cv::Mat& o_descriptor2,
	std::vector<cv::DMatch>& o_good_matches,
	std::vector<cv::Point3f>& o_points,
	cv::Mat& R, cv::Mat& t
	)
{
	//detect_keypoints(img1, img2, img_mask1, img_mask2, keypoints1, keypoints2,
	detect_keypoints(v_img1, v_img2, v_img_mask1, v_img_mask2, o_keypoints1, o_keypoints2,
		o_descriptor1, o_descriptor2, o_good_matches);

	//cv::Mat show;
	//cv::drawKeypoints(img1, keypoints1, show);
	//cv::drawMatches(v_img1, o_keypoints1, v_img2, o_keypoints2, o_good_matches, show);
	//cv::namedWindow("test", cv::WINDOW_NORMAL);
	//cv::imshow("test", show);
	//cv::waitKey();
	
	std::vector<cv::Point2f> points1, points2;
	for (int i = 0; i < (int)o_good_matches.size(); i++)
	{
		points1.push_back(o_keypoints1[o_good_matches[i].queryIdx].pt);
		points2.push_back(o_keypoints2[o_good_matches[i].trainIdx].pt);
	}

	//cv::Mat fundamental_matrix;
	//fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);

	cv::Mat valid_mask;
	cv::Mat essential_matrix;
	
	essential_matrix = findEssentialMat(points1, points2, K,cv::LMEDS,0.999,1.0, valid_mask);

	//cv::Mat homography_matrix;
	//homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);

	//std::cout << valid_mask<< std::endl;
	//std::cout << K << std::endl;
	recoverPose(essential_matrix, points1, points2, K, R, t,valid_mask);
	//std::cout << valid_mask << std::endl;
	//std::cout << t << std::endl;
	//std::cout << R << std::endl;
	//std::cout << K << std::endl;

	points1.erase(std::remove_if(points1.begin(), points1.end(), [idx = 0, &valid_mask](auto item)mutable
	{
		bool result = valid_mask.at<bool>(idx);
		idx += 1;
		return result;
	}), points1.end());
	points2.erase(std::remove_if(points2.begin(), points2.end(), [idx = 0, &valid_mask](auto item)mutable
	{
		bool result = valid_mask.at<bool>(idx);
		idx += 1;
		return result;
	}), points2.end());
	
	triangulation(o_keypoints1, o_keypoints2, o_good_matches, R, t, o_points, K);

	o_points.erase(std::remove_if(o_points.begin(), o_points.end(), [](cv::Point3f& item)
	{
			return item.z < 0 || item.z>50;
	}), o_points.end());
	
	return;
}