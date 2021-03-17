#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

void debug_img(const std::vector<cv::Mat>& v_imgs)
{
	cv::Mat show;

	std::vector<cv::Mat> imgs_3_channels;
	for(auto item:v_imgs)
	{
		if(item.channels()==1)
		{
			cv::Mat converted_img;
			cv::cvtColor(item, converted_img, cv::COLOR_GRAY2BGR);
			imgs_3_channels.push_back(converted_img);
		}
		else
			imgs_3_channels.push_back(item);
	}
	
	cv::hconcat(imgs_3_channels, show);
	
	cv::namedWindow("debug", cv::WINDOW_NORMAL);
	cv::imshow("debug", show);
	cv::waitKey();
	cv::destroyWindow("debug");
}