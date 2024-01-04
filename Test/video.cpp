#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string videoStreamAddress = "http://172.16.2.135:4747/video";

    cv::VideoCapture cap(videoStreamAddress);

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream." << std::endl;
        return -1;
    }

    cv::Mat frame;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Frames per second: " << fps << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Process the frame here

        cv::imshow("Frame", frame);

        // Press 'q' to exit the loop
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
