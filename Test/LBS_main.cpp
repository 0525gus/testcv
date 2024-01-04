#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <format>

using namespace cv;
using namespace std;

struct Line {
    Point pt1, pt2;
};
void houghLinesCallback(int, void*);
void on_mouse(int event, int x, int y, int flags, void* userdata);

int rho_slider = 32;
int theta_slider = 180;
int threshold_slider = 104;
int CP_x = -1;
int CP_y = -1;

Mat img;
Mat edges;

vector<Line> linePoints;
vector<vector<Line>> clusters;

int main() {
    string PATH = "imgs/te.png";
    img = imread(PATH);
    double rsizeNum = 0.7;
    resize(img, img, {}, rsizeNum, rsizeNum);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(7, 7), 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    Canny(gray, edges, 0, 0);
    namedWindow("Hough Lines");
    imshow("Hough Lines", img);


    // 트랙바 
    createTrackbar("Rho * 10", "Hough Lines", &rho_slider, 100, houghLinesCallback);
    createTrackbar("Theta", "Hough Lines", &theta_slider, 360, houghLinesCallback);
    createTrackbar("Threshold", "Hough Lines", &threshold_slider, 500, houghLinesCallback);

    waitKey(0);
    return 0;
}

void houghLinesCallback(int, void*) {
    //auto start = std::chrono::high_resolution_clock::now();
    double rho = static_cast<double>(rho_slider) / 10.0;
    vector<Vec2f> lines;

    cv::HoughLines(edges, lines, rho, CV_PI / (theta_slider), threshold_slider);
    //cv::HoughLinesP(edges, lines, rho, CV_PI / (theta_slider), threshold_slider,550,30);

    Mat result = img.clone();

    linePoints.clear();
    clusters.clear();

    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 10 * (-b));
        pt1.y = cvRound(y0 + 10 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        linePoints.push_back({ pt1, pt2 });
    }

    for (const auto& line : linePoints) {
        bool foundCluster = false;
        for (auto& cluster : clusters) {
            const auto& representative = cluster.front();
            if (abs(representative.pt1.x - line.pt1.x) < 10 && abs(representative.pt1.y - line.pt1.y) < 10 &&
                abs(representative.pt2.x - line.pt2.x) < 10 && abs(representative.pt2.y - line.pt2.y) < 10) {
                cluster.push_back(line);
                foundCluster = true;
                break;
            }
        }

        if (!foundCluster) {
            clusters.push_back({ line });
        }
    }


    //cluster line draw
    for (auto& cluster : clusters) {
        int max_overlap = 0;
        Line representative;
        for (const auto& line1 : cluster) {
            int overlap = 0;
            for (const auto& line2 : cluster) {
                if (abs(line1.pt1.x - line2.pt1.x) < 10 && abs(line1.pt1.y - line2.pt1.y) < 10 &&
                    abs(line1.pt2.x - line2.pt2.x) < 10 && abs(line1.pt2.y - line2.pt2.y) < 10) {
                    ++overlap;
                }
            }
            if (overlap > max_overlap) {
                max_overlap = overlap;
                representative = line1;
            }
        }
        cluster = { representative };

        line(result, representative.pt1, representative.pt2, Scalar(0, 255, 0), 2, LINE_AA);
    }
    
    cv::setMouseCallback("Hough Lines", on_mouse);

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = end - start;
    //std::cout << "time = " << elapsed.count() << "second\n";

    imshow("Hough Lines", result);
}

double distanceFromPointToLine(Point pt, Point lineStart, Point lineEnd) {
    double numer = abs((lineEnd.y - lineStart.y) * pt.x - (lineEnd.x - lineStart.x) * pt.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x);
    double denom = sqrt(pow(lineEnd.y - lineStart.y, 2) + pow(lineEnd.x - lineStart.x, 2));
    return numer / denom;
}

float EQua(const float& m, const float& Base_x, const float& Base_y, const float& x) {
    return m * (x - Base_x) + Base_y;
}
void on_mouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        auto start = std::chrono::high_resolution_clock::now();
        cout << x << "," << y << endl;
        CP_x = x;
        CP_y = y;
        int ROI_COORD_VALUE = 240;
        //cout << "CP(" << x << "," << y << ")" << endl;

        Mat img_ROI;
        Mat img_with_keypoints;
        std::vector<cv::KeyPoint> keypoints;
        try {
            cv::Rect roi_rect(CP_x - ROI_COORD_VALUE, CP_y - ROI_COORD_VALUE, ROI_COORD_VALUE * 2, ROI_COORD_VALUE * 2);
            edges(roi_rect).copyTo(img_ROI);
            cv::SimpleBlobDetector::Params params;
            params.filterByArea = true;
            params.minArea = 30;
            auto start = std::chrono::high_resolution_clock::now();

            cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
            detector->detect(img_ROI, keypoints);

            //cv::Mat img_with_keypoints;
            cv::drawKeypoints(img_ROI, keypoints, img_with_keypoints, Scalar(0, 255, 0));

            for (size_t i = 0; i < keypoints.size(); i++) {
                keypoints[i].pt.x += CP_x - ROI_COORD_VALUE;
                keypoints[i].pt.y += CP_y - ROI_COORD_VALUE;
                float x = keypoints[i].pt.x;
                float y = keypoints[i].pt.y;
                //cout << "x = " << x << ", y = " << y << endl;
            }
            // detect된 keypoints
            cv::imshow("Keypoints", img_with_keypoints);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "time = " << elapsed.count() << "second\n";
            cv::waitKey(0);



        }
        catch (...) {
            cerr << "ROI range error but pass" << endl;
        }

        vector<pair<double, vector<Line>>> distances;
        for (const auto& cluster : clusters) {
            for (const auto& line : cluster) {
                double dist = distanceFromPointToLine(Point(x, y), line.pt1, line.pt2);
                //cout << dist << endl;
                distances.push_back(make_pair(dist, cluster));
            }
        }
        sort(distances.begin(), distances.end(), [](const pair<double, vector<Line>>& a, const pair<double, vector<Line>>& b) {
            return a.first < b.first;
            });

        vector<vector<Line>> closestLines;
        closestLines.push_back(distances[0].second);
        closestLines.push_back(distances[1].second);
        closestLines.push_back(distances[2].second);

        //test
        float X1_1, Y1_1, X2_1, Y2_1, X1_2, Y1_2, X2_2, Y2_2, X1_3, Y1_3, X2_3, Y2_3;
        {
        X1_1 = closestLines[0][0].pt1.x;
        Y1_1 = closestLines[0][0].pt1.y;
        X2_1 = closestLines[0][0].pt2.x;
        Y2_1 = closestLines[0][0].pt2.y;
        if (closestLines[1][0].pt1.y > closestLines[2][0].pt2.y) {
            X1_3 = closestLines[1][0].pt1.x;
            Y1_3 = closestLines[1][0].pt1.y;
            X2_3 = closestLines[1][0].pt2.x;
            Y2_3 = closestLines[1][0].pt2.y;

            X1_2 = closestLines[2][0].pt1.x;
            Y1_2 = closestLines[2][0].pt1.y;
            X2_2 = closestLines[2][0].pt2.x;
            Y2_2 = closestLines[2][0].pt2.y;

        }else{
            X1_2 = closestLines[1][0].pt1.x;
            Y1_2 = closestLines[1][0].pt1.y;
            X2_2 = closestLines[1][0].pt2.x;
            Y2_2 = closestLines[1][0].pt2.y;

            X1_3 = closestLines[2][0].pt1.x;
            Y1_3 = closestLines[2][0].pt1.y;
            X2_3 = closestLines[2][0].pt2.x;
            Y2_3 = closestLines[2][0].pt2.y;
        }
        }



        std::vector<cv::KeyPoint> selectedRangeDots;
        std::vector<cv::KeyPoint> baseLinesDot; // 변경
        std::vector<cv::KeyPoint> MDots;
        std::vector<cv::KeyPoint> S1Dots;
        std::vector<cv::KeyPoint> S2Dots;
        std::vector<cv::KeyPoint> stop_point_dots;


        int verticalThreshold = 1;
        bool isVertical = std::abs(X2_1 - X1_1) < verticalThreshold;
        float CP[2] = {};
        float baseDotM;
        float m;



        if (X2_1 - X1_1 == 0 || abs(X2_1 - X1_1) < 0.0001) { m = 0.0001; baseDotM = 10000; }
        else if (Y2_1 - Y1_1 == 0 || abs(Y2_1 - Y1_1) < 0.0001) { m = 10000; baseDotM = 0.0001; }
        else { m = -1.0 / ((Y2_1 - Y1_1) / (X2_1 - X1_1));baseDotM = (Y2_1 - Y1_1) / (X2_1 - X1_1); }

        int RANGEOFVERTICALLINE = 30;
        for (const auto& key : keypoints) {
            float a = EQua(m, CP_x, CP_y, key.pt.x);
            float CLE = EQua(baseDotM, X1_1, Y1_1, key.pt.x);
            float closeLine = EQua(baseDotM, X1_2, Y1_2, key.pt.x);
            float farLine = EQua(baseDotM, X1_3, Y1_3, key.pt.x);

            if (key.pt.x - 3 <= CP_x && CP_x <= key.pt.x + 3 && key.pt.y - 3 <= CP_y && CP_y <= key.pt.y + 3) { CP[0] = key.pt.x;CP[1] = key.pt.y; }
            if (m > 0) {
                float rear = EQua(m, CP_x, CP_y, key.pt.x - RANGEOFVERTICALLINE);
                float fore = EQua(m, CP_x, CP_y, key.pt.x + RANGEOFVERTICALLINE);

                if (key.pt.x - RANGEOFVERTICALLINE <= CP_x && CP_x <= key.pt.x + RANGEOFVERTICALLINE) {
                    if (Y1_2 <= key.pt.y && key.pt.y <= Y2_3) {
                        selectedRangeDots.push_back(key);
                    }
                }

                if (isVertical) {
                    // 수직 선의 경우, x좌표를 기준으로 필터링
                    if (X1_2 - 5 <= key.pt.x && key.pt.x <= X1_3 + 5) {
                        if (closeLine <= key.pt.y && key.pt.y <= farLine) {
                            selectedRangeDots.push_back(key);
                        }
                    }
                }
                else {
                    if (CLE - 5 <= key.pt.y && key.pt.y <= CLE + 5) {
                        MDots.push_back(key);
                    }

                    if (closeLine - 5 <= key.pt.y && key.pt.y <= closeLine + 5) {
                        //selectedRangeDots.push_back(key);
                        S1Dots.push_back(key);
                    }

                    if (farLine - 5 <= key.pt.y && key.pt.y <= farLine + 5) {
                        //selectedRangeDots.push_back(key);
                        S2Dots.push_back(key);
                    }
                }
            }

            //M<0 case -> 나중에
            else if (m < 0) {
                auto rear = EQua(m, CP_x, CP_y, key.pt.x - RANGEOFVERTICALLINE);
                auto fore = EQua(m, CP_x, CP_y, key.pt.x + RANGEOFVERTICALLINE);
                auto foreB = EQua(baseDotM, CP_x, CP_y, key.pt.x + 10);
                cout << "except" << endl;
                if (fore <= a && a <= rear && key.pt.x - RANGEOFVERTICALLINE <= CP_x && CP_x <= key.pt.x + RANGEOFVERTICALLINE) {
                    selectedRangeDots.push_back(key);
                }

                if ((X1_1 - 10) <= key.pt.x && key.pt.x <= X2_1 + 10 && Y1_1 - 10 <= key.pt.y <= Y2_1 + 10) {
                    selectedRangeDots.push_back(key);
                }

            }


        }


        cout << "X = " << CP[0] << endl;
        cout << "Y = " << CP[1] << endl;

        //for (const auto& keys : selectedRangeDots) {
        //    cout << "x = " << keys.pt.x << ", y = " << keys.pt.y << endl;
        //}

        Mat src = imread("imgs/te.png");
        double rsizeNum = 0.7;
        cv::resize(src, src, {}, rsizeNum, rsizeNum);
        cv::imshow("Keypoints1", img_with_keypoints);
        cv::drawKeypoints(src, selectedRangeDots, src, Scalar(0, 255, 0));
        imshow("Keypoints2", src);
        cv::waitKey(0);

        float CP_X = CP[0];
        float CP_Y = CP[1];
        //int Range = selectedRangeDots.size();
        //cout << Range;
        if(m>0){
            sort(selectedRangeDots.begin(), selectedRangeDots.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.pt.y < b.pt.y;
                }
            );
            int startPoint = 0;
            for (const auto& dot : selectedRangeDots) {
                if (CP_Y == dot.pt.y && CP_X == dot.pt.x) {
                    break;
                }
                startPoint++;
            }
            //대칭점 detect
            for (int i = startPoint+1;i < selectedRangeDots.size();i++) {
                float A = EQua(m, CP_X, CP_Y, selectedRangeDots[i].pt.x);

            }
            for (int i = startPoint-1;i > -1;i--) {

            }

            //대칭점으로부터 확장
            //양쪽으로 2번 (5X5)


            //Keypoint 점 4개(warpPerspective Point)

            //warpPerspecive하고 (10X10 정도) 이후 centerPoint와 Keypoint 뽑아서 벡터로 전달.
            //1. 전체 Keypoints
            //2. CP (x,y)
            //3. 기준선의 (x,y)
            
        }
        else {
            sort(selectedRangeDots.begin(), selectedRangeDots.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.pt.x < b.pt.x;
                }
            );
        }
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = end - start;
    //std::cout << "time = " << elapsed.count() << "second\n";
    }
}