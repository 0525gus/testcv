#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
// 트랙바
int rho_slider = 2;
int theta_slider = 180;
int threshold_slider = 104;
int start_x = -1;
int start_y = -1;

Mat img;
Mat edges;

vector<Line> linePoints;
vector<vector<Line>> clusters;

int main() {

    string PATH = "imgs/dot_base.png";
    img = imread(PATH);
    double rsizeNum = 1;
    resize(img, img, {}, rsizeNum, rsizeNum);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    Canny(gray, edges, 0, 0);

    namedWindow("Hough Lines");
    imshow("Hough Lines", img);


    // 트랙바 콜백함수 생성
    createTrackbar("Rho * 10", "Hough Lines", &rho_slider, 100, houghLinesCallback);
    createTrackbar("Theta", "Hough Lines", &theta_slider, 360, houghLinesCallback);
    createTrackbar("Threshold", "Hough Lines", &threshold_slider, 500, houghLinesCallback);
    //시간측정


    waitKey(0);

    return 0;
}

void houghLinesCallback(int, void*) {
    double rho = static_cast<double>(rho_slider) / 10.0;
    vector<Vec2f> lines;
    HoughLines(edges, lines, rho, CV_PI / (theta_slider), threshold_slider);

    Mat result = img.clone();

    linePoints.clear();
    clusters.clear();

    // 검출된 라인 linePoints(x,y 두점의 셋) 저장
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        linePoints.push_back({ pt1, pt2 });
    }

    // linepoints 기준으로 cluster
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
        // Find the line that overlaps with the most other lines in the cluster
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

        // Keep only the representative line in this cluster
        cluster = { representative };

        // Draw the representative line
        line(result, representative.pt1, representative.pt2, Scalar(0, 255, 0), 2, LINE_AA);
        //cout << "Number of lines in this cluster: " << cluster.size() << endl;
    }

    cv::setMouseCallback("Hough Lines", on_mouse);
    imshow("Hough Lines", result);
}

double distanceFromPointToLine(Point pt, Point lineStart, Point lineEnd) {
    double numer = abs((lineEnd.y - lineStart.y) * pt.x - (lineEnd.x - lineStart.x) * pt.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x);
    double denom = sqrt(pow(lineEnd.y - lineStart.y, 2) + pow(lineEnd.x - lineStart.x, 2));
    return numer / denom;
}
float eq(float m ,int x, int y) {
    return m * x + y;
}

void on_mouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        start_x = x;
        start_y = y;
        cout << x << "," << y << endl;
        
        //시간측정


        //x,y좌표를 기준으로 ROI
        try {

            Mat img_ROI(edges, Rect(start_x - 170, start_y - 170, 340, 340 ));


            cv::SimpleBlobDetector::Params params;
            params.filterByArea = true;
            params.minArea = 30; // 픽셀 수에 따라 조절

            cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

            std::chrono::steady_clock::time_point start, end;
            start = std::chrono::steady_clock::now();

            //가장 많이 프로세스 타임을 먹는 부분 .
            std::vector<cv::KeyPoint> keypoints;
            detector->detect(img_ROI, keypoints);
            
            end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            std::cout << std::format("{:.6f}s\n", duration);

            cv::Mat img_with_keypoints;
            cv::drawKeypoints(img_ROI, keypoints, img_with_keypoints,Scalar(0,255,0));
        //    cout << "keypoints size = " << keypoints.size() << endl;

            for (size_t i = 0; i < keypoints.size(); i++) {

                keypoints[i].pt.x += start_x - 170;
                keypoints[i].pt.y += start_y - 170;
                //float x = keypoints[i].pt.x;
                //float y = keypoints[i].pt.y;

               /* cout << "x = " << x << ", y = " << y << endl;*/
            }


             //SBD check part
       //     cv::imshow("Keypoints", img_with_keypoints);
       //     cv::waitKey(0);
            

        }

        catch (...) {
            cerr << "ROI range error but pass" << endl;
        }


        //waitKey(0);
        //



        vector<pair<double, vector<Line>>> distances;
        for (const auto& cluster : clusters) {
            for (const auto& line : cluster) {
                double dist = distanceFromPointToLine(Point(x, y), line.pt1, line.pt2);
                distances.push_back(make_pair(dist, cluster));
            }
        }
        //가장 가까운라인 sort
        sort(distances.begin(), distances.end(), [](const pair<double, vector<Line>>& a, const pair<double, vector<Line>>& b) {
            return a.first < b.first;
            });


        vector<vector<Line>> closestLines;
        closestLines.push_back(distances[0].second);
        closestLines.push_back(distances[1].second);


        
        //for (const auto& cluster : closestLines) {
        //    for (const auto& line : cluster) {
        //       cout << "Line from (" << line.pt1.x << "," << line.pt1.y << ") to (" << line.pt2.x << "," << line.pt2.y << ")" << endl;
        //    }
        //}
        //equation
        //cout << "X1=" << closestLines[0][0].pt1.x << " Y1=" << closestLines[0][0].pt1.y << " X2=" << closestLines[0][0].pt2.x << " Y2=" << closestLines[0][0].pt2.y << endl;
        //cout << "X1=" << closestLines[1][0].pt1.x << " Y1=" << closestLines[1][0].pt1.y << " X2=" << closestLines[1][0].pt2.x << " Y2=" << closestLines[1][0].pt2.y << endl;


        //가장 가까운 라인의 기울기를 구하고 
        // 그 라인의 기울기와 수직인 기울기를 다시 구하고 .
        // CP(Center Point) 기준으로 아래로탐색(일단
        float X1 = closestLines[0][0].pt1.x;
        float Y1 = closestLines[0][0].pt1.y;
        float X2 = closestLines[0][0].pt2.x;
        float Y2 = closestLines[0][0].pt2.y;

        float m;
        if (X2 - X1 != 0) {
            m = -1.0 / ((Y2 - Y1) / (X2 - X1));
        }
        else {
            m = std::numeric_limits<float>::infinity();

        }
        //float eq_num = (m, x, y);
        //cout << eq_num << endl;















        



    }
}