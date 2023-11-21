#include <iostream>
#include <vector>
#include <tuple>


#include <opencv2/opencv.hpp>

struct ExtractArteries {
    ExtractArteries(bool show)
    :
    show_{show}
    {
        for (auto sz : std::vector<int>{5,11,23} ) {
            structuringElements_.push_back(
                cv::getStructuringElement( 
                    cv::MORPH_RECT, 
                    cv::Size(sz,sz),
                    cv::Point(1,1)
                )
            );
        }

        clahe_ = cv::createCLAHE();
        clahe_->setClipLimit(3);
    }

    bool show() const { return show_; }

    cv::Mat clahe(cv::Mat image) {
        cv::Mat result;
        clahe_->apply(image, result);
        return result;
    }

    cv::Mat color_filter(cv::Mat test_image) {
        cv::Mat lab;
        cv::cvtColor(test_image, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> planes;
        cv::split(lab, planes);
        return planes[1];
    }

    cv::Mat erosion(cv::Mat image, cv::Mat se, int iterations = 1) {
        cv::Mat result;
        cv::morphologyEx(image, result, cv::MORPH_OPEN, se, cv::Point(-1,-1), iterations);
        return result;
    }

    cv::Mat dilation(cv::Mat image, cv::Mat se, int iterations = 1) {
        cv::Mat result;
        cv::morphologyEx(image, result, cv::MORPH_CLOSE, se, cv::Point(-1,-1), iterations);
        return result;
    }

    cv::Mat large_arteries(cv::Mat test_image) {
        cv::Mat close;
        cv::Mat open;
        test_image.copyTo(close);

        for (auto const& se : structuringElements_) {
            open = erosion(close, se);
            close = dilation(open, se);
        }

        cv::Mat f4;
        cv::subtract(close, test_image, f4);
        cv::Mat f5;
        f5 = clahe(f4);
        return f5;
    }

    


protected:
    bool show_;
    std::vector< cv::Mat > structuringElements_;
    cv::Ptr<cv::CLAHE> clahe_;

};

int main(int argc, char* argv[]) {
    auto ex = ExtractArteries(true);
    auto input_img = cv::imread("drive/DRIVE/test/images/01_test.png");
    cv::imwrite("output/output.png", ex.clahe(input_img));
    //cv::imshow("clahe", ex.clahe(input_img) );

    return 0;
}