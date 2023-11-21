#include <iostream>
#include <vector>
#include <tuple>
#include <set>

#include <opencv2/opencv.hpp>

void show_image(cv::Mat image, std::string const& title) {
    cv::imshow(title, image);
    int key = 0;

    while (key != 27 && key !='q' && key != ' ') {
        key = cv::waitKey(10) & 0xff;
    }
    cv::destroyWindow(title);
}

cv::Mat imread_rgb(std::string const& filename) {
    cv::Mat bgr = cv::imread(filename);
    cv::Mat result;
    cv::cvtColor(bgr, result, cv::COLOR_RGB2BGR);
    return result;
}

void print_info(std::string const& str, cv::Mat image) {
    std::cout << str << " " << image.size() << " " << image.channels() << std::endl;
}

cv::Mat plane(cv::Mat image, int index) {
    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    return planes[index];
}

struct ExtractArteries {
    ExtractArteries(bool show)
    :
    show_{show}
    {
        
        for (auto morph_size : std::vector<int>{2,5,11} ) {
            auto sz = 2*morph_size + 1;
            structuringElements_.push_back(
                cv::getStructuringElement( 
                    cv::MORPH_RECT, 
                    cv::Size(sz,sz),
                    cv::Point(morph_size,morph_size)
                )
            );
        }

        clahe_ = cv::createCLAHE();
        clahe_->setClipLimit(3);
    }

    bool show() const { return show_; }

    cv::Mat clahe(cv::Mat image, int channel_index = 0) {
        cv::Mat result;
        cv::Mat channel;
        std::cout << "clahe" << std::endl;
        cv::extractChannel(image, channel, channel_index);
        std::cout << "channel.size()=" << channel.size() << std::endl;
        clahe_->apply(channel, result);
        return result;
    }

    cv::Mat color_filter(cv::Mat test_image) {
        cv::Mat lab;
        cv::cvtColor(test_image, lab, cv::COLOR_BGR2Lab);
        // show_image( plane(lab,0), "L");
        // show_image( plane(lab,1), "a");
        // show_image( plane(lab,2), "b");
        return plane(lab,0);
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
            std::cout << "open with se.size()=" << se.size() << std::endl;
            open = erosion(close, se);
            close = dilation(open, se);
        }

        cv::Mat f4;
        cv::subtract(close, test_image, f4);
        auto f5 = clahe(f4);
        //show_image(f5, "large_arteries");
        return f5;
    }


    cv::Mat extract(cv::Mat test_image) {
        auto large_arteries_img = large_arteries( color_filter(test_image) );
        show_image(large_arteries_img, "extract");
        return large_arteries_img;
    }


protected:
    bool show_;
    std::vector< cv::Mat > structuringElements_;
    cv::Ptr<cv::CLAHE> clahe_;

};

int main(int argc, char* argv[]) {
    auto ex = ExtractArteries(true);
    auto input_img = cv::imread("../drive/DRIVE/test/images/01_test.png");
    // std::cout << "input.size()=" << input_img.size() << std::endl;
    // std::cout << input_img.size() << std::endl;
    ex.extract(input_img);

    return 0;
}