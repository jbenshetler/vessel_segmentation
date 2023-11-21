/// vessel_segmentation.cpp by Jeff Benshetler, (c) 2023
/// Purpose: Segment arteries from input image.

#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <filesystem>

#include <opencv2/opencv.hpp>

// Utility functions

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

// float mean(cv::Mat image) {
//     std::vector<cv::Mat> planes;
//     cv::split(image, planes);
//     cv::Mat mean_img3;
//         if (img.channels() == 3) {
//             mean_img3 = (planes[0] + planes[1] + planes[2]) / 3.0;
//         }
// }


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
        cv::extractChannel(image, channel, channel_index);
        clahe_->apply(channel, result);
        return result;
    }

    cv::Mat color_filter(cv::Mat test_image) {
        cv::Mat lab;
        cv::cvtColor(test_image, lab, cv::COLOR_BGR2Lab);
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
            open = erosion(close, se);
            close = dilation(open, se);
        }

        cv::Mat background_removed;
        cv::subtract(close, test_image, background_removed);
        return clahe(background_removed);
    }

    cv::Mat remove_blobs(cv::Mat image) {
        cv::Mat result;
        cv::medianBlur(image, result, 3);
        return result;
    }

    cv::Mat threshold(cv::Mat image) {
        cv::Mat result;
        auto mean = cv::mean(image);
        cv::Mat threshold_img;
        cv::threshold(image, threshold_img, mean[0], 255, cv::THRESH_BINARY);
        auto blobs_removed = remove_blobs(threshold_img);
        if (show()) {
            show_image(blobs_removed, "threshold(): blobs_removed");
        }
        return blobs_removed;
    }


    cv::Mat extract(cv::Mat test_image) {
        auto large_arteries_img = large_arteries( color_filter(test_image) );
        if (show()) {
            show_image(large_arteries_img, "extract(): large_arteries_img");
        }
        auto cleaned_img = threshold(large_arteries_img);
        if (show()) {
            show_image(cleaned_img, "extract(): cleaned_img");
        }
        return large_arteries_img;
    }


protected:
    bool show_;
    std::vector< cv::Mat > structuringElements_;
    cv::Ptr<cv::CLAHE> clahe_;

};

void help(std::string const& program_name, std::string error_msg = "") {
    std::cout << program_name << " [-h] [<input_img> <output_img>]*" << std::endl;
    std::cout << "\t-h : print help\n";
    std::cout << "\t-s : show images\n";
    std::cout << "\t<input_img> input image that is read and processed.\n";
    std::cout << "\t<output_img> path where output image is written\n";
    if (error_msg.size()) {
        std::cerr << error_msg << std::endl;
    }
    exit(-1);
}


void process_image(
    std::string const& program_name, 
    ExtractArteries& ex, 
    std::string const& input_path, 
    std::string const& output_path,
    bool show = true
    ) 
{
    if (!std::filesystem::exists(input_path)) {
        std::ostringstream oss;
        oss << input_path << " input does not exist";
        help(program_name, oss.str());
    }
    auto input_img = cv::imread(input_path);
    auto output_img = ex.extract(input_img);
    if (show) show_image(output_img, "output_path");

}


int main(int argc, char* argv[]) {
    std::string program_name(argv[0]);
    if (argc==1) {
        exit(0);
    }
    if (argc>1 && strcmp(argv[1],"-h")==0) {
        help(program_name, "Help requested");
    }

    if ( (argc>2) &&  (argc % 2 == 0) ) {
        std::ostringstream oss;
        oss << "Wrong number of arguments, argc=" << argc; 
        help(program_name, oss.str() );
    }

    auto ex = ExtractArteries(true);

    for (int i=1; i<argc; i+=2) {
        process_image(program_name, ex, argv[i], argv[i+1]);
    }


   
    // auto input_img = cv::imread("../drive/DRIVE/test/images/01_test.png");
    // ex.extract(input_img);

    return 0;
}