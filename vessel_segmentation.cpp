/// vessel_segmentation.cpp by Jeff Benshetler, (c) 2023
/// Purpose: Segment arteries from input image.

#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <filesystem>
#include <tuple>
#include <set>
#include <algorithm>
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
        auto luminance = plane(lab,0);
        cv::Mat equalized;
        clahe_->apply(luminance, equalized);
        cv::Mat result;
        cv::merge(std::vector<cv::Mat>{equalized, equalized, equalized}, result);
        return result;
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

    cv::Mat threshold(cv::Mat image) {
        cv::Mat result;
        auto mean = cv::mean(image);
        cv::Mat threshold_img;
        cv::threshold(image, threshold_img, mean[0], 255, cv::THRESH_BINARY);
        return threshold_img;
    }

    cv::Mat remove_blobs(cv::Mat binary_image) {
        cv::Mat result;
        binary_image.copyTo(result);

        std::vector< cv::Mat > contours;
        cv::findContours( binary_image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        double const min_valid_area = 25.0;
        for (auto const& cnt : contours) {
            auto area = cv::contourArea(cnt);
            if (area < min_valid_area) {
                cv::drawContours(result, cnt, -1, cv::Scalar(0), -1);
            }
        }
        return result;
    }


    cv::Mat extract(cv::Mat test_image) {
        auto large_arteries_img = large_arteries( color_filter(test_image) );
        if (show()) show_image(large_arteries_img, "extract(): large_arteries_img");
        cv::Mat median_img;
        cv::medianBlur(large_arteries_img, median_img, 3);

        auto threshold_img = threshold(median_img);
        if (show()) show_image(threshold_img, "extract(): threshold");
        auto cleaned_img = remove_blobs( threshold_img );
        if (show()) show_image(cleaned_img, "extract(): cleaned");
        cv::medianBlur(cleaned_img, median_img, 3);
        return median_img;
    }


protected:
    bool show_;
    std::vector< cv::Mat > structuringElements_;
    cv::Ptr<cv::CLAHE> clahe_;

};



enum class Flag { show, help};

using Options = std::set<Flag>;



void help(std::string const& program_name, std::string error_msg = "") {
    std::cout << program_name << " [-h] [-s] [<input_img> <output_img>]*" << std::endl;
    std::cout << "\t-h : print help\n";
    std::cout << "\t-s : show images\n";
    std::cout << "\t<input_img> input image that is read and processed.\n";
    std::cout << "\t<output_img> path where output image is written\n";
    if (error_msg.size()) {
        std::cerr << error_msg << std::endl;
    }
}


bool process_image(
    std::string const& program_name, 
    ExtractArteries& ex, 
    std::string const& input_path, 
    std::string const& output_path
    ) 
{
    bool success = true;
    if (!std::filesystem::exists(input_path)) {
        std::ostringstream oss;
        oss << input_path << " input does not exist";
        help(program_name, oss.str());
        success = false;
    } else {
        auto input_img = cv::imread(input_path);
        cv::Mat bgr_img;
        cv::cvtColor(input_img, bgr_img, cv::COLOR_RGB2BGR);
        auto output_img = ex.extract(bgr_img);
        if (ex.show()) show_image(output_img, "output_path");
        cv::imwrite(output_path, output_img);
        if (!std::filesystem::exists(output_path)) {
            std::ostringstream oss;
            std::cerr << "Error: Failed to write " << output_path;
            success = false;
        }
    }
    return success;
}



std::tuple< Options, std::vector<std::string>, int, std::string> parse_args(int const argc, char* argv[]) {
    int result = 0;
    Options options;

    std::string program_name(argv[0]);
    std::vector<std::string> image_files;

    std::vector< std::pair< std::string, std::string > > input_output;
    for (int i=1; i<argc; i++) {
        std::string arg( argv[i] );
        if ( arg == "-h") {
            options.insert(Flag::help);
        } else if ( arg == "-s" ) {
            options.insert(Flag::show);
        } else {
            image_files.push_back( arg );
        }
    }

    std::copy( image_files.begin(), image_files.end(), std::ostream_iterator<std::string>(std::cout, ", ") );
    std::cout << "\n";

    if (image_files.size() % 2 == 1) {
        std::ostringstream oss;
        oss << "Wrong number of arguments, argc=" << argc; 
        help(program_name, oss.str() );
        result = -1;
    } 

    return std::make_tuple(options, image_files, result, program_name);
}


int main(int argc, char* argv[]) {
    auto [options, image_files, result, program_name] = parse_args(argc, argv);
    
    if ( options.contains(Flag::help) ) {
        help(program_name);
    }

    if (result==0) {
        auto ex = ExtractArteries( options.contains(Flag::show) );

        for (int i=0; i<image_files.size(); i+=2) {
            result = process_image(program_name, ex, image_files.at(i), image_files.at(i+1) );
            if (result != 0) {
                break;
            }
        }
    }

    return result;
}
