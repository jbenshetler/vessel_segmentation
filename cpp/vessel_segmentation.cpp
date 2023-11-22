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

/////////////////////////
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


/////////////////////////
// Does the segmentation
struct ExtractArteries {
    /// @brief Construct structuring elements and adaptive contrast enhancement data structures
    /// @param show Whether to incrementally show image as it is processed
    /// @note Based on [Contour Based Blood Vessel Segmentation in Retinal Fundus Images](https://github.com/sachinmb27/Contour-Based-Blood-Vessel-Segmentation-in-Retinal-Fundus-Images/blob/main/segmentation.py)
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

    /// @brief Perform adaptive contrast enhancement
    /// @param image Source for contrast enhancement
    /// @param channel_index Optional channel index for multi-channel images
    /// @return Contrast enhanced image
    cv::Mat clahe(cv::Mat image, int channel_index = 0) {
        cv::Mat result;
        cv::Mat channel;
        cv::extractChannel(image, channel, channel_index);
        clahe_->apply(channel, result);
        return result;
    }

    /// @brief Perform contrast enhancement on luminance
    /// @param test_image Source for filtering
    /// @return Contrast-enhanced luminance image
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

    /// @brief Morphological opening
    /// @param image Source for morpho operation
    /// @param se Structuring element
    /// @param iterations How many times to perform operation
    /// @return Opening result
    cv::Mat erosion(cv::Mat image, cv::Mat se, int iterations = 1) {
        cv::Mat result;
        cv::morphologyEx(image, result, cv::MORPH_OPEN, se, cv::Point(-1,-1), iterations);
        return result;
    }

    /// @brief Morphological closing
    /// @param image Source for morpho operation
    /// @param se Structing element
    /// @param iterations How many time to perform operation
    /// @return Closing result
    cv::Mat dilation(cv::Mat image, cv::Mat se, int iterations = 1) {
        cv::Mat result;
        cv::morphologyEx(image, result, cv::MORPH_CLOSE, se, cv::Point(-1,-1), iterations);
        return result;
    }

    /// @brief Extract the large arteries 
    /// @param test_image Source for extraction
    /// @return Grayscale image with everything other than larger arteries suppressed
    /// @note "Large arteries" is a relative
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

    /// @brief Set everything below the image mean to black
    /// @param image Source for threshold
    /// @return Binary image with thresholding results
    cv::Mat threshold(cv::Mat image) {
        cv::Mat result;
        auto mean = cv::mean(image);
        cv::Mat threshold_img;
        cv::threshold(image, threshold_img, mean[0], 255, cv::THRESH_BINARY);
        return threshold_img;
    }

    /// @brief Remove blobs from image based on size
    /// @param binary_image Source for suppression
    /// @return Binary image with blobs suppressed
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


    /// @brief Primary interface to extract arteries from image
    /// @param test_image BGR source image
    /// @return Binary image with mask of large arteries
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


/// @brief Output help text
/// @param program_name Included in output
/// @param error_msg Optional message to include in output to STDERR
void help(std::string const& program_name, std::string error_msg = "") {
    std::cout << program_name << " [-h] [-s] [<input_img> <output_img>]*" << std::endl;
    std::cout << "\t-h : print help\n";
    std::cout << "\t-s : show images. Press 'q', SPACE, or ESC to close window.\n";
    std::cout << "\t<input_img> input image that is read and processed.\n";
    std::cout << "\t<output_img> path where output image is written\n";
    if (error_msg.size()) {
        std::cerr << error_msg << std::endl;
    }
}

/// @brief Read image, extract arteries, and store resulting image to file
/// @param program_name Used in error text
/// @param ex Performs artery extraction
/// @param input_path Input image path on disk
/// @param output_put Output path on disk where to store image
/// @return bool `true` if succeeded in processing, `false` otherwise
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
        // create 2-up composite to show result
        cv::Mat output_color;
        cv::cvtColor(output_img, output_color, cv::COLOR_GRAY2RGB);
        cv::Mat twoup;
        cv::hconcat(input_img, output_color, twoup);
        if (ex.show()) show_image(output_img, "output_path");
        cv::imwrite(output_path, twoup);
        if (!std::filesystem::exists(output_path)) {
            std::ostringstream oss;
            std::cerr << "Error: Failed to write " << output_path;
            success = false;
        }
    }
    return success;
}


/// @brief Parse command line arguments
/// @param argc Number of command line arguments, including the program name
/// @param argv Array of strings passed on the command line
/// @returns Options passed as flags, input and output image file names in a `vector`, a shell return code, and the program name
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
