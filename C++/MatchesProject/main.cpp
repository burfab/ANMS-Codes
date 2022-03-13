#include <anms.h>
#include <numeric>

#define USE_KNN
//#undef USE_KNN

class Benchmark{
  public:
    ~Benchmark()
    {
      double time_passed = clock() - t_;
      time_passed = time_passed * 1000 / (double)CLOCKS_PER_SEC;
      std::cout << "Finished " << label_ << " in " << time_passed << " milliseconds" << std::endl;
    }
    Benchmark(std::string label)
    {
      label_ = label;
      t_ = clock();
    }

    std::string label_;
    clock_t t_;
};

cv::Ptr<cv::Feature2D> detector = nullptr;
cv::Ptr<cv::DescriptorMatcher> matcher = nullptr;

struct OrbConfig {
  int fast_thresh = 50;
  int first_level = 0;
  int nlevels = 5;
  int nfeatures = 30000;
  float scale_factor = 1.2f;
  int patch_size = 31;
  int edge_threshold = 31;
  int wta_k = 2;
  cv::ORB::ScoreType score_type = cv::ORB::ScoreType::HARRIS_SCORE;
};

struct Img {
  Img(cv::Mat im, std::string name) : im(im), name(name) {

  }
  cv::Mat im;
  std::string name;
  std::vector<cv::KeyPoint> kps = {};
  std::vector<cv::KeyPoint> kps_anms = {};
};

void initDetector(OrbConfig &config){
  detector = cv::ORB::create(config.nfeatures, config.scale_factor, config.nlevels, 
  config.edge_threshold, config.first_level, config.wta_k, 
  config.score_type, config.patch_size, config.fast_thresh);

#ifndef USE_KNN
  matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
#else
  //matcher = cv::FlannBasedMatcher::create();
  matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
#endif
}

std::vector<cv::DMatch> match(Img &img1, Img &img2){
  Benchmark("Matching");
  std::vector<cv::DMatch> ret, temp;
  
  cv::Mat d1, d2;
  detector->compute(img1.im, img1.kps_anms, d1);
  detector->compute(img2.im, img2.kps_anms, d2);

  #ifndef USE_KNN
  std::vector<cv::DMatch> matcher_matches;
  matcher->match(d1, d2, matcher_matches);

  std::vector<cv::Point2f> p1, p2;
  for (auto m : matcher_matches)
  {
    temp.push_back(m);
    auto pt1 = img1.kps_anms[m.queryIdx];
    auto pt2 = img2.kps_anms[m.trainIdx];
    p1.push_back(pt1.pt);
    p2.push_back(pt2.pt);
  }
  #else
  std::vector<std::vector<cv::DMatch>> matcher_matches;
  matcher->knnMatch(d1, d2, matcher_matches, 2);

  std::vector<cv::Point2f> p1, p2;
  for (auto m : matcher_matches)
  {
    if (m.size() == 0) continue;
    if (m.size() > 1 && m[0].distance >= m[1].distance * 0.8)
      continue;
    temp.push_back(m[0]);
    auto pt1 = img1.kps_anms[m[0].queryIdx];
    auto pt2 = img2.kps_anms[m[0].trainIdx];
    p1.push_back(pt1.pt);
    p2.push_back(pt2.pt);
  }
#endif
  if(temp.size() < 4) return ret;
  cv::Mat inliers;
  cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 10, inliers,10000, 0.994);
  if(H.empty()) {
    std::cout << "Find homography failed" << std::endl;
    return ret;
  }
  for(int i = 0; i < inliers.rows; i++){
    if(inliers.at<char>(i) != 0) ret.push_back(temp[i]);
  }
  return ret;
}

void detectKeypoints(Img &img){
  Benchmark bm("Detect keypoints");
  detector->detect(img.im, img.kps);

  // Sorting keypoints by deacreasing order of strength
  std::vector<float> responseVector;
  for (unsigned int i = 0; i < img.kps.size(); i++)
    responseVector.push_back(img.kps[i].response);
  std::vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  cv::sortIdx(responseVector, Indx, cv::SORT_DESCENDING);

  std::vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < img.kps.size(); i++)
    keyPointsSorted.push_back(img.kps[Indx[i]]);

  img.kps.swap(keyPointsSorted);
}

std::vector<Img> load_images(const std::vector<std::string> &paths){
  std::vector<Img> images;
  for(auto p : paths){
    cv::Mat im = cv::imread(p);
    std::string name = p.substr(p.find_last_of('/'));
    images.emplace_back(im, name);
  }

  return images;
}


std::vector<std::string> parse_paths(int argc, char *argv[]){
  assert(argc > 1);

  std::vector<std::string> paths;
  std::string base(argv[1]);

  int i = 2;
  while(argc > i){
    paths.push_back(base + "/" + std::string(argv[i]));
    i++;
  }
  return paths;
}



enum class AlgType {
  SDC, BROWNANMS, KDTANMS, RTANMS, SSC
};

std::string alg_name(AlgType type){
  if(type == AlgType::SDC) return "sdc";
  if(type == AlgType::BROWNANMS) return "brown anms";
  if(type == AlgType::KDTANMS) return "kdt anms";
  if(type == AlgType::RTANMS) return "rt anms";
  if(type == AlgType::SSC) return "ssc";

  throw std::runtime_error("Invalid type");
}

void run_alg_fixed(AlgType type, Img &im, int numRetPoints, float tolerance) {
  Benchmark("Algorithm " + alg_name(type));

  std::vector<cv::KeyPoint> temp;
  switch (type)
  {
    case AlgType::BROWNANMS:
    temp = brownANMS(im.kps, numRetPoints); break;
    case AlgType::SDC:
    temp = sdc(im.kps, numRetPoints, tolerance, im.im.cols, im.im.rows); break;
    case AlgType::KDTANMS:
    temp = kdTree(im.kps, numRetPoints, tolerance, im.im.cols, im.im.rows); break;
    case AlgType::RTANMS:
    temp = rangeTree(im.kps, numRetPoints, tolerance, im.im.cols, im.im.rows); break;
    case AlgType::SSC:
    temp = ssc(im.kps, numRetPoints, tolerance, im.im.cols, im.im.rows); break;
  }
  im.kps_anms.swap(temp);
}

void run_alg_percentage(AlgType type, Img &im, float percentage, float tolerance) {
  run_alg_fixed(type, im, (int)(percentage * im.kps.size()), tolerance);
}

int main(int argc, char *argv[]) {
  assert(argc > 2);
  std::cout << "Args: " << std::endl;
  for(int i = 1; i<argc; i++){
    std::cout << "\t" << argv[i] << std::endl;
  }
  OrbConfig orb_cfg;
  AlgType ALG_TYPE = AlgType::KDTANMS;
  float percentage = 0.2;
  float tolerance = 0.3; // tolerance of the number of return points
  int numRetPoints = 2000;
  //numRetPoints = 0; //to use percentage



  initDetector(orb_cfg);
  auto paths = parse_paths(argc, argv);
  auto images = load_images(paths);
  for(auto &im : images){
    detectKeypoints(im);
  }


  for(auto &im : images){
    if (numRetPoints == 0) run_alg_percentage(ALG_TYPE, im, percentage, tolerance);
    else run_alg_fixed(ALG_TYPE, im, numRetPoints, tolerance);
  }
  
  for (auto &im : images)
  {
    cv::Mat preview;
    cv::Mat preview_unfiltered;
    cv::Mat preview_anms;
    cv::drawKeypoints(im.im, im.kps, preview_unfiltered,
                      cv::Scalar(94.0, 206.0, 165.0, 0.0));

    cv::drawKeypoints(im.im, im.kps_anms, preview_anms,
                      cv::Scalar(206.0, 94.0, 0.0, 0.0));

    cv::hconcat(preview_unfiltered, preview_anms, preview);
    cv::imshow(im.name, preview);
  }

  cv::waitKey(0); // Wait for a keystroke in the window

  for (int i = 0; i < images.size() - 1; i+=2)
  {
    cv::Mat preview;
    auto matches = match(images[i], images[i + 1]);
    cv::drawMatches(images[i].im, images[i].kps_anms, images[i+1].im, images[i+1].kps_anms, matches, preview);
    std::string match_name = images[i].name + " <-> " + images[i+1].name;
    std::cout << match_name << " - good matches: " << matches.size() << std::endl;
    cv::imshow(match_name, preview);
  }
  cv::waitKey(0);

  return 0;
}
