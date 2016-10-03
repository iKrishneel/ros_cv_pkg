
#include <evaluation_ground_truth_annotator/evaluate_point_cloud_accuracy.h>

EvaluateAccuracy::EvaluateAccuracy(
    const std::string gt_text, const std::string t_text,
    const std::string result_path) {
    this->DISTANCE_THRESH_ = 0.005f;
    this->write_path_ = result_path;
    std::vector<std::string> ground_truth;
    this->readDataFromFile(ground_truth, gt_text);
    std::vector<std::string> test_data;
    this->readDataFromFile(test_data, t_text);
    this->evaluate(ground_truth, test_data);
}

void EvaluateAccuracy::readDataFromFile(
    std::vector<std::string> &path_list,
    const std::string path_to_file) {
    std::ifstream infile(path_to_file.c_str());
    std::string line;
    path_list.clear();
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string path_to_pcd;
      iss >> path_to_pcd;
      if (!path_to_pcd.empty()) {
         path_list.push_back(path_to_pcd);
      }
    }
}

void EvaluateAccuracy::evaluate(
    std::vector<std::string> ground_truth,
    std::vector<std::string> test_data) {
    if (ground_truth.empty() || test_data.empty()) {
       std::cout << "EMPTY CANNOT BE EVLAUTED!"  << "\n";
       return;
    }
    const int skip = 1;
    PointCloud::Ptr grnd_truth_points(new PointCloud);
    PointCloud::Ptr test_points(new PointCloud);
    std::vector<float> recall;
    std::vector<float> precision;
    for (int j = 0; j < test_data.size(); j += skip) {
       
       int i = std::floor(j/skip);
       grnd_truth_points->clear();
       int r = pcl::io::loadPCDFile<PointT>(
          ground_truth[i], *grnd_truth_points);

       std::cout << "computing..." << grnd_truth_points->size()  << "\n";
       std::cout << i << " " << j  << " " << r<< "\n";
       if (r != -1) {

          std::cout << "kdtree..."  << "\n";
          
          pcl::KdTreeFLANN<PointT>::Ptr kdtree(
             new pcl::KdTreeFLANN<PointT>);
          kdtree->setInputCloud(grnd_truth_points);

          std::cout << "running now"  << "\n";
          
          int true_positive = 0;
          int false_positive = 0;
          for (int k = 0; k < skip; k++) {
             int l = k + i;

             if (l > test_data.size()) {
                return;
             }
             
             test_points->clear();
             r = pcl::io::loadPCDFile<PointT>(
                test_data[l], *test_points);
             this->accuracy(true_positive, false_positive,
                            test_points, kdtree);

             std::cout << "computing..." << test_points->size()  << "\n";
             
             float re = static_cast<float>(true_positive)/
                static_cast<float>(grnd_truth_points->size());
             float pr = static_cast<float>(true_positive)/
                static_cast<float>(test_points->size());
             recall.push_back(re);
             precision.push_back(pr);

             std::cout << "RE/PR: " << re << " " << pr  << "\n";
          }
       }
    }

    std::ofstream outfile(write_path_.c_str());
    for (int i = 0; i < recall.size(); i++) {
       outfile << recall[i] << " " << precision[i] << "\n";
    }
    outfile.close();
}

void EvaluateAccuracy::accuracy(
    int &true_positive, int &false_positive,
    const PointCloud::Ptr test_points,
    const pcl::KdTreeFLANN<PointT>::Ptr kdtree) {
    if (test_points->empty()) {
       return;
    }
    true_positive = 0;
    false_positive = 0;
    std::vector<int> neigbor_indices;
    std::vector<float> distances;
    for (int i = 0; i < test_points->size(); i++) {
       PointT pt = test_points->points[i];
       this->getPointNeigbour<int>(neigbor_indices, distances,
                                   kdtree, pt, 1, true);
       if (!distances.empty()) {
          if (distances[0] < DISTANCE_THRESH_) {
             true_positive++;
          } else {
             false_positive++;
          }
       } else {
          false_positive++;
       }
    }
}

template<class T>
void EvaluateAccuracy::getPointNeigbour(
    std::vector<int> &neigbor_indices,
    std::vector<float> &point_squared_distance,
    const pcl::KdTreeFLANN<PointT>::Ptr kdtree,
    const PointT seed_point, const T K, bool is_knn) {
    neigbor_indices.clear();
    point_squared_distance.clear();
    if (is_knn) {
       int search_out = kdtree->nearestKSearch(
         seed_point, K, neigbor_indices, point_squared_distance);
    } else {
      int search_out = kdtree->radiusSearch(
         seed_point, K, neigbor_indices, point_squared_distance);
    }
}



int main(int argc, char *argv[]) {
    if (argc < 4) {
       std::cout << "\033[31m  USAGE:"
                 << "<path_to_ground_truth>, <path_to_test_data>"
                 << "<path_to_write_the_results>"
                 << "\033[0m \n";
       std::cout << "The path contains the text file with full ";
       std::cout << "path to the .pcds"  << "\n";
       return -1;
    }
    EvaluateAccuracy ea(argv[1], argv[2], argv[3]);
    return 0;
}
