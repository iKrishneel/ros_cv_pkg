
#include <evaluation_ground_truth_annotator/evaluation_ground_truth_annotator.h>

CvAlgorithmEvaluation::CvAlgorithmEvaluation() :
    labels_(0) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->marked_cloud_ = PointCloud::Ptr(new PointCloud);
    this->onInit();
}

void CvAlgorithmEvaluation::onInit() {
    this->subscribe();
    
}

void CvAlgorithmEvaluation::subscribe() {
    this->sub_gt_cloud_ = this->pnh_.subscribe(
       "input_gt_cloud", 1, &CvAlgorithmEvaluation::groundTCB, this);

   
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              this->sub_indices_);
    this->sync_->registerCallback(
       boost::bind(&CvAlgorithmEvaluation::cloudCB, this, _1, _2));
}

void CvAlgorithmEvaluation::unsubscribe() {
   
}


void CvAlgorithmEvaluation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_msgs::ClusterPointIndices::ConstPtr &indices_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
   
}

void CvAlgorithmEvaluation::groundTCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    ROS_WARN("RECEIVED GROUND TRUTH CLIUD: %d", cloud->size());

    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       pt.r = this->labels_;
       pt.b = this->labels_;
       pt.g = this->labels_;
       this->marked_cloud_->push_back(pt);
    }
    
    char ch;
    std::cout << "pressed y to save:"  << "\n";
    std::cin >> ch;
    if (ch == 'y') {
       ROS_ERROR("saving the marked cloud: %d", cloud->size());
       std::string save_path = "/home/krishneel/Desktop/acvr-eccv/ground-truth/";
       std::string name = "scene8.pcd";

       pcl::io::savePCDFileASCII(save_path + name, *marked_cloud_);
       this->marked_cloud_ = PointCloud::Ptr(new PointCloud);
       this->labels_ = 0;
       
    } else {
       this->labels_++;
    }
    std::cout << "label: " << labels_  << "\n";
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "evaluation_ground_truth_annotator");
    CvAlgorithmEvaluation cvae;
    ros::spin();
    return 0;
}
