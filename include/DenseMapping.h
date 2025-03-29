#ifndef DENSEMAPPING_H
#define DENSEMAPPING_H

#include "Map.h"
#include "Atlas.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include "KeyFrame.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <condition_variable>
#include <pcl/filters/voxel_grid.h>
#include <mutex>
#include <pcl/visualization/pcl_visualizer.h>
#include "StereoMatch.h"

namespace ORB_SLAM3 {

class DenseMapping {
public:
    enum MappingSensor {
        RGBD = 0,
        STEREO
    };

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    DenseMapping(double resolution_, double meank_, double stdthresh_, double unit_);
    DenseMapping(double resolution_, double meank_, double stdthresh_, double unit_,
                 double mindisp_, double maxdisp_, Stereo_Algorithm::AlgorithmType type = Stereo_Algorithm::AlgorithmType::ELAS);
    
    // // 插入关键帧
    // void InsertKeyFrame(KeyFrame* kf, const cv::Mat& imLeft, const cv::Mat& imRight);
    
    // 获取完整点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetGlobalCloud();
    
    // 保存点云
    void SavePointCloud(const std::string& filename);
    
    // 新增方法
    //void UpdateCloud(Map* pMap);
    void shutdown();
    bool mabIsUpdating = false;
    
    // 新增显示线程
    void Viewer();

    // // Coordinate system visualization
    // void AddCoordinateSystem(std::shared_ptr<pcl::visualization::PCLVisualizer> viewer, 
    //                         const Eigen::Matrix4f &pose, const std::string &prefix);

    // Point cloud operations
    PointCloud::Ptr GeneratePointCloudFromRGBD(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    PointCloud::Ptr GeneratePointCloudFromStereo(KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& Q);
    PointCloud::Ptr GeneratePointCloudFromStereoWithDisp(KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& disp, cv::Mat& Q);

    // Keyframe insertion methods
    void InsertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void InsertKeyFrame(KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& Q);
    void InsertKeyFrame(KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& disp, cv::Mat& Q);

protected:
    // 点云滤波和优化
    void FilterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    // 从双目图像生成点云
    // PointCloud::Ptr GeneratePointCloudFromStereo(KeyFrame* kf, cv::Mat& left_img, cv::Mat& right_img);
    
    shared_ptr<thread>  viewerThread;
    
    double resolution = 0.04;  // 体素滤波网格大小
    pcl::VoxelGrid<PointT>  voxel;
    PointCloud::Ptr globalMap;

    std::mutex mMutexCloud;
    std::mutex mKeyFrameMtx;

    std::vector<KeyFrame *> keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;

    condition_variable mKeyFrameUpdated;
    mutex               keyFrameUpdateMutex;

    bool    shutDownFlag    =false;
    mutex   shutDownMutex;

    uint16_t                lastKeyframeSize =0;

    // // 添加临时存储结构
    // struct KeyFrameWithImages {
    //     KeyFrame* kf;
    //     cv::Mat imLeft;
    //     cv::Mat imRight;
    // };
    // std::vector<KeyFrameWithImages> keyframesWithImages;

    // Sensor type
    MappingSensor mSensor;

    // Stereo matching
    std::shared_ptr<Stereo_Algorithm> stereo;
    int numDisp = 0;

    // Point cloud processing
    pcl::StatisticalOutlierRemoval<PointT> sor;
    double meank = 10;
    double stdthresh = 1;
    double unit = 1000;  // default mm

    // Additional image storage
    std::vector<cv::Mat> rightImgs;
    std::vector<cv::Mat> dispImgs;
    cv::Mat Q;

    // 计算Q矩阵的辅助函数
    cv::Mat ComputeQMatrix(double fx, double fy, double cx, double cy, double baseline);

};

}  // namespace ORB_SLAM3

#endif 