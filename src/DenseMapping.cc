#include <boost/make_shared.hpp>
#include "DenseMapping.h"
#include <opencv2/core/eigen.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <pcl/point_cloud.h>

namespace ORB_SLAM3 {

DenseMapping::DenseMapping(double resolution_, double meank_, double stdthresh_, double unit_)
    : resolution(resolution_), meank(meank_), stdthresh(stdthresh_), unit(unit_)
{
    std::cout << "Initializing DenseMapping with default parameters" << std::endl;
    // Voxel grid filter
    voxel.setLeafSize(resolution, resolution, resolution);
    
    // Statistical outlier removal
    sor.setMeanK(meank);
    sor.setStddevMulThresh(stdthresh);
    
    // Initialize global map
    
    globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Start viewer thread
    viewerThread = make_shared<thread>(bind(&DenseMapping::Viewer, this));
    
    std::cout << "Voxel filter resolution: " << resolution << std::endl;
    std::cout << "Outlier removal params - meank: " << meank 
              << ", stdthresh: " << stdthresh << std::endl;
    std::cout << "Unit conversion: " << unit << std::endl;
}

DenseMapping::DenseMapping(double resolution_, double meank_, double stdthresh_, double unit_,
                         double mindisp_, double maxdisp_, Stereo_Algorithm::AlgorithmType type)
    : resolution(resolution_), meank(meank_), stdthresh(stdthresh_), unit(unit_)
{
    std::cout << "Initializing DenseMapping with stereo matching" << std::endl;
    // Voxel grid filter
    voxel.setLeafSize(resolution, resolution, resolution);
    
    // Statistical outlier removal
    sor.setMeanK(meank);
    sor.setStddevMulThresh(stdthresh);
    
    // Initialize stereo matching
    stereo = Stereo_Algorithm::create(mindisp_, maxdisp_, type);
    numDisp = static_cast<int>(maxdisp_ - mindisp_);
    
    // Initialize global map
    globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Start viewer thread
    viewerThread = make_shared<thread>(bind(&DenseMapping::Viewer, this));
    
    std::cout << "Voxel filter resolution: " << resolution << std::endl;
    std::cout << "Outlier removal params - meank: " << meank 
              << ", stdthresh: " << stdthresh << std::endl;
    std::cout << "Unit conversion: " << unit << std::endl;
    std::cout << "Stereo matching - min disp: " << mindisp_ 
              << ", max disp: " << maxdisp_ << std::endl;
}

void DenseMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        mKeyFrameUpdated.notify_one();
    }
    viewerThread->join();
    SavePointCloud("DenseMap.pcd");
}

// void DenseMapping::InsertKeyFrame(KeyFrame* kf, const cv::Mat& imLeft, const cv::Mat& imRight) {
//     if(!kf) {
//         std::cout << "Error: Null keyframe" << std::endl;
//         return;
//     }

//     // 检查图像是否为空
//     if(imLeft.empty() || imRight.empty()) {
//         std::cout << "Warning: Empty images for keyframe id = " << kf->mnId << std::endl;
//         return;
//     }

//     {
//         std::unique_lock<std::mutex> lock(mKeyFrameMtx);
//         // 存储关键帧和对应的图像
//         KeyFrameWithImages kfWithImages;
//         kfWithImages.kf = kf;
//         kfWithImages.imLeft = imLeft.clone();  // 使用clone()确保深拷贝
//         kfWithImages.imRight = imRight.clone();
//         keyframesWithImages.push_back(kfWithImages);
        
//         std::cout << "Received keyframe id = " << kf->mnId 
//                   << ", image size: " << imLeft.size()
//                   << ", type: " << imLeft.type() << std::endl;
//     }
//     mKeyFrameUpdated.notify_one();
// }

// stereo with the disparity
void DenseMapping::InsertKeyFrame(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &disp, cv::Mat &Q){
     if(!kf) {
        std::cout << "Error: Null keyframe" << std::endl;
        return;
    }

    // 检查图像是否为空
    if(left.empty() || right.empty() || disp.empty()) {
        std::cout << "Warning: Empty images for keyframe id = " << kf->mnId << std::endl;
        return;
    }

    // 如果 Q 为空，则计算 Q 矩阵
    if (Q.empty()) {
        double fx = kf->fx;
        double fy = kf->fy;
        double cx = kf->cx;
        double cy = kf->cy;
        double baseline = kf->mb;
        
        if (baseline <= 0) {
            baseline = 0.53716;  // 默认基线值
            std::cout << "Warning: Using default baseline = " << baseline << std::endl;
        }
        
        Q = ComputeQMatrix(fx, fy, cx, cy, baseline);
        std::cout << "Computed Q matrix due to empty input Q" << std::endl;
    }

    mSensor = MappingSensor::STEREO;
    cout << "receive a keyframe, id = " << kf->mnId << endl;
    unique_lock<mutex> lck(mKeyFrameMtx);
    std::vector<KeyFrame *> keyframes;
    colorImgs.push_back(left.clone());
    rightImgs.push_back(right.clone());
    this->Q = Q.clone();
    mKeyFrameUpdated.notify_one();
}

// stereo without the disparity
void DenseMapping::InsertKeyFrame(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &Q){
    if(!kf) {
        std::cout << "Error: Null keyframe" << std::endl;
        return;
    }
    // 检查图像是否为空
    if(left.empty() || right.empty()) {
        std::cout << "Warning: Empty images for keyframe id = " << kf->mnId << std::endl;
        return;
    }

    // 如果 Q 为空，则计算 Q 矩阵
    if (Q.empty()) {
        double fx = kf->fx;
        double fy = kf->fy;
        double cx = kf->cx;
        double cy = kf->cy;
        double baseline = kf->mb;
        
        if (baseline <= 0) {
            baseline = 0.53716;  // 默认基线值
            std::cout << "Warning: Using default baseline = " << baseline << std::endl;
        }
        
        Q = ComputeQMatrix(fx, fy, cx, cy, baseline);
        std::cout << "Computed Q matrix due to empty input Q" << std::endl;
    }

    mSensor = MappingSensor::STEREO;
    cout << "receive a keyframe, id = " << kf->mnId << endl;
    unique_lock<mutex> lck(mKeyFrameMtx);
    std::vector<KeyFrame *> keyframes;
    colorImgs.push_back(left.clone());        
    rightImgs.push_back(right.clone());
    this->Q = Q.clone();
    mKeyFrameUpdated.notify_one();
}

// 添加 RGBD 的 InsertKeyFrame 函数
void DenseMapping::InsertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth) {
    if(!kf) {
        std::cout << "Error: Null keyframe" << std::endl;
        return;
    }

    // 检查图像是否为空
    if(color.empty() || depth.empty()) {
        std::cout << "Warning: Empty images for keyframe id = " << kf->mnId << std::endl;
        return;
    }

    mSensor = MappingSensor::RGBD;
    cout << "receive a keyframe, id = " << kf->mnId << endl;

    unique_lock<mutex> lck(mKeyFrameMtx);
    std::vector<KeyFrame *> keyframes;
    colorImgs.push_back(color.clone());
    depthImgs.push_back(depth.clone());
    mKeyFrameUpdated.notify_one();
}

void DenseMapping::Viewer() {
    //std::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viwer(new pcl::visualization::PCLVisualizer("Dense Mapping Viewer"));
    std::shared_ptr<pcl::visualization::CloudViewer> cloud_viwer(new pcl::visualization::CloudViewer("Dense Mapping Viewer"));
    
    while (1) {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag) {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            mKeyFrameUpdated.wait( lck_keyframeUpdated );
        }

        size_t N=0;
        {
            unique_lock<mutex> lck( mKeyFrameMtx );
            N = keyframes.size();
        }

        // 如果正在更新点云或需要停止，跳过本次循环
        if (mabIsUpdating) {
            continue;
        }

        PointCloud::Ptr p;
        // Check the sensor type to determine the point cloud generation method
        switch (mSensor) {
            case MappingSensor::RGBD:
                // Process RGBD keyframes
                for (size_t i = lastKeyframeSize; i < N; i++) {
                    p = GeneratePointCloudFromRGBD(keyframes[i], colorImgs[i], depthImgs[i]);
                    if (p && !p->empty()) {
                        (*globalMap) += *p;
                        // Optional: Add coordinate system visualization
                        // addCoordinateSystem(viewer, keyframes[i]->GetPoseInverse().matrix(), std::to_string(i));
                    }
                }
                break;

            case MappingSensor::STEREO:
                // Process stereo keyframes
                for (size_t i = lastKeyframeSize; i < N; i++) {
                    if (dispImgs.empty()) {
                        // Stereo without precomputed disparity
                        p = GeneratePointCloudFromStereo(keyframes[i], colorImgs[i], rightImgs[i], Q);
                    } else {
                        // Stereo with precomputed disparity
                        p = GeneratePointCloudFromStereoWithDisp(keyframes[i], colorImgs[i], rightImgs[i], dispImgs[i], Q);
                    }
                    
                    if (p && !p->empty()) {
                        (*globalMap) += *p;
                        // Optional: Add coordinate system visualization
                        // addCoordinateSystem(viewer, keyframes[i]->GetPoseInverse().matrix(), std::to_string(i));
                    }
                }
                break;

            default:
                std::cerr << "Error: Unknown sensor type!" << std::endl;
                break;
        }

        //PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *globalMap );
        //globalMap->swap( *tmp );

        //outlier filter
        sor.setInputCloud(globalMap);
        sor.filter(*globalMap);

        std::cout << "Global map size after filtering: " << globalMap->points.size() << std::endl;
        // if (!pcl_viewer.updatePointCloud(globalMap, "global_map")) {
        //     pcl_viewer.addPointCloud(globalMap, "global_map");
        // }
        cloud_viwer->showCloud(globalMap);
        lastKeyframeSize = N;
    }
}

void DenseMapping::SavePointCloud(const string& filename) {
    unique_lock<mutex> lck(mMutexCloud);
    pcl::io::savePCDFileBinary(filename, *globalMap);
    cout<<"globalMap save finished"<<endl;
}

// void DenseMapping::UpdateCloud(Map* pMap) {
//     // 加锁保护更新过程
//     std::unique_lock<std::mutex> lck(mMutexCloud);
    
//     mabIsUpdating = true;
//     std::cout << "开始点云更新" << std::endl;

//     // 创建临时点云
//     PointCloud::Ptr tmpGlobalMap(new PointCloud);
//     PointCloud::Ptr curPointCloud(new PointCloud);
//     PointCloud::Ptr tmpGlobalMapFilter(new PointCloud);

//     // 获取所有关键帧
//     vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    
//     // 按时间顺序排序
//     sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
    
//     // 遍历每个关键帧
//     for (size_t i = 0; i < vpKFs.size(); i++) {
//         // 检查是否需要中断更新
//         if (!mabIsUpdating) {
//             std::cout << "中断点云更新" << std::endl;
//             return;
//         }

//         KeyFrame* pKF = vpKFs[i];
//         if (pKF && !pKF->isBad()) {
//             // 生成点云
//             cv::Mat left_img = pKF->GetLeftImage();
//             cv::Mat right_img = pKF->GetRightImage();
            
//             // 生成点云，传递三个参数
//             PointCloud::Ptr cloud = GeneratePointCloudFromStereo(pKF, left_img, right_img);
//             if (cloud) {
//                 // 将点云转换到世界坐标系
//                 Sophus::SE3f Tcw = pKF->GetPose();
//                 Eigen::Matrix4f Tcw_mat = Tcw.matrix();
//                 pcl::transformPointCloud(*cloud, *curPointCloud, Tcw_mat);
                
//                 // 合并到临时点云
//                 *tmpGlobalMap += *curPointCloud;
                
//                 // 对临时点云进行滤波
//                 voxel.setInputCloud(tmpGlobalMap);
//                 voxel.filter(*tmpGlobalMapFilter);
//                 tmpGlobalMap->swap(*tmpGlobalMapFilter);
//             }
//         }
//     }

//     // 更新全局点云
//     globalMap = tmpGlobalMap;
//     std::cout << "点云更新完成" << std::endl;
    
//     mabIsUpdating = false;
// }

// RGBD point cloud generation
pcl::PointCloud<DenseMapping::PointT>::Ptr DenseMapping::GeneratePointCloudFromRGBD(
    KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp(new PointCloud());
    
    // Convert depth to float if needed
    if (depth.type() != CV_32F)
        depth.convertTo(depth, CV_32F);

    // Generate point cloud from depth
    for (int m = 0; m < depth.rows; m += 1) {
        for (int n = 0; n < depth.cols; n += 1) {
            float d = depth.ptr<float>(m)[n];
            if (d / unit < 0.01 || d / unit > 10.0)
                continue;
                
            PointT p;
            p.z = d / unit;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;

            // Use color from RGB image
            p.b = color.ptr<uchar>(m)[n * 3];
            p.g = color.ptr<uchar>(m)[n * 3 + 1];
            p.r = color.ptr<uchar>(m)[n * 3 + 2];

            tmp->points.push_back(p);
        }
    }

    // Transform to world coordinates
    PointCloud::Ptr cloud(new PointCloud());
    pcl::transformPointCloud(*tmp, *cloud, kf->GetPoseInverse().matrix());
    cloud->is_dense = false;
    //cout << "generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;
    return cloud;
}

// Stereo point cloud generation without disparity
pcl::PointCloud<DenseMapping::PointT>::Ptr DenseMapping::GeneratePointCloudFromStereo(
    KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& Q)
{
    cv::Mat left_r, right_r;
    left.copyTo(left_r);
    right.copyTo(right_r);
    
    // 确保 Q 矩阵正确对齐
    cv::Mat Q_aligned;
    Q.copyTo(Q_aligned);
    
    // Compute disparity
    cv::Mat disp = stereo->inference(left_r, right_r);
    double min = 0;
    double max = 0;
    cv::minMaxLoc(disp,&min,&max);
    std::cout<<"min disp : "<< min <<" max disp : "<<max<<std::endl;
    // disp = (disp - min) / (max - min)
    cv::Mat disp_vis = ((disp - min) / (max - min))*255;
    disp_vis.convertTo(disp_vis,CV_8U);
    cv::imshow("disp_vis",disp_vis);
    cv::waitKey(1);

    // Convert to 3D points
    cv::Mat points_image;
    cv::reprojectImageTo3D(disp, points_image, Q);
    points_image.convertTo(points_image, CV_32F);
    
    PointCloud::Ptr tmp(new PointCloud());

    // Generate point cloud
    for (int m = 0; m < points_image.rows; m += 1) {
        for (int n = 0; n < points_image.cols; n += 1) {
            float z = points_image.ptr<float>(m)[3 * n + 2];
            if (z / unit < 1.0 || z / unit > 50.0)
                continue;
                
            PointT p;
            p.x = points_image.ptr<float>(m)[3 * n];
            p.y = points_image.ptr<float>(m)[3 * n + 1];
            p.z = z;
            
            // Use color from left image
            p.b = left_r.ptr<uchar>(m)[n * 3];
            p.g = left_r.ptr<uchar>(m)[n * 3 + 1];
            p.r = left_r.ptr<uchar>(m)[n * 3 + 2];
            
            tmp->points.push_back(p);
        }
    }

    // Transform to world coordinates
    PointCloud::Ptr cloud(new PointCloud());
    pcl::transformPointCloud(*tmp, *cloud, kf->GetPoseInverse().matrix());
    cloud->is_dense = false;
    //cout << "generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;
    return cloud;
}

// Stereo point cloud generation with disparity
pcl::PointCloud<DenseMapping::PointT>::Ptr DenseMapping::GeneratePointCloudFromStereoWithDisp(
    KeyFrame* kf, cv::Mat& left, cv::Mat& right, cv::Mat& disp, cv::Mat& Q)
{
    cv::Mat left_r, disp_c, Q_aligned;
    left.copyTo(left_r);
    disp.copyTo(disp_c);
    Q.copyTo(Q_aligned);
    
    // Convert disparity if needed
    if (disp_c.type() != CV_16S)
        disp_c.convertTo(disp_c, CV_16S);
    
    // Convert to 3D points
    cv::Mat points_image;
    cv::reprojectImageTo3D(disp_c, points_image, Q, true, CV_32F);
    
    PointCloud::Ptr tmp(new PointCloud());

    // Generate point cloud
    for (int m = 0; m < points_image.rows; m += 1) {
        for (int n = 0; n < points_image.cols; n += 1) {
            float z = points_image.ptr<float>(m)[3 * n + 2];
            if (z / unit < 0.3 || z / unit > 10.0)
                continue;
                
            PointT p;
            p.x = points_image.ptr<float>(m)[3 * n];
            p.y = points_image.ptr<float>(m)[3 * n + 1];
            p.z = z;
            
            // Use color from left image
            p.b = left_r.ptr<uchar>(m)[n * 3];
            p.g = left_r.ptr<uchar>(m)[n * 3 + 1];
            p.r = left_r.ptr<uchar>(m)[n * 3 + 2];
            
            tmp->points.push_back(p);
        }
    }

    // Transform to world coordinates
    PointCloud::Ptr cloud(new PointCloud());
    pcl::transformPointCloud(*tmp, *cloud, kf->GetPoseInverse().matrix());
    cloud->is_dense = false;
    //cout << "generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;
    return cloud;
}

// 添加新的成员函数来计算Q矩阵
cv::Mat DenseMapping::ComputeQMatrix(double fx, double fy, double cx, double cy, double baseline) {
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    
    // 按照标准Q矩阵格式填充
    Q.at<double>(0, 0) = 1.0;
    Q.at<double>(0, 3) = -cx;    // -cx
    Q.at<double>(1, 1) = 1.0;
    Q.at<double>(1, 3) = -cy;    // -cy
    Q.at<double>(2, 3) = fx;     // f
    Q.at<double>(3, 2) = -1.0/baseline;  // -1/Tx
    Q.at<double>(3, 3) = 0;      // (cx - cx')/Tx = 0，假设双目校正后左右相机主点相同

    return Q;
}

}  // namespace ORB_SLAM3 