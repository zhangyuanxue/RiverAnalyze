#include "AnalysisManager.h"
#include "TensorflowThread.h"

#include <iostream>
#include <X11/Xlib.h>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <glog/logging.h>

AnalysisManager::AnalysisManager() {
    std::cout << "Support Mulit-thread show" << std::endl;
    XInitThreads();
}

AnalysisManager::~AnalysisManager() {
    for (size_t i = 0; i < inst_vec.size(); ++i) {
        auto ptr = inst_vec[i];
        if (ptr == NULL)
            continue;
        ptr->Stop();
        delete ptr;
        ptr = NULL;
    }
    TensorflowThread::Uninitalize();
}

int AnalysisManager::Initalize(const char* argv0, const char* model_path/*./model/out3.pb*/) {
    google::InitGoogleLogging(argv0);
   
    FLAGS_logbufsecs = 0;  //设置可以缓冲日志的最大秒数，0指实时输出
    FLAGS_max_log_size = 100;  //设置最大日志文件大小（以MB为单位）
    FLAGS_log_dir = "./log";
#ifdef _DEBUG
    FLAGS_logtostderr = true;
    FLAGS_alsologtostderr = true;
#else
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
#endif
    FLAGS_colorlogtostderr = true;

    return TensorflowThread::Initalize(model_path);
}

void AnalysisManager::Uninitalize() {
    TensorflowThread::Uninitalize();
    google::ShutdownGoogleLogging();
}

int AnalysisManager::LoadConfig(const char* argv/*RiverAnalysis*/, const char* config_path) {
    if (argv == NULL || config_path == NULL)
        return -1;
    int fd = open(config_path, O_RDONLY);//以只读方式打开
    if (fd < 0) {
        return -1;
    }
    google::protobuf::io::FileInputStream fileinput(fd);
    fileinput.SetCloseOnDelete(true);
    if (!google::protobuf::TextFormat::Parse(&fileinput, &config)) {
        return -1;
    }
    //做glog与tensorflow的初始化
    if (Initalize(argv, config.model_path().c_str()) != 0) 
        return -1;
    for (size_t i = 0; i < config.configs_size(); ++i) {
        const std::string& _inst_config = config.configs(i);
        Analysis* _inst = new(std::nothrow) Analysis();
        if (_inst == NULL)
            continue;
        if (_inst->LoadConfig(_inst_config.c_str()) != 0) {
            delete _inst;
            continue;
        }
        inst_vec.push_back(_inst);
    }
    LOG(INFO) << "AnalysisManager LoadConfig succ" ;
 //   std::cout<<"AnalysisManager LoadConfig succ"<<std::endl;
    return 0;
}

int AnalysisManager::run() {
    if (inst_vec.empty()) {
        LOG(ERROR) << "AnalysisManager has No inst";
        return -1;
    }
    for (auto ptr : inst_vec) {
        if (ptr == NULL)
            continue;
        ptr->Start();
    }
    sleep(10);
    while (true) {
        for (auto ptr : inst_vec) {
            if (!ptr->_is_run) {
                LOG(WARNING) << ptr->Config().input_url() << " is Stop.";
                ptr->Stop();
                LOG(WARNING) << ptr->Config().input_url() << "is Start";
                ptr->Start();
            }
            //每五秒检查一次是否有实例分析是否中断，中断则种起
            ptr->WaitFor(1000*5);
        }
    }
    return 0;
}
