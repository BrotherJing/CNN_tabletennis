cmake_minimum_required(VERSION 2.8)

find_path(TRACKER_INCLUDE_DIR
	NNTracker.h
	HINTS
	./include)
	
include_directories(${TRACKER_INCLUDE_DIR})

include_directories(/home/jing/caffe/build_cmake2/include)

add_library(tracker SHARED NNTracker.cpp Classifier.cpp)
target_link_libraries(tracker ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(predict main.cpp Classifier.cpp NNTracker.cpp)
target_link_libraries(predict ${OpenCV_LIBS} ${Caffe_LIBRARIES})
