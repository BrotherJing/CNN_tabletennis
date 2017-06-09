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

add_executable(tracker_test tracker_test.cpp Classifier.cpp NNTracker.cpp)
target_link_libraries(tracker_test ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(tracker_vis tracker_vis.cpp)
target_link_libraries(tracker_vis ${OpenCV_LIBS})
