cmake_minimum_required(VERSION 2.8)
add_executable(predict main.cpp Classifier.cpp)
target_link_libraries(predict ${OpenCV_LIBS} ${Caffe_LIBRARIES})
