cmake_minimum_required(VERSION 2.8)
project(ProbMap)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

add_subdirectory(src)
