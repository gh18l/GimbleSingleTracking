## CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Dec 29, 2017
#

cmake_minimum_required (VERSION 3.1)
set(name "Gimbal_Giga")
project ( ${name} )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# c++ version
set (CMAKE_CXX_STANDARD 11)

add_definitions(
    -DGOOGLE_GLOG_DLL_DECL=
	-DGLOG_NO_ABBREVIATED_SEVERITIES
)
# opencv 
if ( WIN32 )
    set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
    set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
    include_directories(${WIN_HEADER_PATH})
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3
if (WIN32)
	set(EIGEN_INCLUDE_DIR "E:/libs/eigen")
elseif ( UNIX )
		set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
endif()
include_directories(${EIGEN_INCLUDE_DIR})

# opengl
find_package(OpenGL REQUIRED)
# Ask the user to give the paths to GLFW, GLEW and GLM.
# On Linux this can be found automatically with pkg-config; 
# see http://stackoverflow.com/questions/6889264/cmake-flags-for-opengl-using-glew-and-glfw, for instance.
if ( WIN32 )
	set(GLFW_INCLUDE_PATH "E:/libs/OpenGL/include/GLFW")
	set(GLFW_LIBRARY "E:/libs/OpenGL/lib/glfw3.lib")
	set(GLEW_INCLUDE_PATH "E:/libs/OpenGL/include")
	set(GLEW_LIBRARY "E:/libs/OpenGL/lib/glew32.lib")
	set(GLEW_LIBRARY_DEBUG "E:/libs/OpenGL/lib/glew32d.lib")
	set(GLM_INCLUDE_PATH "E:/libs/OpenGL/include")
	set(GLAD_INCLUDE_PATH "E:/libs/OpenGL/include/glad")
	include_directories(${GLAD_INCLUDE_PATH}
						${GLFW_INCLUDE_PATH}
						${GLEW_INCLUDE_PATH}
						${GLM_INCLUDE_PATH})
	add_definitions(
		#-DTW_STATIC
		#-DTW_NO_LIB_PRAGMA
		#-DTW_NO_DIRECT3D
		-DGLEW_STATIC
		#-D_CRT_SECURE_NO_WARNINGS
	)
	
#	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

# cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

# pointgrey camera sdk
set(PTGREY_SDK_INCLUDE_DIR "E:/libs/Spinnaker/include")
set(PTGREY_SDK_LIBRARY_DEBUG "E:/libs/Spinnaker/lib64/vs2015/Spinnakerd_v140.lib")
set(PTGREY_SDK_LIBRARY_RELEASE "E:/libs/Spinnaker/lib64/vs2015/Spinnaker_v140.lib")
include_directories(${PTGREY_SDK_INCLUDE_DIR})

# XIMEA camera
set(XIMEA_SDK_INCLUDE_DIR "C:/XIMEA/API")
set(XIMEA_SDK_LIBRARY_DEBUG "C:/XIMEA/API/x64/xiapi64.lib")
set(XIMEA_SDK_LIBRARY_RELEASE "C:/XIMEA/API/x64/xiapi64.lib")
include_directories(${XIMEA_SDK_INCLUDE_DIR})

#ceres solver
set(CERES_INCLUDE_DIR "E:/libs/ceres_solver/include")
set (CERES_LIBS
    debug "E:/libs/ceres_solver/lib/ceresd.lib"
    debug "E:/libs/ceres_solver/lib/libglog_staticd.lib"
    optimized "E:/libs/ceres_solver/lib/ceres.lib"
    optimized "E:/libs/ceres_solver/lib/libglog_static.lib"
)
include_directories(${CERES_INCLUDE_DIR}
					${CERES_INCLUDE_DIR}/ceres
					${CERES_INCLUDE_DIR}/glog)


					
set(MY_HEADER_FILES
    CameraDriver/Exceptions.h
    CameraDriver/helper_cuda.h
    CameraDriver/helper_string.h
    CameraDriver/NPPJpegCoder.h
    CameraDriver/GenCameraDriver.h
    CameraDriver/RealCameraDriver.h
    # point grey camera
    PointGrey/PointGreyCamera.h
    # XIMEA camera
    XIMEA/XIMEACamera.h
	#giga
	giga/Quad.h
    giga/Mesh.h
    giga/ASAP.h
    giga/FeatureMatch.h
    giga/TinyStitcher.h
	giga/DeepFlow.h
    giga/SysUtil.hpp
	#serial
	Serial/Serial.h
	#cost
	Cost/Cost.h
	#common
	Common/Common.hpp
	#RobustStitcher
	RobustStitcher/FeatureMatch.h
	RobustStitcher/CameraParamEstimator.h
    RobustStitcher/Compositor.h
    RobustStitcher/BundleAdjustment.h
	#OpenGL
	OpenGL/shader_m.h
	OpenGL/glad.c
	OpenGL/Display.h
	#KCF
	KCF/ffttools.hpp
	KCF/fhog.hpp
	KCF/kcftracker.hpp
	KCF/labdata.hpp
	KCF/recttools.hpp
	KCF/tracker.h
)

set(MY_SOURCE_FILES 
    # point grey camera
    PointGrey/PointGreyCamera.cpp
    # XIMEA camera
    XIMEA/XIMEACamera.cpp
    # C++ source file
    CameraDriver/NPPJpegCoder.cpp
    # generic camera driver
    CameraDriver/GenCameraDriver.cpp
	CameraDriver/GenCameraDriverImpl.cpp
    CameraDriver/RealCameraDriver.cpp
	#giga
	giga/Quad.cpp
    giga/Mesh.cpp
    giga/ASAP.cpp
    giga/ASAPadd.cpp
    giga/ASAPwarp.cpp
    giga/FeatureMatch.cpp
    giga/TinyStitcher.cpp
	giga/DeepFlow.cpp
	#serial
	Serial/Serial.cpp
	#cost
	Cost/Cost.cpp
	#RobustStitcher
	RobustStitcher/FeatureMatch.cpp
	RobustStitcher/CameraParamEstimator.cpp
    RobustStitcher/Compositor.cpp
    RobustStitcher/BundleAdjustment.cpp
	#OpenGL
	OpenGL/display.cpp
	#KCF
	KCF/fhog.cpp
	KCF/kcftracker.cpp
    # main function
    main.cpp
)

source_group("PointGrey" FILES
    PointGrey/PointGreyCamera.h
    PointGrey/PointGreyCamera.cpp 
)

source_group("XIMEA" FILES
    XIMEA/XIMEACamera.h
    XIMEA/XIMEACamera.cpp
)

source_group("cuda" FILES
    cuda/NPPJpegCoderKernel.h
    cuda/NPPJpegCoderKernel.cu
)
source_group("giga" FILES
    giga/Quad.cpp
    giga/Mesh.cpp
    giga/ASAP.cpp
    giga/ASAPadd.cpp
    giga/ASAPwarp.cpp
    giga/FeatureMatch.cpp
    giga/TinyStitcher.cpp
	giga/DeepFlow.cpp
	giga/Quad.h
    giga/Mesh.h
    giga/ASAP.h
    giga/FeatureMatch.h
    giga/TinyStitcher.h
	giga/DeepFlow.h
    giga/SysUtil.hpp	
)
source_group("CameraDriver" FILES
    # C++ source file
    CameraDriver/NPPJpegCoder.cpp
    # generic camera driver
    CameraDriver/GenCameraDriver.cpp
	CameraDriver/GenCameraDriverImpl.cpp
    CameraDriver/RealCameraDriver.cpp
	CameraDriver/Exceptions.h
    CameraDriver/helper_cuda.h
    CameraDriver/helper_string.h
    CameraDriver/NPPJpegCoder.h
    CameraDriver/GenCameraDriver.h
    CameraDriver/RealCameraDriver.h	
)
source_group("RobustStitcher" FILES
    RobustStitcher/FeatureMatch.cpp
	RobustStitcher/CameraParamEstimator.cpp
    RobustStitcher/Compositor.cpp
    RobustStitcher/BundleAdjustment.cpp
	RobustStitcher/FeatureMatch.h
	RobustStitcher/CameraParamEstimator.h
    RobustStitcher/Compositor.h
    RobustStitcher/BundleAdjustment.h
)
source_group("KCF" FILES
	KCF/fhog.cpp
	KCF/kcftracker.cpp
	KCF/ffttools.hpp
	KCF/fhog.hpp
	KCF/kcftracker.hpp
	KCF/labdata.hpp
	KCF/recttools.hpp
	KCF/tracker.h
)


set(YOLO_LIBS_DEBUG
    debug E:/libs/cuDNN/lib/x64/cudnn.lib
	debug E:/libs/YOLO/lib/Debug/pthreadVC2.lib
	#E:/libs/YOLO/lib/Debug/yolo_exe.lib
	#E:/libs/YOLO/lib/Debug/yolo_lib_exe.lib
	debug E:/libs/YOLO/lib/Debug/yolo_libd.lib
)
set(YOLO_LIBS_RELEASE
    optimized E:/libs/cuDNN/lib/x64/cudnn.lib
	optimized E:/libs/YOLO/lib/Release/pthreadVC2.lib
	#E:/libs/YOLO/lib/Release/yolo_exe.lib
	#E:/libs/YOLO/lib/Release/yolo_lib_exe.lib
	optimized E:/libs/YOLO/lib/Release/yolo_lib.lib
)

include_directories(
	${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/CameraDriver
	${PROJECT_SOURCE_DIR}/giga
	${PROJECT_SOURCE_DIR}/Serial
	${PROJECT_SOURCE_DIR}/XIMEA	
	${PROJECT_SOURCE_DIR}/PointGrey		
	${PROJECT_SOURCE_DIR}/Cost
	${PROJECT_SOURCE_DIR}/Common
	${PROJECT_SOURCE_DIR}/RobustStitcher
	${PROJECT_SOURCE_DIR}/OpenGL
	${PROJECT_SOURCE_DIR}/KCF
	E:/libs/YOLO/include
    E:/libs/cuDNN/include
)

cuda_add_executable(${name}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
	YOLO/YOLODetector.hpp
)

add_definitions(-DYOLO_DETECTOR_EXPORTS)
add_definitions(-DYOLODLL_EXPORTS)
add_definitions(-DGPU)
add_definitions(-DCUDNN)
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

target_link_libraries(${name}
    ${OpenCV_LIBS}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
	${CUDA_LIBRARIES}
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_curand_LIBRARY}
	${CERES_LIBS}
	${GLFW_LIBRARY}
    ${PTGREY_SDK_LIBRARY_DEBUG} 
	${XIMEA_SDK_LIBRARY_DEBUG} 
	${YOLO_LIBS_DEBUG}
    ${PTGREY_SDK_LIBRARY_RELEASE} 
	${XIMEA_SDK_LIBRARY_RELEASE} 
	${YOLO_LIBS_RELEASE}
)