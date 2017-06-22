add_library(jniisee_mscan_reid SHARED jni_isee_mscan_reid.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

target_link_libraries(jniisee_mscan_reid isee_mscan_reid)

include_directories(jniisee_mscan_reid PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(jniisee_mscan_reid PUBLIC ${JAVA_HOME}/include)

if (MSVC)
    include_directories(jniisee_mscan_reid PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories(jniisee_mscan_reid PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install(TARGETS jniisee_mscan_reid DESTINATION lib)
