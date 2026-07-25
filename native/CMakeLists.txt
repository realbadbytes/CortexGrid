cmake_minimum_required(VERSION 3.20)
project(cortex_grid_executor LANGUAGES CXX)

option(CORTEX_ENABLE_CUDA "Build the CUDA activation kernel when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)

set(CORTEX_WORKER_SOURCES
    src/main.cpp
    src/crypto.cpp
    src/executor.cpp
)

if(CORTEX_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        list(APPEND CORTEX_WORKER_SOURCES src/cuda_executor.cu)
        set(CORTEX_HAS_CUDA ON)
        message(STATUS "CortexGrid CUDA execution backend enabled")
    else()
        message(STATUS "CUDA compiler not found; building native CPU execution backend")
    endif()
endif()

add_executable(cortex_worker ${CORTEX_WORKER_SOURCES})
target_include_directories(cortex_worker PRIVATE include)
target_link_libraries(cortex_worker PRIVATE OpenSSL::Crypto)
target_compile_options(cortex_worker PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
)

if(CORTEX_HAS_CUDA)
    target_compile_definitions(cortex_worker PRIVATE CORTEX_HAS_CUDA=1)
    set_target_properties(cortex_worker PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "native"
    )
endif()

install(TARGETS cortex_worker RUNTIME DESTINATION bin)
