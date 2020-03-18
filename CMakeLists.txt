cmake_minimum_required(VERSION 3.5)
project(languagemodels)

set(CMAKE_CXX_STANDARD 14)

# I don't want to use find_package since I want proper control over where my LLVM comes from.
# find_package(LLVM REQUIRED)

add_executable(languagemodels
        rnn.cpp codegenc.h lang.h codegenmlir.h)

# Attempt to take these as command line arguments. IF that fails,
# lookup environment.
IF(NOT MLIR_INCLUDE_PATH)
    set (MLIR_INCLUDE_PATH $ENV{MLIR_INCLUDE_PATH})
endif()

IF(NOT MLIR_LIBRARY_PATH)
    set (MLIR_LIBRARY_PATH $ENV{MLIR_LIBRARY_PATH})
endif()

target_include_directories(languagemodels PRIVATE ${MLIR_INCLUDE_PATH})
find_library(MLIRAnalysis MLIRAnalysis ${MLIR_LIBRARY_PATH})
find_library(MLIRIR MLIRIR ${MLIR_LIBRARY_PATH})
find_library(MLIRParser MLIRParser ${MLIR_LIBRARY_PATH})
find_library(MLIRSideEffects MLIRSideEffects ${MLIR_LIBRARY_PATH})
find_library(MLIRTransforms MLIRTransforms ${MLIR_LIBRARY_PATH})
find_library(LLVMCore LLVMCore ${MLIR_LIBRARY_PATH})
find_library(LLVMSupport LLVMSupport ${MLIR_LIBRARY_PATH})

# debugging to check if it's been set properly
message(MLIR_INCLUDE_PATH ${MLIR_INCLUDE_PATH})
message(MLIR_LIBRARY_PATH ${MLIR_LIBRARY_PATH})
message(MLIRAnalysis ${MLIRAnalysis})

target_link_libraries(languagemodels
        ${MLIRAnalysis}
        ${MLIRIR}
        ${MLIRParser}
        ${MLIRSideEffects}
        ${MLIRTransforms}
        ${LLVMCore}
        ${LLVMSupport})
