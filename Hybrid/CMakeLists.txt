# 1. Basic setup
cmake_minimum_required(VERSION 3.15)
project(Hybrid)


# 2. Global settings
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set macOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")

# 3. Add dependencies
add_subdirectory(JUCE)

# 4. Define variables
# ONNX Runtime paths - modify these to match your installation
set(ONNX_RUNTIME_ROOT_PATH "${CMAKE_SOURCE_DIR}/onnxruntime-osx-universal2-1.22.0")
set(ONNX_RUNTIME_INCLUDE_DIR "${ONNX_RUNTIME_ROOT_PATH}/include")
set(ONNX_RUNTIME_LIB_DIR "${ONNX_RUNTIME_ROOT_PATH}/lib")
set(ONNX_RUNTIME_DYLIB "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.1.22.0.dylib")


# Neural model path - modify this to your model location
set(NEURAL_MODEL_PATH "${CMAKE_SOURCE_DIR}/EDCL1BTapePreamp.onnx")

# Source files
set(SOURCES
    ../../Source/PluginProcessor.cpp
    ../../Source/PluginProcessor.h
    ../../Source/PluginEditor.cpp
    ../../Source/PluginEditor.h
)


# 5. Checks/validations
# Check if ONNX Runtime exists
if(NOT EXISTS ${ONNX_RUNTIME_DYLIB})
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNX_RUNTIME_DYLIB}")
endif()
# Check if model exists
if(NOT EXISTS ${NEURAL_MODEL_PATH})
    message(FATAL_ERROR "Neural model not found at ${NEURAL_MODEL_PATH}")
endif()



# 6. Create target

# Create the VST plugin
juce_add_plugin(Hybrid
    VERSION 1.0.0
    COMPANY_NAME "UiO"
    COMPANY_COPYRIGHT "Copyright (c) 2025"
    COMPANY_WEBSITE "https://uio.com"
    COMPANY_EMAIL "info@uio.com"
    
    PLUGIN_MANUFACTURER_CODE Yco1
    PLUGIN_CODE Hybr
    
    FORMATS VST3
    
    PRODUCT_NAME "Hybrid"
    
    VST3_CATEGORIES "Fx"
    AU_MAIN_TYPE "kAudioUnitType_Effect"
    
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    
    IS_SYNTH FALSE
    WANTS_MIDI_INPUT FALSE
    PRODUCES_MIDI_OUTPUT FALSE
    
    IS_MIDI_EFFECT FALSE
    
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    
    COPY_PLUGIN_AFTER_BUILD TRUE
    
    PLUGIN_DESCRIPTION "A VST plugin with ONNX Runtime inference"
)
if(APPLE)
    set_target_properties(Hybrid PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    )
endif()
juce_generate_juce_header(Hybrid)


# 7. Configure target (any order)
# Add source files to the target
target_sources(Hybrid PRIVATE ${SOURCES})

# Include directories
target_include_directories(Hybrid PRIVATE
    ../
    ${ONNX_RUNTIME_INCLUDE_DIR}
)


# Link JUCE libraries
target_link_libraries(Hybrid PRIVATE
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_plugin_client
    juce::juce_audio_processors
    juce::juce_audio_utils
)

# Link ONNX Runtime
target_link_libraries(Hybrid PRIVATE ${ONNX_RUNTIME_DYLIB})


# Compiler definitions
target_compile_definitions(Hybrid PRIVATE
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_STRICT_REFCOUNTEDPOINTER=1
)


# 8. Platform-specific settings
# Platform-specific settings
if(APPLE)
    target_compile_definitions(Hybrid PRIVATE
        JUCE_MAC=1
    )
    
    # Set bundle identifier
    set_target_properties(Hybrid PROPERTIES
        BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.uio.Hybrid"
        MACOSX_BUNDLE_BUNDLE_NAME "Hybrid"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0.0"
    )
endif()


# 9. Bundle resources
# Using juce_add_bundle_resources_directory

# Remove quarantine from source before copying
if(APPLE)
    execute_process(
        COMMAND xattr -dr com.apple.quarantine ${ONNX_RUNTIME_LIB_DIR}
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()

#juce_add_bundle_resources_directory(Hybrid 
#    "${ONNX_RUNTIME_LIB_DIR}" 
#    DESTINATION "MacOS"
#)


# For a single model file
juce_add_bundle_resources_directory(Hybrid 
    "${NEURAL_MODEL_PATH}"
    DESTINATION "Resources"
)

if(APPLE)
    add_custom_command(TARGET Hybrid_VST3 POST_BUILD
        # Copy the dylib to the VST3 bundle
        COMMAND ${CMAKE_COMMAND} -E copy
        ${ONNX_RUNTIME_DYLIB}
        $<TARGET_FILE:Hybrid_VST3>/Contents/MacOS/
        
        # Fix dylib paths for distribution
        COMMAND install_name_tool -id "@loader_path/libonnxruntime.1.22.0.dylib" 
        $<TARGET_FILE:Hybrid_VST3>/Contents/MacOS/libonnxruntime.1.22.0.dylib
        
        COMMAND install_name_tool -change 
        "${ONNX_RUNTIME_DYLIB}"
        "@loader_path/libonnxruntime.1.22.0.dylib"
        $<TARGET_FILE:Hybrid_VST3>
        
        COMMENT "Copying and fixing ONNX Runtime dylib"
    )
endif()


# 10. Final messages

# Print configuration info
message(STATUS "ONNX Runtime library: ${ONNX_RUNTIME_DYLIB}")
message(STATUS "Neural model: ${NEURAL_MODEL_PATH}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")

# Build commands help
message(STATUS "Build with: cmake --build . --config Release")
message(STATUS "Verify bundle: cmake --build . --target verify_bundle")
