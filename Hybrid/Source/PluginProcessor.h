/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

//==============================================================================
/**
*/
class HybridAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    HybridAudioProcessor();
    ~HybridAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }

private:
    //==============================================================================
    float softLimit(float input, float threshold = 0.8f)
    {
        if (std::abs(input) <= threshold)
            return input;
        
        //float sign = std::copysign(1.0f, input);
        //float excess = std::abs(input) - threshold;
        //float limitedExcess = threshold * std::tanh(excess / threshold);
        
        //return sign * (threshold + limitedExcess);
        return std::tanh(input);
    }
    
    juce::AudioProcessorValueTreeState parameters;

    // ONNX Runtime components
    std::unique_ptr<Ort::Env> ortEnv;
    Ort::AllocatorWithDefaultOptions ortAllocator;
    std::unique_ptr<Ort::SessionOptions> ortSessionOptions;
    Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU)};
    std::unique_ptr<Ort::Session> ortSession;
    
    // Model information
    std::vector<const char*> inputNamesCStr{"inputs", "c", "p", "t", "h1", "h2"};

    std::vector<const char*> outputNamesCStr{"outputs", "new_h1", "new_h2"};
    
  
    // Model parameters
    static constexpr int CONDITIONING_SIZE = 1;      // User controllable parameters
    
    // State management
    static const int STATE_SIZE = 8;
    static const int NUM_STATES = 2;
    
    
    // Create input tensors
 
    std::vector<std::vector<float>> channelStates[NUM_STATES]; // [state_index][channel][state_data]
    std::vector<Ort::Value> inputTensor[2];
    
    bool modelLoaded = false;

    // Batch processing buffers
    std::vector<float> inputBatchData[2];
    std::vector<float> cBatchData;
    std::vector<float> tBatchData;
    std::vector<float> pBatchData;

    // Methods
    void initializeOnnxRuntime();
    void getModelInputOutputInfo();
    void loadModel(const juce::String& modelPath); 
    void processWithModelBatch(juce::AudioBuffer<float>& buffer);
    void initializeStates(int numChannels);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HybridAudioProcessor)
};
