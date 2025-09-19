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
class NeuralCL1BAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    NeuralCL1BAudioProcessor();
    ~NeuralCL1BAudioProcessor() override;

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
    std::vector<const char*> inputNameCStr = {"input", "params_inputs", "params_inputs1", "params_inputs2", "params_inputs3", "states1", "states2", "hidden"};
    std::vector<const char*> outputNameCStr = {"output", "new_states1", "new_states2", "new_hidden"};
  
    // State management
    static const int STATE_SIZE = 6;
    static const int NUM_STATES = 2;
    
    static const int STATE_SIZE_FILM = 4;
    static const int NUM_STATES_FILM = 1;
    
    bool modelLoaded = false;
    
    // Per-channel states (for stereo support)
    std::vector<std::vector<float>> channelStates[NUM_STATES]; // [state_index][channel][state_data]
    std::vector<std::vector<float>> channelStates_film[NUM_STATES_FILM]; // [state_index][channel][state_data]
    
    std::vector<int64_t> stateShape = {1, 1, STATE_SIZE}; // 1x1x6 shape for each state
    std::vector<int64_t> stateShape_film = {1, 1, STATE_SIZE_FILM}; // 1x1x4 shape for each state
    std::vector<Ort::Value> inputTensor[2];

    // Batch processing buffers
    std::vector<float> inputBatchData[2];
    std::vector<float> thresholdBatchData;
    std::vector<float> ratioBatchData;
    std::vector<float> attackBatchData;
    std::vector<float> releaseBatchData;

    // Methods
    void initializeOnnxRuntime();
    void getModelInputOutputInfo();
    void loadModel(const juce::String& modelPath);
    void processWithModelBatch(juce::AudioBuffer<float>& buffer);
    void initializeStates(int numChannels);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralCL1BAudioProcessor)
};
