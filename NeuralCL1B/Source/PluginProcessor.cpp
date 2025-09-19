/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NeuralCL1BAudioProcessor::NeuralCL1BAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ), parameters(*this, nullptr, "Parameters",
                                     juce::AudioProcessorValueTreeState::ParameterLayout{
                                         std::make_unique<juce::AudioParameterFloat>(
                                             "threshold", "Threshold", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f),
                           std::make_unique<juce::AudioParameterFloat>(
                               "ratio", "Ratio", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f),
                           std::make_unique<juce::AudioParameterFloat>(
                               "attack", "Attack", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f),
             std::make_unique<juce::AudioParameterFloat>(
                 "release", "Release", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f)
                                     })
#endif
{
    initializeOnnxRuntime();
}

NeuralCL1BAudioProcessor::~NeuralCL1BAudioProcessor()
{
}

void NeuralCL1BAudioProcessor::initializeOnnxRuntime()
{
    try
    {
        // Initialize ONNX Runtime environment
        ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeuralCL1BAudioProcessorVST");
        
        // Create session options
        ortSessionOptions = std::make_unique<Ort::SessionOptions>();
        ortSessionOptions->SetIntraOpNumThreads(1); // Important for audio plugins!
        ortSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ortSessionOptions->SetExecutionMode(ORT_SEQUENTIAL); // Sequential execution across nodes
        ortSessionOptions->EnableMemPattern();
        ortSessionOptions->EnableCpuMemArena();
        
        DBG("ONNX Runtime initialized successfully");
    }
    catch (const std::exception& e)
    {
        DBG("Failed to initialize ONNX Runtime: " + juce::String(e.what()));
    }
}

void NeuralCL1BAudioProcessor::initializeStates(int numChannels)
{
    for (int stateIdx = 0; stateIdx < NUM_STATES; ++stateIdx)
    {
        channelStates[stateIdx].resize(numChannels);
        for (int channel = 0; channel < numChannels; ++channel)
        {
            channelStates[stateIdx][channel].resize(STATE_SIZE, 0.0f);
        }
    }
    for (int stateIdx = 0; stateIdx < NUM_STATES_FILM; ++stateIdx)
    {
        channelStates_film[stateIdx].resize(numChannels);
        for (int channel = 0; channel < numChannels; ++channel)
        {
            channelStates_film[stateIdx][channel].resize(STATE_SIZE_FILM, 0.0f);
        }
    }
    DBG("States initialized for " + juce::String(numChannels) + " channels");
}
//==============================================================================
const juce::String NeuralCL1BAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool NeuralCL1BAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool NeuralCL1BAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool NeuralCL1BAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double NeuralCL1BAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int NeuralCL1BAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int NeuralCL1BAudioProcessor::getCurrentProgram()
{
    return 0;
}

void NeuralCL1BAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String NeuralCL1BAudioProcessor::getProgramName (int index)
{
    return {};
}

void NeuralCL1BAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

void NeuralCL1BAudioProcessor::loadModel(const juce::String& modelPath)
{
    try
    {
        // Load ONNX model
        ortSession = std::make_unique<Ort::Session>(*ortEnv, modelPath.toStdString().c_str(), *ortSessionOptions);
        
        modelLoaded = true;
        DBG("ONNX model loaded successfully: " + modelPath);
        
    }
    catch (const std::exception& e)
    {
        DBG("Failed to load ONNX model: " + juce::String(e.what()));
        modelLoaded = false;
    }

}

//==============================================================================
void NeuralCL1BAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
  
    // Pre-allocate ONNX input and output buffers to avoid dynamic allocations in processBlock
    
    inputBatchData[0].resize(samplesPerBlock, 0.0f);
    inputBatchData[1].resize(samplesPerBlock, 0.0f);
    thresholdBatchData.resize(samplesPerBlock);
    ratioBatchData.resize(samplesPerBlock);
    attackBatchData.resize(samplesPerBlock);
    releaseBatchData.resize(samplesPerBlock);

    // Initialize states for the current channel configuration
    initializeStates(getTotalNumOutputChannels());
    
    auto modelPath = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                        .getParentDirectory()
                        .getParentDirectory()
                        .getChildFile("Resources")
                        .getChildFile("CL1B_nof.onnx")
                        .getFullPathName();
    
    
    DBG("Attempting to load model from: " + modelPath);
    
    NeuralCL1BAudioProcessor::loadModel(modelPath);
    
    DBG("PrepareToPlay - numSamples: " + juce::String(samplesPerBlock));
    DBG("PrepareToPlay - numChannels: " + juce::String(getTotalNumOutputChannels()));
    
    
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(samplesPerBlock), 1};
    std::vector<int64_t> condShape = {1, static_cast<int64_t>(samplesPerBlock), 1};

    // Create tensors once
    try {
        for (int channel = 0; channel < getTotalNumOutputChannels(); ++channel)
        {
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
         memoryInfo, inputBatchData[channel].data(), inputBatchData[channel].size(),
         inputShape.data(), inputShape.size()));
        
        DBG("inputBatchData.size(): " + juce::String((int)inputBatchData[channel].size()));
        DBG("inputShape: [" + juce::String((int)inputShape[0]) + ", " +
                             juce::String((int)inputShape[1]) + ", " +
                             juce::String((int)inputShape[2]) + "]");
        DBG("Expected tensor size: " + juce::String((int)(inputShape[0] * inputShape[1] * inputShape[2])));
               
        DBG("condBatchData.size(): " + juce::String((int)thresholdBatchData.size()));
        DBG("condShape: [" + juce::String((int)condShape[0]) + ", " +
                             juce::String((int)condShape[1]) + ", " +
                             juce::String((int)condShape[2]) + "]");
        
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
         memoryInfo, thresholdBatchData.data(), thresholdBatchData.size(),
         condShape.data(), condShape.size()));
        
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
         memoryInfo, ratioBatchData.data(), ratioBatchData.size(),
         condShape.data(), condShape.size()));

        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
         memoryInfo, attackBatchData.data(), attackBatchData.size(),
         condShape.data(), condShape.size()));
        
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
         memoryInfo, releaseBatchData.data(), releaseBatchData.size(),
         condShape.data(), condShape.size()));
        
        DBG("Expected tensor size: " + juce::String((int)(condShape[0] * condShape[1] * condShape[2])));
                 
            // Create state tensors (1 x 6 each)
            for (int stateIdx = 0; stateIdx < NUM_STATES; ++stateIdx)
              {
                  inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
                      memoryInfo,
                      channelStates[stateIdx][channel].data(),
                      channelStates[stateIdx][channel].size(),
                      stateShape.data(),
                      stateShape.size()));
              }
    
            // Create state tensors (1 x 4 each)
            for (int stateIdx = 0; stateIdx < NUM_STATES_FILM; ++stateIdx)
            {
                  inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
                      memoryInfo,
                      channelStates_film[stateIdx][channel].data(),
                      channelStates_film[stateIdx][channel].size(),
                      stateShape_film.data(),
                      stateShape_film.size()));
            }
    }

     DBG("Tensors created successfully");
    }
    catch (const std::exception& e) {
     DBG("Failed to create tensors: " + juce::String(e.what()));
    }
}

void NeuralCL1BAudioProcessor::releaseResources()
{
    inputBatchData[0].clear();
    inputBatchData[1].clear();
    thresholdBatchData.clear();
    ratioBatchData.clear();
    attackBatchData.clear();
    releaseBatchData.clear();
}


#ifndef JucePlugin_PreferredChannelConfigurations
bool NeuralCL1BAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void NeuralCL1BAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());
        
    // Process with ML model if loaded
    if (modelLoaded)
    {
        processWithModelBatch(buffer);
    }
}

void NeuralCL1BAudioProcessor::processWithModelBatch(juce::AudioBuffer<float>& buffer)
{
    
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    std::vector<float> inputSamples;
    inputSamples.resize(numSamples);
    float ratio = *parameters.getRawParameterValue("ratio");
    float threshold = *parameters.getRawParameterValue("threshold");
    float attack = *parameters.getRawParameterValue("attack");
    float release = *parameters.getRawParameterValue("release");

    std::fill(thresholdBatchData.begin(), thresholdBatchData.end(), threshold);
    std::fill(ratioBatchData.begin(), ratioBatchData.end(), ratio);
    std::fill(attackBatchData.begin(), attackBatchData.end(), attack);
    std::fill(releaseBatchData.begin(), releaseBatchData.end(), release);

    //DBG("ratioBatchData: " + juce::String(ratioBatchData[0]));

    // Process each channel independently
    for (int channel = 0; channel < numChannels; ++channel)
    {
        float* channelData = buffer.getWritePointer(channel);
        
        try
        {
            std::copy(channelData, channelData + numSamples, inputBatchData[channel].begin());
            
            // Perform inference
            auto outputTensor = ortSession->Run(
                   Ort::RunOptions{nullptr},
                   inputNameCStr.data(),
                   inputTensor[channel].data(),
                   inputNameCStr.size(), // Number of inputs
                   outputNameCStr.data(),
                   outputNameCStr.size() // Number of outputs
               );
            const float* outputData = outputTensor[0].GetTensorMutableData<float>();
            
            // Update states with new state values
            for (int stateIdx = 0; stateIdx < NUM_STATES; ++stateIdx)
            {
                const float* newStateData = outputTensor[stateIdx + 1].GetTensorMutableData<float>();
                std::copy(newStateData, newStateData + STATE_SIZE,
                         channelStates[stateIdx][channel].begin());
                
                //DBG("newStateData: " + juce::String(*newStateData));
            }
            for (int stateIdx = 0; stateIdx < NUM_STATES_FILM; ++stateIdx)
            {
                const float* newStateData = outputTensor[stateIdx + 3].GetTensorMutableData<float>();
                std::copy(newStateData, newStateData + STATE_SIZE_FILM,
                         channelStates_film[stateIdx][channel].begin());
                
                //DBG("newStateData: " + juce::String(*newStateData));

            }
            for (int i = 0; i < numSamples; ++i)
            {

                channelData[i] = softLimit(outputData[i]);
                //DBG("channelData: " + juce::String(channelData[i]));
            }
            
        }
        catch (const std::exception& e)
        {
            DBG("Error processing batch: " + juce::String(e.what()));
        }
    }
}



//==============================================================================
bool NeuralCL1BAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* NeuralCL1BAudioProcessor::createEditor()
{
    return new NeuralCL1BAudioProcessorEditor (*this);
}

//==============================================================================
void NeuralCL1BAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void NeuralCL1BAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralCL1BAudioProcessor();
}
