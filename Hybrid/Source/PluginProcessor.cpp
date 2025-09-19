/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
HybridAudioProcessor::HybridAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ), parameters(*this, nullptr, "Parameters",
                                    
                         juce::AudioProcessorValueTreeState::ParameterLayout{std::make_unique<juce::AudioParameterFloat>("t", "Tape", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.f),
                             std::make_unique<juce::AudioParameterFloat>("p", "PreAmp", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.f),
                             std::make_unique<juce::AudioParameterFloat>("c", "Comp", juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.f)
                                     })
#endif
{
    initializeOnnxRuntime();
}

HybridAudioProcessor::~HybridAudioProcessor()
{
}

void HybridAudioProcessor::initializeOnnxRuntime()
{
    try
    {
        // Initialize ONNX Runtime environment
        ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HybridAudioVST");
        
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


void HybridAudioProcessor::initializeStates(int numChannels)
{
    for (int stateIdx = 0; stateIdx < NUM_STATES; ++stateIdx)
    {
        channelStates[stateIdx].resize(numChannels);
        for (int channel = 0; channel < numChannels; ++channel)
        {
            channelStates[stateIdx][channel].resize(STATE_SIZE, 0.0f);
        }
    }
    DBG("States initialized for " + juce::String(numChannels) + " channels");
}

//==============================================================================
const juce::String HybridAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool HybridAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool HybridAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool HybridAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double HybridAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int HybridAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int HybridAudioProcessor::getCurrentProgram()
{
    return 0;
}

void HybridAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String HybridAudioProcessor::getProgramName (int index)
{
    return {};
}

void HybridAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

void HybridAudioProcessor::loadModel(const juce::String& modelPath)
{
    try
    {
        // Load ONNX model
        ortSession = std::make_unique<Ort::Session>(*ortEnv, modelPath.toStdString().c_str(), *ortSessionOptions);

        // Get model input/output information
        //getModelInputOutputInfo();
        
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
void HybridAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Prepare batch processing buffers
    // Each buffer needs to hold samplesPerBlock * sequence_length elements
    //encoderBatchData.resize(getTotalNumOutputChannels());
    //decoderBatchData.resize(getTotalNumOutputChannels());
    //historyData.resize(getTotalNumOutputChannels());
    //inputTensor.resize(getTotalNumOutputChannels());

    for (int channel = 0; channel < getTotalNumOutputChannels(); ++channel)
     {
         // These will be resized dynamically in processWithModelBatch
         inputBatchData[channel].clear();
         inputBatchData[channel].resize(samplesPerBlock);
         
     }
    cBatchData.resize(samplesPerBlock);
    tBatchData.resize(samplesPerBlock);
    pBatchData.resize(samplesPerBlock);
    // Initialize states for the current channel configuration
    initializeStates(getTotalNumOutputChannels());
    
    auto modelPath = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                        .getParentDirectory()
                        .getParentDirectory()
                        .getChildFile("Resources")
                        .getChildFile("CL1BTapePreamp__lstm_8.onnx")
                        .getFullPathName();
    
    
    DBG("Attempting to load ONNX model from: " + modelPath);

    HybridAudioProcessor::loadModel(modelPath);
    
    DBG("PrepareToPlay - numSamples: " + juce::String(samplesPerBlock));
    DBG("PrepareToPlay - numChannels: " + juce::String(getTotalNumOutputChannels()));
    
    
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(samplesPerBlock), 1};
    std::vector<int64_t> condShape = {1, static_cast<int64_t>(samplesPerBlock), 1};
    std::vector<int64_t> stateShape = {1, 1, 8};

    // Create tensors once
    try {
        for (int channel = 0; channel < getTotalNumInputChannels(); ++channel)
        {
        DBG("inputBatchData.size(): " + juce::String((int)inputBatchData[channel].size()));
        DBG("inputShape: [" + juce::String((int)inputShape[0]) + ", " +
                             juce::String((int)inputShape[1]) + ", " +
                             juce::String((int)inputShape[2]) + "]");
        DBG("Expected tensor size: " + juce::String((int)(inputShape[0] * inputShape[1] * inputShape[2])));
               
        DBG("condBatchData.size(): " + juce::String((int)cBatchData.size()));
        DBG("condShape: [" + juce::String((int)condShape[0]) + ", " +
                             juce::String((int)condShape[1]) + "]");
        
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
             memoryInfo, inputBatchData[channel].data(), inputBatchData[channel].size(),
             inputShape.data(), inputShape.size()));
        
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
             memoryInfo, cBatchData.data(), cBatchData.size(),
             condShape.data(), condShape.size()));
            
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
             memoryInfo, tBatchData.data(), tBatchData.size(),
             condShape.data(), condShape.size()));
            
        inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
             memoryInfo, pBatchData.data(), pBatchData.size(),
             condShape.data(), condShape.size()));
            
        DBG("Expected tensor size: " + juce::String((int)(condShape[0] * condShape[1])));
                 
        // Create state tensors (1 x 8 each)
        for (int stateIdx = 0; stateIdx < NUM_STATES; ++stateIdx)
        {
            inputTensor[channel].push_back(Ort::Value::CreateTensor<float>(
                      memoryInfo,
                      channelStates[stateIdx][channel].data(),
                      channelStates[stateIdx][channel].size(),
                      stateShape.data(),
                      stateShape.size()));
        }
    }
    DBG("Tensors created successfully");
    }
    catch (const std::exception& e) {
        DBG("Failed to create tensors: " + juce::String(e.what()));
    }
    
}

void HybridAudioProcessor::releaseResources()
{
    inputBatchData[0].clear();
    inputBatchData[1].clear();

    cBatchData.clear();
    tBatchData.clear();
    pBatchData.clear();
}


#ifndef JucePlugin_PreferredChannelConfigurations
bool HybridAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
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

void HybridAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
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

void HybridAudioProcessor::processWithModelBatch(juce::AudioBuffer<float>& buffer)
{
    
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    float cValue = *parameters.getRawParameterValue("c");
    float tValue = *parameters.getRawParameterValue("t");
    float pValue = *parameters.getRawParameterValue("p");
    
    // Conditioning inputs - same values for all samples in the batch
    std::fill(cBatchData.begin(), cBatchData.end(), cValue);
    std::fill(tBatchData.begin(), tBatchData.end(), tValue);
    std::fill(pBatchData.begin(), pBatchData.end(), pValue);

    // Prepare encoder inputs: for each sample, get the 63 previous samples
    for (int channel = 0; channel < numChannels; ++channel)
    {
        float* channelData = buffer.getWritePointer(channel);
    
        try
        {
            std::copy(channelData, channelData + numSamples, inputBatchData[channel].begin());
                
            // Perform inference
            auto outputTensor = ortSession->Run(
                           Ort::RunOptions{nullptr},
                           inputNamesCStr.data(),
                           inputTensor[channel].data(),
                           inputNamesCStr.size(), // Number of inputs
                           outputNamesCStr.data(),
                           outputNamesCStr.size() // Number of outputs
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
            for (int i = 0; i < numSamples; ++i)
            {
                channelData[i] = softLimit(outputData[i]);
                DBG("channelData: " + juce::String(channelData[i]));
            }
        }
        catch (const std::exception& e)
        {
            DBG("Error processing batch: " + juce::String(e.what()));
        }
    }
}

//==============================================================================
bool HybridAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* HybridAudioProcessor::createEditor()
{
    return new HybridAudioProcessorEditor (*this);
}

//==============================================================================
void HybridAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void HybridAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HybridAudioProcessor();
}
