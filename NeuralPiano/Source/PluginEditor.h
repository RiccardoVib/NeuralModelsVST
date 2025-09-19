/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class NeuralPianoAudioProcessorEditor  : public juce::AudioProcessorEditor,  public juce::Slider::Listener, juce::Button::Listener
{
public:
    NeuralPianoAudioProcessorEditor (NeuralPianoAudioProcessor&);
    ~NeuralPianoAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged (juce::Slider*) override;
    void buttonClicked (juce::Button* button) override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    NeuralPianoAudioProcessor& audioProcessor;
    
    juce::Slider keySlider;
    juce::Label keyLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> keyAttachment;

    // Second parameter (e.g., Gain)
    juce::Slider velSlider;
    juce::Label velLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> velAttachment;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralPianoAudioProcessorEditor)
};
