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
class NeuralCL1BAudioProcessorEditor  : public juce::AudioProcessorEditor,  public juce::Slider::Listener, juce::Button::Listener
{
public:
    NeuralCL1BAudioProcessorEditor (NeuralCL1BAudioProcessor&);
    ~NeuralCL1BAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged (juce::Slider*) override;
    void buttonClicked (juce::Button* button) override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    NeuralCL1BAudioProcessor& audioProcessor;
    
    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    juce::Slider ratioSlider;
    juce::Label ratioLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> ratioAttachment;

    juce::Slider attackSlider;
    juce::Label attackLabel;
    juce::Slider releaseSlider;
    juce::Label releaseLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> releaseAttachment;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralCL1BAudioProcessorEditor)
};
