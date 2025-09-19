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
class HybridAudioProcessorEditor  : public juce::AudioProcessorEditor,  public juce::Slider::Listener, juce::Button::Listener
{
public:
    HybridAudioProcessorEditor (HybridAudioProcessor&);
    ~HybridAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged (juce::Slider*) override;
    void buttonClicked (juce::Button* button) override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    HybridAudioProcessor& audioProcessor;
    
    juce::Slider cSlider;
    juce::Label cLabel;
    juce::Slider pSlider;
    juce::Label pLabel;
    juce::Slider tSlider;
    juce::Label tLabel;
    
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> cAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> pAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> tAttachment;
   
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HybridAudioProcessorEditor)
};
