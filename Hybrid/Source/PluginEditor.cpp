/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
HybridAudioProcessorEditor::HybridAudioProcessorEditor (HybridAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Set up the slider
    cSlider.setSliderStyle(juce::Slider::SliderStyle::LinearVertical);
    cSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 50, 15);

    addAndMakeVisible(cSlider);
    
    // Set up the label
    cLabel.setText ("Tape", juce::dontSendNotification);
    cLabel.attachToComponent (&cSlider, false);
    cLabel.setJustificationType(juce::Justification::centred);
    cLabel.setColour (juce::Label::textColourId, juce::Colours::deepskyblue);
    cLabel.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));

    addAndMakeVisible(cLabel);
    
    // Set up the slider
    tSlider.setSliderStyle(juce::Slider::SliderStyle::LinearVertical);
    tSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 50, 15);

    addAndMakeVisible(tSlider);
    
    // Set up the label
    tLabel.setText ("PreAmp", juce::dontSendNotification);
    tLabel.attachToComponent (&tSlider, false);
    tLabel.setJustificationType(juce::Justification::centred);
    tLabel.setColour (juce::Label::textColourId, juce::Colours::deepskyblue);
    tLabel.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));

    addAndMakeVisible(tLabel);
    
    // Set up the slider
    pSlider.setSliderStyle(juce::Slider::SliderStyle::LinearVertical);
    pSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 50, 15);

    addAndMakeVisible(pSlider);
    
    // Set up the label
    pLabel.setText ("Compressor", juce::dontSendNotification);
    pLabel.attachToComponent (&pSlider, false);
    pLabel.setJustificationType(juce::Justification::centred);
    pLabel.setColour (juce::Label::textColourId, juce::Colours::deepskyblue);
    pLabel.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));

    addAndMakeVisible(pLabel);
    
    // Create the attachment - this connects the slider to the parameter
    cAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "c", cSlider);
    tAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "t", tSlider);
    pAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "p", pSlider);
    setSize (400, 300);
    
}

HybridAudioProcessorEditor::~HybridAudioProcessorEditor()
{
}

void HybridAudioProcessorEditor::buttonClicked (juce::Button* button)
{

}


void HybridAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
   
       
}
//==============================================================================
void HybridAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll(juce::Colours::white);
    g.setColour (juce::Colours::deepskyblue);
    g.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));
    g.drawText ("Hybrid", getLocalBounds(), juce::Justification::top, true);
}

void HybridAudioProcessorEditor::resized()
{
    
    cSlider.setBounds(50, 80, 100, 150);
    tSlider.setBounds(150, 80, 100, 150);
    pSlider.setBounds(250, 80, 100, 150);
    
}
