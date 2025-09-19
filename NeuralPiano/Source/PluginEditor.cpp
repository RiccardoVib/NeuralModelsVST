/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NeuralPianoAudioProcessorEditor::NeuralPianoAudioProcessorEditor (NeuralPianoAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    // Set up the slider
    velSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    velSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    
    // Set slider colors for better visibility
    velSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
  
    
    addAndMakeVisible(velSlider);
    
    // Set up the label
    velLabel.setText("Velocity", juce::dontSendNotification);
    velLabel.attachToComponent(&velSlider, true);
    velLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);

    addAndMakeVisible(velLabel);
    
    
    // Set up Gain slider
    keySlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    // Set slider colors for better visibility
    keySlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
    
    keySlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    //gainSlider.setTextValueSuffix(" dB");  // Add dB suffix
    addAndMakeVisible(keySlider);
    
    keyLabel.setText("Key Number", juce::dontSendNotification);
    keyLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);
    keyLabel.attachToComponent(&keySlider, true);
    addAndMakeVisible(keyLabel);
    
    
    // Create the attachment - this connects the slider to the parameter
    velAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "v", velSlider);
    keyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "k", keySlider);
    
    setSize (400, 200);
    
}

NeuralPianoAudioProcessorEditor::~NeuralPianoAudioProcessorEditor()
{
}

void NeuralPianoAudioProcessorEditor::buttonClicked (juce::Button* button)
{

}


void NeuralPianoAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
   
       
}
//==============================================================================
void NeuralPianoAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll(juce::Colours::white);
    g.setColour (juce::Colours::deepskyblue);
    g.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));
    g.drawText ("NeuralPianoPressureController", getLocalBounds(), juce::Justification::top, true);
}

void NeuralPianoAudioProcessorEditor::resized()
{
    
    auto area = getLocalBounds();
    area.removeFromTop(30); // Title space
    area.reduce(20, 10);    // Margins
    
    auto sliderArea = area.removeFromTop(80);
    sliderArea.removeFromLeft(120); // Label space
    keySlider.setBounds(sliderArea);
    
    //area.removeFromTop(10); // Spacing between sliders
    
    // Gain slider
    auto gainArea = area.removeFromTop(120);
    gainArea.removeFromLeft(120); // Label space
    velSlider.setBounds(gainArea);

}
