/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NeuralCL1BAudioProcessorEditor::NeuralCL1BAudioProcessorEditor (NeuralCL1BAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Set up Threshold slider
    thresholdSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    thresholdSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    thresholdSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(thresholdSlider);
    
    // Set up threshold label (NO attachToComponent - we'll position manually)
    thresholdLabel.setText("Threshold", juce::dontSendNotification);
    thresholdLabel.setJustificationType(juce::Justification::centred);
    thresholdLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(thresholdLabel);
    
    // Set up Ratio slider
    ratioSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    ratioSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    ratioSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(ratioSlider);
    
    ratioLabel.setText("Ratio", juce::dontSendNotification);
    ratioLabel.setJustificationType(juce::Justification::centred);
    ratioLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(ratioLabel);
    
    // Set up Attack slider
    attackSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    attackSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    attackSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(attackSlider);
    
    attackLabel.setText("Attack", juce::dontSendNotification);
    attackLabel.setJustificationType(juce::Justification::centred);
    attackLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(attackLabel);
    
    // Set up Release slider
    releaseSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    releaseSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    releaseSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(releaseSlider);
    
    releaseLabel.setText("Release", juce::dontSendNotification);
    releaseLabel.setJustificationType(juce::Justification::centred);
    releaseLabel.setColour(juce::Label::textColourId, juce::Colours::deepskyblue);
    addAndMakeVisible(releaseLabel);
    
    // Create parameter attachments
    thresholdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "threshold", thresholdSlider);
    ratioAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "ratio", ratioSlider);
    attackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "attack", attackSlider);
    releaseAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "release", releaseSlider);
    
    setSize(900, 200);
    
}

NeuralCL1BAudioProcessorEditor::~NeuralCL1BAudioProcessorEditor()
{
}

void NeuralCL1BAudioProcessorEditor::buttonClicked (juce::Button* button)
{

}


void NeuralCL1BAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
   
       
}
//==============================================================================
void NeuralCL1BAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll(juce::Colours::white);
    g.setColour (juce::Colours::deepskyblue);
    g.setFont (juce::FontOptions ("Helvetica", 20.0f, juce::Font:: italic));
    g.drawText ("NeuralCL1B", getLocalBounds(), juce::Justification::top, true);
}

void NeuralCL1BAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
        area.removeFromTop(30); // Title space
        area.reduce(20, 10);    // Margins
        
        // Calculate dimensions for 4 sliders in a row
        int numSliders = 4;
        int sliderWidth = area.getWidth() / numSliders;
        int spacing = 10; // Space between sliders
        
        // Define heights for different sections
        int labelHeight = 10;
        int sliderHeight = 120;  // Height for rotary slider
        
        // Create areas for each slider column
        auto thresholdColumn = area.removeFromLeft(sliderWidth);
        auto ratioColumn = area.removeFromLeft(sliderWidth);
        auto attackColumn = area.removeFromLeft(sliderWidth);
        auto releaseColumn = area.removeFromLeft(sliderWidth);
        
        // Layout Threshold Slider
        {
            auto column = thresholdColumn;
            column.removeFromRight(spacing); // Add spacing
            
            thresholdLabel.setBounds(column.removeFromTop(labelHeight));
            thresholdSlider.setBounds(column.removeFromTop(sliderHeight));
            
        }
        
        // Layout Ratio Slider
        {
            auto column = ratioColumn;
            column.removeFromRight(spacing);
            
            ratioLabel.setBounds(column.removeFromTop(labelHeight));
            ratioSlider.setBounds(column.removeFromTop(sliderHeight));
      
        }
        
        // Layout Attack Slider
        {
            auto column = attackColumn;
            column.removeFromRight(spacing);
            
            attackLabel.setBounds(column.removeFromTop(labelHeight));
            attackSlider.setBounds(column.removeFromTop(sliderHeight));
         
        }
        
        // Layout Release Slider (no spacing needed on last column)
        {
            auto column = releaseColumn;
            
            releaseLabel.setBounds(column.removeFromTop(labelHeight));
            releaseSlider.setBounds(column.removeFromTop(sliderHeight));
            
        }
 /*   auto area = getLocalBounds();
    area.removeFromTop(30); // Title space
    area.reduce(20, 10);    // Margins
    
    auto thresholdArea = area.removeFromTop(80);
    thresholdArea.removeFromLeft(60); // Label space
    thresholdSlider.setBounds(thresholdArea);
    
    //area.removeFromTop(10); // Spacing between sliders
    
    // Gain slider
    auto ratioArea = area.removeFromTop(80);
    ratioArea.removeFromLeft(60); // Label space
    ratioSlider.setBounds(ratioArea);
    
    
    auto attackArea = area.removeFromTop(120);
    attackArea.removeFromLeft(120); // Label space
    attackSlider.setBounds(attackArea);
    
    //area.removeFromTop(10); // Spacing between sliders
    
    // Gain slider
    auto releaseArea = area.removeFromTop(120);
    releaseArea.removeFromLeft(240); // Label space
    releaseSlider.setBounds(releaseArea);
*/
}
