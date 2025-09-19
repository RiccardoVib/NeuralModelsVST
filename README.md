# Neural Models

This repository repository is collecting all the VST created from our research.

To know more about our research visit our pages
[Neural Modeling of Musical Devices](https://riccardovib.github.io/)

[Neural Sample-based Piano](https://riccardovib.github.io/NeuralSample-basedPiano_pages/)


This code relies on JUCE 8 and onnxruntime library. This code runs on macOS systems using Intel or Apple silicon (soon Windows).


### Folder Structure

```
./_EffectName_
├── Builds
├── ExternalLibs
├── JuceLibraryCode
├── Models
├── Source
└── _EffectName_.jucer
```

# How to Build the VST

To build the VST, verify that the appropriate release of ONNX Runtime is loaded into its designated folder (from [onnxruntime-releases](https://github.com/microsoft/onnxruntime/releases)
```
./_EffectName_
├── Builds
├── ExternalLibs <--- here add onnxruntime-osx-universalX.XX.X
├── JuceLibraryCode
├── Models
├── Source
└── _EffectName_.jucer
```

To update the ONNX Runtime version replace all references to the current version with the new one in the specified configuration locations in Exporter section of the JUCE (jucer) file:

Change the name of the ONNX Runtime dylib in the Post-build shell script (replace libonnxruntime.1.19.2.dylib with the desired version).

Update External Libraries to Link (replace "onnxruntime.1.19.2" with the new version).

Modify Header Search Paths (replace "../../ExternalLibs/onnxruntime-osx-universal2-1.19.2/include" to match the new version path).

Change Extra Library Search Path (replace "../../ExternalLibs/onnxruntime-osx-universal2-1.19.2/lib" to the appropriate new version path).

# Models

To model are saved in Models folder
```
./_EffectName_
├── Builds
├── ExternalLibs 
├── JuceLibraryCode
├── Models        <--- here
├── Source
└── _EffectName_.jucer
```
