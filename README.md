# NeuralNet
This is the NeuralNet module for the Swift AI project. Full details on the project can be found in the [main repo](https://github.com/Swift-AI/Swift-AI).

The `NeuralNet` class contains a fully connected, 3-layer feed-forward artificial neural network. This neural net uses a standard backpropagation training algorithm, and is designed for flexibility and use in performance-critical applications.

## Importing

### Swift Package Manager
SPM makes it easy to import the package into your own project. Just add this line to `package.swift`:
```swift
.Package(url: "https://github.com/Swift-AI/NeuralNet.git", majorVersion: 0, minor: 1)
```

### Manually
Since SPM doesn't support iOS or Cocoa applications yet, you may need to import the package manually. To do this, simply drag and drop the files from `Sources` into your project.

This isn't as elegant as using a package manager, but we anticipate SPM support for iOS soon. For this reason we've decided not to use alternatives like CocoaPods.

## Usage
(Full documentation coming soon)
