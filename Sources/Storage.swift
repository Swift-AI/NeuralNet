//
//  Storage.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/4/17.
//
//

import Foundation


// Utilities for persisting/retrieving a NeuralNet from disk.

public extension NeuralNet {
    
    // -------------------
    // NOTE: Our storage protocol writes data to JSON file in plaintext,
    // rather than adopting a standard storage protocol like NSCoding.
    // This will allow Swift AI components to be written/read across platforms without compatibility issues.
    // -------------------
    
    
    // MARK: JSON keys
    
    static let layerNodeCountsKey = "layerNodeCounts"
    static let momentumKey = "momentum"
    static let learningRateKey = "learningRate"
    static let batchSizeKey = "batchSize"
    static let hiddenActivationKey = "hiddenActivation"
    static let outputActivationKey = "outputActivation"
    static let weightsKey = "weights"
    
    
    /// Attempts to initialize a `NeuralNet` from a file stored at the given URL.
    public convenience init(url: URL) throws {
        // Read data
        let data = try Data(contentsOf: url)
        // Extract top-level object from data
        let json = try JSONSerialization.jsonObject(with: data, options: [])
        // Attempt to convert object into an Array
        guard let array = json as? [String : Any] else {
            throw Error.initialization("Unable to read JSON data from file.")
        }
        
        // Read all required values from JSON
        guard let layerNodeCounts = array[NeuralNet.layerNodeCountsKey] as? [Int],
            let lr = array[NeuralNet.learningRateKey] as? Float,
            let momentum = array[NeuralNet.momentumKey] as? Float,
            let batchSize = array[NeuralNet.batchSizeKey] as? Int,
            let hiddenActivationStr = array[NeuralNet.hiddenActivationKey] as? String,
            let outputActivationStr = array[NeuralNet.outputActivationKey] as? String,
            let weights = array[NeuralNet.weightsKey] as? [[Float]]
            else {
                throw Error.initialization("One or more required NeuralNet properties are missing.")
        }
        
        // Convert hidden activation function to enum
        var hiddenActivation: ActivationFunction.Hidden
        // Check for custom activation function
        if hiddenActivationStr == "custom" {
            // Note: Here we simply warn the user and set the activation to defualt (sigmoid)
            // The user should reset the activation to the original custom function manually
            print("NeuralNet warning: custom hidden layer activation function detected in stored network. Defaulting to sigmoid activation. It is your responsibility to reset the network's hidden layer activation to the original function, or it is unlikely to perform correctly.")
            hiddenActivation = ActivationFunction.Hidden.sigmoid
        } else {
            guard let function = ActivationFunction.Hidden.from(hiddenActivationStr) else {
                throw Error.initialization("Unrecognized hidden layer activation function in file: \(hiddenActivationStr)")
            }
            hiddenActivation = function
        }
        
        // Convert output activation function to enum
        var outputActivation: ActivationFunction.Output
        // Check for custom activation function
        if outputActivationStr == "custom" {
            // Note: Here we simply warn the user and set the activation to defualt (sigmoid)
            // The user should reset the activation to the original custom function manually
            print("NeuralNet warning: custom output activation function detected in stored network. Defaulting to sigmoid activation. It is your responsibility to reset the network's output activation to the original function, or it is unlikely to perform correctly.")
            outputActivation = ActivationFunction.Output.sigmoid
        } else {
            guard let function = ActivationFunction.Output.from(outputActivationStr) else {
                throw Error.initialization("Unrecognized output activation function in file: \(outputActivationStr)")
            }
            outputActivation = function
        }
        
        // Recreate Structure object
        let structure = try Structure(nodes: layerNodeCounts,
                                      hiddenActivation: hiddenActivation, outputActivation: outputActivation,
                                      batchSize: batchSize, learningRate: lr, momentum: momentum)
        
        // Initialize neural network
        try self.init(structure: structure, weights: weights)
    }
    
    
    /// Persists the `NeuralNet` to a file at the given URL.
    public func save(to url: URL) throws {
        // Create top-level JSON object
        let json: [String : Any] = [
            NeuralNet.layerNodeCountsKey : layerNodeCounts,
            NeuralNet.momentumKey : momentumFactor,
            NeuralNet.learningRateKey : learningRate,
            NeuralNet.batchSizeKey : batchSize,
            NeuralNet.hiddenActivationKey : hiddenActivation.stringValue(),
            NeuralNet.outputActivationKey : outputActivation.stringValue(),
            NeuralNet.weightsKey : allWeights()
        ]
        
        // Serialize array into JSON data
        let data = try JSONSerialization.data(withJSONObject: json, options: [])
        
        // Write data to file
        try data.write(to: url)
    }
    
}

