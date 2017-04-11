//
//  NeuralNet.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation
import Accelerate


/// A 3-layer, feed-forward artificial neural network.
public final class NeuralNet {
    
    // MARK: Errors
    
    /// Possible errors that may be thrown by `NeuralNet`.
    public enum Error: Swift.Error {
        case initialization(String)
        case weights(String)
        case inference(String)
        case train(String)
    }
    
    // MARK: Public properties
    
    /// The basic structure of the neural network (read-only).
    /// This includes the number of input, hidden and output nodes.
    public let structure: Structure
    
    /// The activation function to apply to hidden nodes during inference.
    public var hiddenActivation: ActivationFunction
    /// The activation function to apply to the output layer during inference.
    public var outputActivation: ActivationFunction
    
    /// The cost function to apply during backpropagation.
    public var costFunction: CostFunction
    
    /// The 'learning rate' parameter to apply during backpropagation.
    /// This property may be safely mutated at any time.
    public var learningRate: Float {
        // Must update mfLR whenever this property is changed
        didSet(newRate) {
            cache.mfLR = (1 - momentumFactor) * newRate
        }
    }
    
    /// The 'momentum factor' to apply during backpropagation.
    /// This property may be safely mutated at any time.
    public var momentumFactor: Float {
        // Must update mfLR whenever this property is changed
        didSet(newMomentum) {
            cache.mfLR = (1 - newMomentum) * learningRate
        }
    }
    
    
    // MARK: Private properties and caches
    
    /// A set of pre-computed values and caches used internally for performance optimization.
    fileprivate var cache: Cache
    
    
    // MARK: Initialization
    
    public init(structure: Structure, config: Configuration, weights: [Float]? = nil) throws {
        // Initialize basic properties
        self.structure = structure
        self.hiddenActivation = config.hiddenActivation
        self.outputActivation = config.outputActivation
        self.costFunction = config.cost
        self.learningRate = config.learningRate
        self.momentumFactor = config.momentumFactor
        
        // Initialize computed properties and caches
        self.cache = Cache(structure: structure, config: config)
        
        // Set initial weights, or randomize if none are provided
        if let weights = weights {
            try self.setWeights(weights)
        } else {
            randomizeWeights()
        }
    }
    
}


// MARK: Weights

public extension NeuralNet {
    
    /// Resets the network with the given weights (i.e. from a pre-trained network).
    /// This change may safely be performed at any time.
    ///
    /// - Parameter weights: A serialized array of `Float`, to be used as hidden *and* output weights for the network.
    /// - IMPORTANT: The number of weights must equal `hidden * (inputs + 1) + outputs * (hidden + 1)`, or the weights will be rejected.
    public func setWeights(_ weights: [Float]) throws {
        // Ensure valid number of weights
        guard weights.count == structure.numHiddenWeights + structure.numOutputWeights else {
            throw Error.weights("Invalid number of weights provided: \(weights.count). Expected: \(structure.numHiddenWeights + structure.numOutputWeights).")
        }
        // Reset all weights in the network
        cache.hiddenWeights = Array(weights[0..<structure.numHiddenWeights])
        cache.outputWeights = Array(weights[structure.numHiddenWeights..<weights.count])
    }
    
    /// Returns a serialized array of the network's current weights.
    public func allWeights() -> [Float] {
        return cache.hiddenWeights + cache.outputWeights
    }
    
    /// Randomizes all of the network's weights.
    fileprivate func randomizeWeights() {
        // Randomize hidden weights.
        for i in 0..<structure.numHiddenWeights {
            cache.hiddenWeights[i] = randomHiddenWeight()
        }
        for i in 0..<structure.numOutputWeights {
            cache.outputWeights[i] = randomOutputWeight()
        }
    }
    
    
    // TODO: Generate random weights along a normal distribution, rather than a uniform distribution.
    // Also, these weights are optimized for sigmoid activation.
    // Alternatives should be considered for other activation functions.
    
    /// Generates a random weight for a hidden node, based on the parameters set for the network.
    fileprivate func randomHiddenWeight() -> Float {
        // Note: Hidden weight distribution depends on number of *input* nodes
        return randomWeight(layerInputs: structure.numInputNodes)
    }
    
    /// Generates a random weight for an output node, based on the parameters set for the network.
    fileprivate func randomOutputWeight() -> Float {
        // Note: Output weight distribution depends on number of *hidden* nodes
        return randomWeight(layerInputs: structure.numHiddenNodes)
    }
    
    /// Generates a single random weight.
    ///
    /// - Parameter layerInputs: The number of inputs to the layer in which this weight will be used.
    /// E.g., if this weight will be placed in the hidden layer, `layerInputs` should be the number of input nodes (including bias node).
    /// - Returns: A randomly-generated weight optimized for this layer.
    private func randomWeight(layerInputs: Int) -> Float {
        let range = 1 / sqrt(Float(layerInputs))
        let rangeInt = UInt32(2_000_000 * range)
        
        
        
        let randomFloat = Float(arc4random_uniform(rangeInt)) - Float(rangeInt / 2)
        return randomFloat / 1_000_000
    }
    
}


// MARK: Inference

public extension NeuralNet {
    
    // Note: The inference method is somewhat complex, but testing has shown that
    // keeping the code in-line allows the Swift compiler to make better optimizations.
    // Thus, we achieve improved performance at the cost of slightly less readable code.
    
    /// Inference: propagates the given inputs through the neural network, returning the network's output.
    /// This is the typical method for 'using' a trained neural network.
    /// This is also used during the training process.
    ///
    /// - Parameter inputs: An array of `Float`, each element corresponding to one input node.
    /// - Returns: The network's output after applying the given inputs.
    /// - Throws: An error if an incorrect number of inputs is provided.
    /// - IMPORTANT: The number of inputs provided must exactly match the network's number of inputs (defined in its `Structure`).
    @discardableResult
    public func infer(_ inputs: [Float]) throws -> [Float] {
        // Ensure that the correct number of inputs is given
        guard inputs.count == structure.inputs else {
            throw Error.inference("Invalid number of inputs provided: \(inputs.count). Expected: \(structure.inputs).")
        }
        
        // Cache the inputs
        // Note: A bias node is inserted at index 0, followed by all of the given inputs
        cache.inputCache[0] = 1
        // Note: This loop appears to be the fastest way to make this happen
        for i in 1..<structure.numInputNodes {
            cache.inputCache[i] = inputs[i - 1]
        }
        
        // Calculate the weighted sums for the hidden layer inputs
        vDSP_mmul(cache.hiddenWeights, 1,
                  cache.inputCache, 1,
                  &cache.hiddenOutputCache, 1,
                  vDSP_Length(structure.hidden), 1,
                  vDSP_Length(structure.numInputNodes))
        
        // Apply the activation function to the hidden layer nodes
        for i in (1...structure.hidden).reversed() {
            // Note: Array elements are shifted one index to the right, in order to efficiently insert the bias node at index 0
            cache.hiddenOutputCache[i] = hiddenActivation.activation(cache.hiddenOutputCache[i - 1])
        }
        cache.hiddenOutputCache[0] = 1
        
        // Calculate the weighted sum for the output layer
        vDSP_mmul(cache.outputWeights, 1,
                  cache.hiddenOutputCache, 1,
                  &cache.outputCache, 1,
                  vDSP_Length(structure.outputs), 1,
                  vDSP_Length(structure.numHiddenNodes))
        
        // Apply the activation function to the output layer nodes
        for i in 0..<structure.outputs {
            cache.outputCache[i] = outputActivation.activation(cache.outputCache[i])
        }
        
        // Return the final outputs
        return cache.outputCache
    }
    
}


// MARK: Training

public extension NeuralNet {
    
    // Note: The backpropagation method is somewhat complex, but testing has shown that
    // keeping the code in-line allows the Swift compiler to make better optimizations.
    // Thus, we achieve improved performance at the cost of slightly less readable code.
    
    // Note: Refer to Chapter 3 in the following paper to view the equations applied
    // for this backpropagation algorithm (with modifications to allow for customizable activation/cost functions):
    // https://www.cheshireeng.com/Neuralyst/doc/NUG14x.pdf
    
    
    /// Applies modifications to the neural network by comparing its most recent output to the given `labels`, adjusting the network's weights as needed.
    /// This method should be used for training a neural network manually.
    ///
    /// - Parameter labels: The 'target' desired output for the most recent inference cycle, as an array `[Float]`.
    /// - Throws: An error if an incorrect number of outputs is provided.
    /// - IMPORTANT: The number of labels provided must exactly match the network's number of outputs (defined in its `Structure`).
    public func backpropagate(_ labels: [Float]) throws {
        // Ensure that the correct number of outputs was given
        guard labels.count == structure.outputs else {
            throw Error.train("Invalid number of labels provided: \(labels.count). Expected: \(structure.outputs).")
        }
        
        
        // -----------------------------------------------------
        //
        // NOTE:
        //
        // In the following equations, it is assumed that the network has 3 layers.
        // The subscripts [i], [j], [k] will be used to refer to input, hidden and output layers respectively.
        // In networks with multiple hidden layers, [i] can be assumed to represent whichever layer preceeds the current layer (j),
        // and [k] can be assumed to represent the succeeding layer.
        //
        // -----------------------------------------------------
        
        
        // MARK: Output error gradients --------------------------------------
        
        // Note: Rather than calculating the cost gradient with respect to each output weight,
        // we calculate the gradient with respect to each output node's INPUT, cache the result,
        // and then calculate the gradient for each weight while simultaneously updating the weight.
        // This results in lower memory consumption and fewer calculations.
        
        // Calculate the error gradient with respect to each output node's INPUT.
        // e[k] = outputActivationDerivative(output) * costDerivative(output)
        for (index, output) in cache.outputCache.enumerated() {
            cache.outputErrorGradientsCache[index] = outputActivation.derivative(output) *
                costFunction.derivative(real: output, target: labels[index])
        }
        
        
        // MARK: Hidden error gradients --------------------------------------
        
        // Important: The cost function does not apply to hidden nodes.
        // Instead, the cost derivative component is replaced with the sum of the nodes' error gradients in the succeeding layer
        // (with respect to their inputs, as calculated above) multipled by the weight connecting this node to the following layer.
        
        // Below, w' represents the previous (and still current) weight connecting this node to the following layer.
        // e[j] = hiddenActivationDerivative(hiddenOutput) * Σ(e[k] * w'[j][k])
        
        // Calculate the sums of the output error gradients multiplied by the output weights
        vDSP_mmul(cache.outputErrorGradientsCache, 1,
                  cache.outputWeights, 1,
                  &cache.outputErrorGradientSumsCache, 1,
                  1, vDSP_Length(structure.numHiddenNodes),
                  vDSP_Length(structure.outputs))
        
        // Calculate the error gradient for each hidden node, with respect to the node's INPUT
        for (index ,error) in cache.outputErrorGradientSumsCache.enumerated() {
            cache.hiddenErrorGradientsCache[index] = hiddenActivation.derivative(cache.hiddenOutputCache[index]) * error
        }
        
        
        // MARK: Output weights ----------------------------------------------
        
        // Update output weights
        // Note: In this equation, w' represents the current (old) weight and w'' represents the PREVIOUS weight.
        // In addition, M represents the momentum factor and LR represents the learning rate.
        // X[j] represents the jth input to the node (or, the activated output from the jth hidden node)
        //
        // w[j][k] = w′[j][k] + (1 − M) ∗ LR ∗ e[k] ∗ X[j] + M ∗ (w′[j][k] − w′′[j][k])
        for index in 0..<structure.numOutputWeights {
            // Pre-computed indices: translates the current weight index into the corresponding output error/hidden output indices
            let outputErrorIndex = cache.outputErrorIndices[index]
            let hiddenOutputIndex = cache.hiddenOutputIndices[index]
            
            // Note: mFLR is a pre-computed constant which equals (1 - M) * LR
            cache.newOutputWeights[index] = cache.outputWeights[index] -
                cache.mfLR * cache.outputErrorGradientsCache[outputErrorIndex] * cache.hiddenOutputCache[hiddenOutputIndex] +
                momentumFactor * (cache.outputWeights[index] - cache.previousOutputWeights[index])
        }
        
        // Efficiently copy output weights from current to 'previous' array
        vDSP_mmov(cache.outputWeights,
                  &cache.previousOutputWeights, 1,
                  vDSP_Length(structure.numOutputWeights),
                  1, 1)
        
        // Copy output weights from 'new' to current array
        vDSP_mmov(cache.newOutputWeights,
                  &cache.outputWeights, 1,
                  vDSP_Length(structure.numOutputWeights),
                  1, 1)
        
        
        // MARK: Hidden weights ----------------------------------------------
        
        // Note: This process is almost identical to the process for updating the output weights,
        // since the error gradients have already been calculated independently for each layer.
        
        // Update hidden weights
        // w[i][j] = w′[i][j] + (1 − M) ∗ LR ∗ e[j] ∗ X[i] + M ∗ (w′[i][j] − w′′[i][j])
        for index in 0..<structure.numHiddenWeights {
            // Pre-computed indices: translates the current weight index into the corresponding hidden error/input indices
            let hiddenErrorIndex = cache.hiddenErrorIndices[index]
            let inputIndex = cache.inputIndices[index]
            
            // Note: mfLR is a pre-computed constant which equals (1 - M) * LR
            cache.newHiddenWeights[index] = cache.hiddenWeights[index] -
                // Note: +1 on hiddenErrorIndex to offset for bias 'error', which is ignored
                cache.mfLR * cache.hiddenErrorGradientsCache[hiddenErrorIndex + 1] * cache.inputCache[inputIndex] +
                momentumFactor * (cache.hiddenWeights[index] - cache.previousHiddenWeights[index])
        }
        
        // Copy hidden weights from current to 'previous' array
        vDSP_mmov(cache.hiddenWeights,
                  &cache.previousHiddenWeights, 1,
                  vDSP_Length(structure.numHiddenWeights),
                  1, 1)
        
        // Copy hidden weights from 'new' to current array
        vDSP_mmov(cache.newHiddenWeights,
                  &cache.hiddenWeights, 1,
                  vDSP_Length(structure.numHiddenWeights),
                  1, 1)
    }
    
    
    /// Attempts to train the neural network using the given dataset.
    ///
    /// - Parameters:
    ///   - data: A `Dataset` containing training and validation data, used to train the network.
    ///   - errorThreshold: The minimum acceptable error, as calculated by the network's cost function.
    /// This error will be averaged across the validation set at the end of each training epoch.
    /// Once the error has dropped below `errorThreshold`, training will cease and return.
    /// This value must be determined by the user, as it varies based on the type of data and desired accuracy.
    /// - Returns: A serialized array containing the network's final weights, as calculated during the training process.
    /// - Throws: An error if invalid data is provided. Checks are performed in advance to avoid problems during the training cycle.
    /// - WARNING: `errorThreshold` should be considered carefully. A value too high will produce a poorly-performing network, while a value too low (i.e. too accurate) may be unachievable, resulting in an infinite training process.
    @discardableResult
    public func train(_ data: Dataset, errorThreshold: Float) throws -> [Float] {
        // Ensure valid error threshold
        guard errorThreshold > 0 else {
            throw Error.train("Training error threshold must be greater than zero.")
        }
        
        // -----------------------------
        // TODO: Allow the trainer to exit early or regenerate new weights if it gets stuck in local minima
        // -----------------------------
        
        // Train forever until the desired error threshold is met
        while true {
            // Complete one full training epoch
            for (index, input) in data.trainInputs.enumerated() {
                // Note: We don't care about outputs or error on the training set
                try infer(input)
                try backpropagate(data.trainLabels[index])
            }
            
            // Calculate the total error of the validation set after each training epoch
            var error: Float = 0
            for (index, inputs) in data.validationInputs.enumerated() {
                let outputs = try infer(inputs)
                error += costFunction.cost(real: outputs, target: data.validationLabels[index])
            }
            // Divide error by number of sets to find average error across full validation set
            error /= Float(data.validationInputs.count)
            
            // Escape training loop if the network has met the error threshold
            if error < errorThreshold {
                break
            }
        }
        
        // Return the weights of the newly-trained neural network
        return allWeights()
    }
    
}
