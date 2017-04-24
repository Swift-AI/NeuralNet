//
//  NeuralNet.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation
import Accelerate


/// A fully-connected, feedforward artificial neural network.
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
    
    /// The total number of layers in the neural network (read only).
    public let numLayers: Int
    
    /// The number of nodes in each layer of the neural network (read only).
    public let layerNodeCounts: [Int]
    
    // TODO: Make batch size mutable - will need to adjust cache sizes to account for this change.
    /// The number of items in each minibatch (read only).
    /// Setting this value to 1 is equivalent to performing online training.
    public let batchSize: Int
    
    /// The activation function to apply to all hidden layers during inference.
    public var hiddenActivation: ActivationFunction.Hidden
    
    /// The activation function to apply to the output layer during inference.
    public var outputActivation: ActivationFunction.Output
    
    /// The 'learning rate' parameter to apply during backpropagation.
    /// This property may be safely mutated at any time.
    public var learningRate: Float {
        // Must update adjusted LR whenever this property is changed
        didSet {
            adjustedLearningRate = ((1 - momentumFactor) * learningRate) / Float(batchSize)
        }
    }
    
    /// The 'momentum factor' to apply during backpropagation.
    /// This property may be safely mutated at any time.
    public var momentumFactor: Float {
        // Must update adjusted LR whenever this property is changed
        didSet {
            adjustedLearningRate = ((1 - momentumFactor) * learningRate) / Float(batchSize)
        }
    }
    
    
    // MARK: Private properties and caches
    
    /// A set of caches used internally for performance optimization.
    fileprivate var cache: Cache
    
    /// The learning rate, adjusted for momentum and batch size.
    /// This is used to scale down the learning rate appropriately as momentum is increased, to avoid rapid gradient descent.
    /// ((1 - momentumFactor) * learningRate) / batchSize
    fileprivate var adjustedLearningRate: Float
    
    
    // MARK: Initialization
    
    public init(structure: Structure, weights: [[Float]]? = nil) throws {
        // Initialize basic properties
        self.numLayers = structure.numLayers
        self.layerNodeCounts = structure.layerNodeCounts
        self.batchSize = structure.batchSize
        self.hiddenActivation = structure.hiddenActivation
        self.outputActivation = structure.outputActivation
        self.learningRate = structure.learningRate
        self.momentumFactor = structure.momentumFactor
        self.adjustedLearningRate = ((1 - structure.momentumFactor) * structure.learningRate) / Float(structure.batchSize)
        
        // Initialize computed properties and caches
        self.cache = Cache(structure: structure)
        
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
    /// - Parameter weights: A 2D array of weights corresponding to each layer in the network.
    public func setWeights(_ weights: [[Float]]) throws {
        // TODO: ensure valid number of weights
        
        // Reset all weights in the network
        cache.layerWeights = weights
    }
    
    /// Returns an array of the network's current weights for each layer.
    public func allWeights() -> [[Float]] {
        return cache.layerWeights
    }
    
    /// Randomizes all of the network's weights.
    fileprivate func randomizeWeights() {
        // Randomize weights for each layer independently
        // Note: Output weights and all biases do not need initialization; they remain 0 until training begins
        for layer in 1..<(numLayers - 1) {
            // Randomize weights
            for weight in 0..<cache.layerWeightCounts[layer] {
                cache.layerWeights[layer][weight] = randomWeight(fanIn: layerNodeCounts[layer - 1], fanOut: layerNodeCounts[layer + 1])
            }
        }
    }
    
    /// Generates a single random weight.
    ///
    /// - Parameter fanIn: The number of inputs to the node in which this weight will be used.
    /// - Parameter fanOut: The number of outputs from the node in which this weight will be used.
    /// - Returns: A randomly-generated weight optimized for this node and the network's hidden activation function.
    private func randomWeight(fanIn: Int, fanOut: Int) -> Float {
        // sqrt(6 / (fanOut + fanIn))
        let range = sqrt(6 / Float(fanIn + fanOut))
        let rangeInt = UInt32(2_000_000_000 * range)
        let randomFloat = (Float(arc4random_uniform(rangeInt)) - Float(rangeInt / 2)) / 1_000_000_000
        
        switch hiddenActivation {
        case .sigmoid:
            // 4 * sqrt(6 / (fanOut + fanIn))
            return randomFloat * 4
        case .hyperbolicTangent:
            // sqrt(6 / (fanOut + fanIn))
            return randomFloat
        default:
            return randomFloat
        }
    }
    
}


// MARK: Inference

public extension NeuralNet {
    
    
    /// Minibatch inference: propagates the given batch of inputs through the neural network, returning the network's output.
    ///
    /// - Parameter inputs: A batch of inputs sets. The number of input sets must exactly match net network's `batchSize`.
    /// - Returns: The full batch of outputs corresponding to the provided inputs.
    /// - Throws: An error if the number of batches or inputs per set are incorrect.
    @discardableResult
    public func infer(_ inputs: [[Float]]) throws -> [[Float]] {
        // Make sure the correct number of batches was provided
        guard inputs.count == batchSize else {
            throw Error.inference("Incorrect number of input sets provided: \(inputs.count). Expected: \(batchSize). The number of input sets must exactly match the network's batch size.")
        }
        
        // Serialize full batch of inputs into a single array
        // Note: This is *much* faster than using `inputs.reduce([], +)`
        var input: [Float] = [Float](repeatElement(0, count: batchSize * layerNodeCounts[0]))
        for a in 0..<batchSize {
            for i in 0..<layerNodeCounts[0] {
                let idx = a * layerNodeCounts[0] + i
                input[idx] = inputs[a][i]
            }
        }
        
        // Perform inference
        let outputs = try infer(input)
        
        // Split result (full batch) into individual rows
        let outputLength = layerNodeCounts[numLayers - 1]
        let count = outputLength * batchSize
        return stride(from: 0, to: count, by: outputLength).map {
            Array(outputs[$0..<min($0 + outputLength, count)])
        }
    }
    
    /// Inference: propagates the given inputs through the neural network, returning the network's output.
    /// This method should be used for performing inference on a single set of inputs,
    /// or for minibatch inference where all inputs have been serialized into a single array.
    /// Regardless, the number of sets must exactly match the `batchSize` defined in the network's `Structure`.
    ///
    /// - Parameter inputs: A single set of inputs, or a minibatch serialized into a single array.
    /// - Returns: The network's output after applying the given inputs.
    ///            The number of output sets will equal the `batchSize` defined in the network's `Structure`.
    /// - Throws: An error if an incorrect number of inputs is provided.
    /// - IMPORTANT: The number of inputs provided must exactly match the network's number of inputs (defined in its `Structure`).
    @discardableResult
    public func infer(_ inputs: [Float]) throws -> [Float] {
        // Ensure that the correct number of inputs is given
        guard inputs.count == layerNodeCounts[0] * batchSize else {
            throw Error.inference("Incorrect number of inputs provided: \(inputs.count). Expected: \(layerNodeCounts[0] * batchSize). The number of total inputs must exactly match the network's input size times its batch size.")
        }
        
        // Cache the inputs
        cache.layerOutputs[0] = inputs
        
        // Loop through each layer in the network and calculate the layer's output.
        // Note: We don't apply any weights or activation to the first (input) layer.
        for layer in 1..<numLayers {
            // Calculate the weighted, summed input for this layer
            // (Stored temporarily in its output cache)
            vDSP_mmul(cache.layerOutputs[layer - 1], 1,
                      cache.layerWeights[layer], 1,
                      &cache.layerOutputs[layer][0], 1,
                      vDSP_Length(batchSize),
                      vDSP_Length(layerNodeCounts[layer]),
                      vDSP_Length(layerNodeCounts[layer - 1]))
            
            // Add each node's bias to its own input, for every item in the batch
            // TODO: Figure out how to vectorize this operation
            // Here, we add the layer's bias (row vector) to each row in the input matrix.
            for b in 0..<batchSize {
                for n in 0..<layerNodeCounts[layer] {
                    let idx = b * layerNodeCounts[layer] + n
                    cache.layerOutputs[layer][idx] += cache.layerBiases[layer][n]
                }
            }
            
            // Apply the activation function to each node in this layer
            if layer == numLayers - 1 {
                // Output layer; special activation
                outputActivation.computeActivation(cache.layerOutputs[layer], result: &cache.layerOutputs[layer],
                                                   rows: batchSize, cols: layerNodeCounts[layer])
            } else {
                // Hidden layer
                hiddenActivation.computeActivation(cache.layerOutputs[layer], result: &cache.layerOutputs[layer],
                                                   rows: batchSize, cols: layerNodeCounts[layer])
            }
        }
        
        // Return the final layer's output
        return cache.layerOutputs[numLayers - 1]
    }
    
}


// MARK: Training

public extension NeuralNet {
    
    // Note: The following papers contain useful information on the backpropagation algorithm applied here.
    // http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5
    // https://www.cheshireeng.com/Neuralyst/doc/NUG14x.pdf  (Chapter 3)
    
    
    /// Minibatch backpropagation: Trains the neural network by updating its weights using the given label sets.
    ///
    /// - Parameter labels: An array of label sets corresponding to the most recent minibatch applied for inference.
    /// - IMPORTANT: The labels must be given in the same order as the corresponding inputs that were provided during inference.
    /// - Throws: An error if a data inconsistency is detected.
    public func backpropagate(_ labels: [[Float]]) throws {
        // Ensure the correct number of sets is given
        guard labels.count == batchSize else {
            throw Error.train("Incorrect number of label sets provided: \(labels.count). Expected: \(batchSize). The number of sets in a minibatch must exactly match the network's batch size.")
        }
        
        // Serialize full batch of labels into a single array.
        // Note: This loop is *much* faster than using `labels.reduce([], +)`
        var serializedLabels: [Float] = [Float](repeatElement(0, count: batchSize * layerNodeCounts[numLayers - 1]))
        for a in 0..<batchSize {
            for i in 0..<layerNodeCounts[numLayers - 1] {
                let idx = a * layerNodeCounts[numLayers - 1] + i
                serializedLabels[idx] = labels[a][i]
            }
        }
        
        // Perform backpropagation
        try backpropagate(serializedLabels)
    }
    
    /// Backpropagation: Trains the neural network by updating its weights using the given labels.
    /// This method may be used for single sets of labels, or for minibatch training where the full batch of labels
    /// has been serialized into a single array.
    ///
    /// - Parameter labels: A single set of labels corresponding to the most recent inference cycle,
    ///                     or a full minibatch of labels serialized into a single array.
    /// - Throws: An error if a data inconsistency is detected.
    public func backpropagate(_ labels: [Float]) throws {
        // Ensure that the correct number of labels was given
        guard labels.count == layerNodeCounts[numLayers - 1] * batchSize else {
            throw Error.train("Incorrect number of labels provided: \(labels.count). Expected: \(layerNodeCounts[numLayers - 1] * batchSize). The total number of labels must exactly match the network's output size times its batch size.")
        }
        
        // Step 1 -----------------------------------
        // Calculate all error gradients (except input layer)
        
        for layer in (1..<numLayers).reversed() {
            if layer == numLayers - 1 {
                // Output layer
                
                // Ask activation function to calculate error gradient for output layer
                outputActivation.calculateErrorGradient(real: cache.layerOutputs[layer],
                                                        target: labels,
                                                        result: &cache.layerErrors[layer],
                                                        rows: batchSize, cols: layerNodeCounts[layer])
                
                // Transpose the error gradient matrix.
                vDSP_mtrans(cache.layerErrors[layer], 1,
                            &cache.layerErrors[layer][0], 1, // Swift compiler bug workaround - we should be able to use cache.layerErrors[layer]
                    vDSP_Length(layerNodeCounts[layer]),
                    vDSP_Length(batchSize))
                
            } else {
                // Hidden layer
                
                // Multiply the weights from the succeeding layer by the error gradient from the succeeding layer.
                // This results in the error gradient with respect to the current layer's OUTPUT.
                // Store this temporarily in the error cache for this layer.
                vDSP_mmul(cache.layerWeights[layer + 1], 1,
                          cache.layerErrors[layer + 1], 1,
                          &cache.layerErrors[layer][0], 1,
                          vDSP_Length(layerNodeCounts[layer]),
                          vDSP_Length(batchSize),
                          vDSP_Length(layerNodeCounts[layer + 1]))
                
                // Calculate the derivative of the activation function for each output in this layer.
                hiddenActivation.calculateDerivative(cache.layerOutputs[layer], result: &cache.layerOutputDerivatives[layer],
                                                     rows: batchSize, cols: layerNodeCounts[layer])
                
                // Transpose the derivative
                vDSP_mtrans(cache.layerOutputDerivatives[layer], 1,
                            &cache.layerOutputDerivatives[layer][0], 1,
                            vDSP_Length(layerNodeCounts[layer]),
                            vDSP_Length(batchSize))
                
                // Multiply the above error gradient by the activation derivative.
                // This results in the error gradient with respect to this layer's INPUT.
                // Store this result in the errors cache.
                vDSP_vmul(cache.layerErrors[layer], 1,
                          cache.layerOutputDerivatives[layer], 1,
                          &cache.layerErrors[layer][0], 1,
                          vDSP_Length(layerNodeCounts[layer] * batchSize))
            }
        }
        
        
        // Step 2 -----------------------------------
        // Perform all weight and bias updates
        // Formula used for weight and bias updates (with learning rate and momentum):
        // wij = w′ij - (adjustedLearningRate ∗ E + M ∗ (w′ij − w′′ij))
        
        for layer in 1..<numLayers {
            
            // Multiply this layer's error matrix by the preceeding layer's outputs.
            // This results in the error gradient with respect to each weight in this layer.
            vDSP_mmul(cache.layerErrors[layer], 1,
                      cache.layerOutputs[layer - 1], 1,
                      &cache.newLayerWeights[layer][0], 1,
                      vDSP_Length(layerNodeCounts[layer]),
                      vDSP_Length(layerNodeCounts[layer - 1]),
                      vDSP_Length(batchSize))
            
            // Transpose delta matrix to get back to the same size/orientation as the weight matrix.
            vDSP_mtrans(cache.newLayerWeights[layer], 1,
                        &cache.newLayerWeights[layer][0], 1,
                        vDSP_Length(layerNodeCounts[layer - 1]),
                        vDSP_Length(layerNodeCounts[layer]))
            
            // Multiply adjusted learning rate by every element in the delta matrix.
            // adjustedLearningRate * E
            vDSP_vsmul(cache.newLayerWeights[layer], 1,
                       &adjustedLearningRate,
                       &cache.newLayerWeights[layer][0], 1,
                       vDSP_Length(cache.layerWeightCounts[layer]))
            
            // Calculate difference between current and previous weights, multiplied by the momentum factor.
            // M * (w′ij − w′′ij)
            vDSP_vsbsm(cache.layerWeights[layer], 1,
                       cache.previousLayerWeights[layer], 1,
                       &momentumFactor,
                       &cache.layerWeightMomentumDeltas[layer][0], 1,
                       vDSP_Length(cache.layerWeightCounts[layer]))
            
            // Sum the two weight delta components.
            // adjustedLearningRate * E + M ∗ (w′ij − w′′ij)
            vDSP_vadd(cache.newLayerWeights[layer], 1,
                      cache.layerWeightMomentumDeltas[layer], 1,
                      &cache.newLayerWeights[layer][0], 1,
                      vDSP_Length(cache.layerWeightCounts[layer]))
            
            // Store current weights in "previous" weights array for later use.
            cache.previousLayerWeights[layer] = cache.layerWeights[layer]
            
            // Subtract the delta matrix from the weight matrix for this layer.
            vDSP_vsub(cache.newLayerWeights[layer], 1,
                      cache.layerWeights[layer], 1,
                      &cache.newLayerWeights[layer][0], 1,
                      vDSP_Length(cache.layerWeightCounts[layer]))
            
            // Store new weights back into cache
            cache.layerWeights[layer] = cache.newLayerWeights[layer]
            
            
            // Bias updates
            
            // Calculate gradients for biases in this layer.
            // Note: The gradient for each bias is simply the gradient of its corresponding node.
            // Here, we just sum the error gradients for each node across every item in the batch.
            sumRows(of: cache.layerErrors[layer],
                    into: &cache.newLayerBiases[layer],
                    rows: layerNodeCounts[layer], columns: batchSize)
            
            // Multiply adjusted learning rate by every element in the bias delta vector.
            // adjustedLearningRate * E
            vDSP_vsmul(cache.newLayerBiases[layer], 1,
                       &adjustedLearningRate,
                       &cache.newLayerBiases[layer][0], 1,
                       vDSP_Length(layerNodeCounts[layer]))
            
            // Calculate difference between current and previous biases, multiplied by the momentum factor.
            // M * (w′ij − w′′ij)
            vDSP_vsbsm(cache.layerBiases[layer], 1,
                       cache.previousLayerBiases[layer], 1,
                       &momentumFactor,
                       &cache.layerBiasMomentumDeltas[layer][0], 1,
                       vDSP_Length(layerNodeCounts[layer]))
            
            // Sum the two bias delta components.
            // adjustedLearningRate * E + M ∗ (w′ij − w′′ij)
            vDSP_vadd(cache.newLayerBiases[layer], 1,
                      cache.layerBiasMomentumDeltas[layer], 1,
                      &cache.newLayerBiases[layer][0], 1,
                      vDSP_Length(layerNodeCounts[layer]))
            
            // Store current biases in "previous" biases array.
            cache.previousLayerBiases[layer] = cache.layerBiases[layer]
            
            // Subtract the bias delta vector from the bias vector for this layer.
            vDSP_vsub(cache.newLayerBiases[layer], 1,
                      cache.layerBiases[layer], 1,
                      &cache.newLayerBiases[layer][0], 1,
                      vDSP_Length(layerNodeCounts[layer]))
            
            // Store new biases back into cache
            cache.layerBiases[layer] = cache.newLayerBiases[layer]
        }
    }
    
    /// Attempts to train the neural network using the given dataset.
    ///
    /// - Parameters:
    ///   - data: A `Dataset` containing training and validation data.
    ///   - maxEpochs: The maximum number of training epochs to complete before returning, *regardless* of error.
    ///                This may be useful for escaping when stuck in local minima.
    ///   - errorThreshold: The minimum acceptable error, as calculated by the provided error function.
    ///                 Once the error has dropped below `errorThreshold`, training will cease and return.
    ///                 This value must be determined by the user, as it varies based on the type of data and desired accuracy.
    ///   - errorFunction: An `ErrorFunction` to use for computing error on the validation set.
    ///                 This error will be computed on the entire validation set after each training epoch.
    ///   - epochCallback: An optional block to execute at the end of each training epoch.
    ///                 If implemented, the handler will be passed the current epoch and validation error.
    ///                 The handler must return a `Bool` indicating whether training should continue.
    ///                 If `false` is returned, the training routine will exit immediately and return.
    ///                 The user may implement this block to monitor the training progress, tune network parameters,
    ///                 or perform any other logic desired.
    /// - Returns: The total number of training epochs performed, and the final validation error.
    /// - Throws: An error if invalid data is provided. Checks are performed in advance to avoid problems during the training cycle.
    public func train(_ data: Dataset, maxEpochs: Int,
                      errorThreshold: Float, errorFunction: ErrorFunction,
                      epochCallback: ((_ epoch: Int, _ error: Float) -> Bool)?) throws -> (epochs: Int, error: Float) {
        // Ensure valid error threshold
        guard errorThreshold > 0 else {
            throw Error.train("Training error threshold must be greater than zero.")
        }
        
        // Store length of each individual output set
        let outputLength = layerNodeCounts[numLayers - 1]
        
        // Store total number of batches in the set
        let numBatches = data.validationLabels.count
        
        // Store length of each full batch output
        let batchOutputLength = outputLength * batchSize
        
        // Serialize all validation labels into a single array
        let validationLabels = data.validationLabels.reduce([], +)
        
        // Reserve space for serializing all validation set outputs
        var validationOutputs = [Float](repeatElement(0, count: outputLength * batchSize * numBatches))
        
        // Train until the desired error threshold is met or the max number of epochs has been executed
        var epochs = 1
        while true {
            // Complete one full training epoch
            for (batchinputs, batchLabels) in zip(data.trainInputs, data.trainLabels) {
                try infer(batchinputs)
                try backpropagate(batchLabels)
            }
            
            // Perform inference on the full validation set
            for (batchIndex, batchInputs) in data.validationInputs.enumerated() {
                let outputs = try infer(batchInputs)
                // Add all outputs to the serialized array
                for i in 0..<batchOutputLength {
                    let idx = batchIndex * batchOutputLength + i
                    validationOutputs[idx] = outputs[i]
                }
            }
            
            // Calculate error on the whole validation set
            let error = errorFunction.computeError(real: validationOutputs, target: validationLabels,
                                                   rows: batchSize * numBatches, cols: outputLength)
            
            // Notify callback of a newly-completed epoch; halt training if requested
            if let toContinue = epochCallback?(epochs, error), toContinue == false {
                return (epochs, error)
            }
            
            // Escape training loop if the network has met the error threshold or max number of epochs
            if error < errorThreshold || epochs == maxEpochs {
                return (epochs, error)
            }
            
            // Increment epoch counter
            epochs += 1
        }
    }
    
}


// MARK: Utilities

fileprivate extension NeuralNet {
    
    /// Sums the rows of the given matrix into a single column vector.
    ///
    /// - Parameters:
    ///   - matrix: The matrix.
    ///   - destination: The vector into which the row sums should be stored.
    ///   - rows: The number of rows in the matrix.
    ///   - columns: The number of columns in the matrix.
    func sumRows(of matrix: [Float], into destination: inout [Float], rows: Int, columns: Int) {
        matrix.withUnsafeBufferPointer { matrix in
            destination.withUnsafeMutableBufferPointer { destination in
                for row in 0..<rows {
                    vDSP_sve(matrix.baseAddress! + row * columns, 1,
                             destination.baseAddress! + row, vDSP_Length(columns))
                }
            }
        }
    }
    
}

