//
//  Dataset.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/4/17.
//
//

import Foundation


public extension NeuralNet {
    
    /// A complete dataset for training a neural network, including training sets and validation sets.
    public struct Dataset {
        
        /// Errors that may be thrown by `Dataset`.
        public enum Error: Swift.Error {
            case data(String)
        }
        
        
        // Training
        
        /// The full set of inputs for the training set.
        /// This is an array of batches, where the size of each batch equals the network's input size * batchSize.
        /// The number of batches should also match the number of batches given for `trainLabels`.
        public let trainInputs: [[Float]]
        /// The full set of labels for the training set.
        /// This is an array of batches, where the size of each batch equals the network's output size * batchSize.
        /// The number of batches should also match the number of batches given for `trainInputs`.
        public let trainLabels: [[Float]]
        
        
        // Validation
        
        /// The full set of inputs for the validation set.
        /// This is an array of batches, where the size of each batch equals the network's input size * batchSize.
        /// The number of batches should also match the number of batches given for `validationLabels`.
        public let validationInputs: [[Float]]
        /// The full set of labels for the validation set.
        /// This is an array of batches, where the size of each batch equals the network's output size * batchSize.
        /// The number of batches should also match the number of batches given for `validationInputs`.
        public let validationLabels: [[Float]]
        
        
        // Initialization
        
        /// Initialization from data batches.
        /// Each of these parameters should be an array of batches, where each batch is an array of
        /// input sets, and an input set is an array of individual network inputs.
        public init(trainInputs: [[[Float]]], trainLabels: [[[Float]]],
                    validationInputs: [[[Float]]], validationLabels: [[[Float]]]) throws {
            // Ensure that no empty sets were given
            guard !trainInputs.isEmpty,
                !trainInputs[0].isEmpty,
                !trainInputs[0][0].isEmpty else {
                    throw Error.data("The training data contains one or more empty sets.")
            }
            guard !trainLabels.isEmpty,
                !trainLabels[0].isEmpty,
                !trainLabels[0][0].isEmpty else {
                    throw Error.data("The training labels contains one or more empty sets.")
            }
            guard !validationInputs.isEmpty,
                !validationInputs[0].isEmpty,
                !validationInputs[0][0].isEmpty else {
                    throw Error.data("The validation data contains one or more empty sets.")
            }
            guard !validationLabels.isEmpty,
                !validationLabels[0].isEmpty,
                !validationLabels[0][0].isEmpty else {
                    throw Error.data("The training labels contains one or more empty sets.")
            }
            
            // Infer network dimensions from batch sizes
            let numTrainingBatches = trainInputs.count
            let batchSize = trainInputs[0].count
            let inputSize = trainInputs[0][0].count
            let numValidationBatches = validationInputs.count
            let outputSize = trainLabels[0][0].count
            
            // Serialize training data
            
            // Note: The following loops execute *much* faster than Swift's `reduce()` method.
            
            let trainSet = [Float](repeatElement(0, count: batchSize * inputSize))
            var trainBatches: [[Float]] = [[Float]](repeatElement(trainSet, count: numTrainingBatches))
            for batch in 0..<numTrainingBatches {
                for row in 0..<batchSize {
                    for col in 0..<inputSize {
                        let idx = row * inputSize + col
                        trainBatches[batch][idx] = trainInputs[batch][row][col]
                    }
                }
            }
            
            let trainLabelSet = [Float](repeatElement(0, count: batchSize * outputSize))
            var trainLabelBatches: [[Float]] = [[Float]](repeatElement(trainLabelSet, count: numTrainingBatches))
            for batch in 0..<numTrainingBatches {
                for row in 0..<batchSize {
                    for col in 0..<outputSize {
                        let idx = row * outputSize + col
                        trainLabelBatches[batch][idx] = trainLabels[batch][row][col]
                    }
                }
            }
            
            
            // Serialize validation data
            
            let validationSet = [Float](repeatElement(0, count: batchSize * inputSize))
            var validationBatches: [[Float]] = [[Float]](repeatElement(validationSet, count: numValidationBatches))
            for batch in 0..<numValidationBatches {
                for row in 0..<batchSize {
                    for col in 0..<inputSize {
                        let idx = row * inputSize + col
                        validationBatches[batch][idx] = validationInputs[batch][row][col]
                    }
                }
            }
            
            let validationLabelSet = [Float](repeatElement(0, count: batchSize * outputSize))
            var validationLabelBatches: [[Float]] = [[Float]](repeatElement(validationLabelSet, count: numValidationBatches))
            for batch in 0..<numValidationBatches {
                for row in 0..<batchSize {
                    for col in 0..<outputSize {
                        let idx = row * outputSize + col
                        validationLabelBatches[batch][idx] = validationLabels[batch][row][col]
                    }
                }
            }
            
            
            // Initialize
            try self.init(trainInputs: trainBatches, trainLabels: trainLabelBatches,
                          validationInputs: validationBatches, validationLabels: validationLabelBatches)
        }
        
        /// Initialization from serialized batches.
        /// Each of these parameters should be an array of batches, where each batch is an array
        /// of *serialized* input sets.
        public init(trainInputs: [[Float]], trainLabels: [[Float]],
                    validationInputs: [[Float]], validationLabels: [[Float]]) throws {
            // Ensure that an equal number of sets were provided for inputs and their corresponding labels
            guard trainInputs.count == trainLabels.count && validationInputs.count == validationLabels.count else {
                throw Error.data("The number of input sets provided for training/validation must equal the number of label sets provided.")
            }
            
            // Initialize properties
            self.trainInputs = trainInputs
            self.trainLabels = trainLabels
            self.validationInputs = validationInputs
            self.validationLabels = validationLabels
        }
        
    }
    
}
