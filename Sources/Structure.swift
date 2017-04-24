//
//  Structure.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//


public extension NeuralNet {
    
    /// A container for the basic structure of a `NeuralNet`.
    public struct Structure {
        
        /// Possible `Structure` errors.
        public enum Error: Swift.Error {
            case initialize(String)
        }
        
        // MARK: Basic structure properties
        
        /// The total number of layers in the neural network.
        public let numLayers: Int
        
        /// The number of nodes in each layer of the neural network.
        public let layerNodeCounts: [Int]
        
        /// The activation function to apply to hidden nodes during inference.
        public let hiddenActivation: ActivationFunction.Hidden
        
        /// The activation function to apply to the output layer during inference.
        public let outputActivation: ActivationFunction.Output
        
        /// The number of training examples in each batch.
        let batchSize: Int
        
        /// The learning rate to apply during training.
        let learningRate: Float
        
        /// The momentum factor to apply during training.
        let momentumFactor: Float
        
        
        // MARK: Initialization
        
        public init(nodes: [Int], hiddenActivation: ActivationFunction.Hidden, outputActivation: ActivationFunction.Output,
                    batchSize: Int = 1, learningRate: Float = 0.5, momentum: Float = 0.9) throws {
            // Check for valid parameters
            guard nodes.count >= 2 else {
                throw Error.initialize("The network must contain a minimum of two layers (input + output).")
            }
            guard !nodes.contains(where: {$0 < 1}) else {
                throw Error.initialize("Each network layer must contain one or more nodes.")
            }
            
            // Store number of nodes for each layer
            self.layerNodeCounts = nodes
            
            // Store total number of layers in the network
            self.numLayers = nodes.count
            
            // Activation functions
            self.hiddenActivation = hiddenActivation
            self.outputActivation = outputActivation
            
            // Batch size
            self.batchSize = batchSize
            
            // Training parameters
            self.learningRate = learningRate
            self.momentumFactor = momentum
        }
        
    }
    
}
