//
//  Cache.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation


internal extension NeuralNet {
    
    /*
     -------------------------------------------
     
     NOTE: The following cache is allocated once during NeuralNet initializtion, in order to prevent frequent
     heap allocations for temporary variables during the inference and backpropagation cycles.
     
     -------------------------------------------
     */
    
    /// A set of caches used internally by `NeuralNet`.
    struct Cache {
        
        // Weights
        
        /// The total number of weights for each layer in the network.
        /// The count for each layer corresponds to the number of weights connecting the previous layer to the current layer.
        public let layerWeightCounts: [Int]
        
        /// All weights leading into each layer, serialized into single arrays.
        var layerWeights: [[Float]]
        
        /// All weights leading into each layer, from the *previous* round of backpropagation.
        /// This is used for applying momentum during backpropagation.
        var previousLayerWeights: [[Float]]
        
        /// A cache used for calculating new weights during backpropagation.
        var newLayerWeights: [[Float]]
        
        /// A cache used for calculating weight deltas during backpropgation.
        var layerWeightMomentumDeltas: [[Float]]
        
        // Biases
        
        /// The biases applied to each node in each layer.
        /// Note: Here, we treat biases as simple additions to the *current* layer,
        /// rather than extra nodes (and weights) in the *previous* layer.
        /// Thus, the input layer does not have biases but the output layer does.
        var layerBiases: [[Float]]
        
        /// All biases from the *previous* round of backpropgation.
        /// This is used for applying momentum during packpropagation.
        var previousLayerBiases: [[Float]]
        
        /// A cache used for calculating new biases during backpropagation.
        var newLayerBiases: [[Float]]
        
        /// A cache used for calculating bias deltas during backpropagation.
        var layerBiasMomentumDeltas: [[Float]]
        
        // Outputs
        
        /// The cached output from each layer in the network, from the most recent inference.
        var layerOutputs: [[Float]]
        
        /// The derivatives of the activation functions from each layer.
        /// Used during backpropagation for calculating error gradients.
        var layerOutputDerivatives: [[Float]]
        
        // Errors
        
        /// The error gradient for each node in every layer, for every batch set, with respect to the node's input.
        /// Note: the outer array corresponds to the layers in the network, while the inner array holds a
        /// serialized array of the errors for each node, for every batch set.
        var layerErrors: [[Float]]
        
        
        init(structure: NeuralNet.Structure) {
            // Layer outputs cache
            self.layerOutputs = []
            self.layerOutputDerivatives = []
            for layer in 0..<structure.numLayers {
                let matrix = [Float](repeatElement(0, count: structure.batchSize * structure.layerNodeCounts[layer]))
                self.layerOutputs.append(matrix)
                self.layerOutputDerivatives.append(matrix)
            }
            
            // Weights cache
            self.layerWeights = [[]] // Empty set for first layer
            self.previousLayerWeights = [[]]
            self.newLayerWeights = [[]]
            self.layerWeightMomentumDeltas = [[]]
            for layer in 1..<structure.numLayers {
                let matrix = [Float](repeatElement(0, count: structure.layerNodeCounts[layer - 1] * structure.layerNodeCounts[layer]))
                self.layerWeights.append(matrix)
                self.previousLayerWeights.append(matrix)
                self.newLayerWeights.append(matrix)
                self.layerWeightMomentumDeltas.append(matrix)
            }
            
            // Weight counts for for each layer
            var weightCounts = [Int]()
            for (index, layer) in structure.layerNodeCounts.enumerated() {
                if index == 0 {
                    // Input layer has no weights
                    weightCounts.append(0)
                } else {
                    weightCounts.append(structure.layerNodeCounts[index - 1] * layer)
                }
            }
            self.layerWeightCounts = weightCounts
            
            // Biases
            self.layerBiases = [[]] // Empty set for first layer
            self.previousLayerBiases = [[]]
            self.newLayerBiases = [[]]
            self.layerBiasMomentumDeltas = [[]]
            for layer in 1..<structure.numLayers {
                let row = [Float](repeatElement(0, count: structure.layerNodeCounts[layer]))
                self.layerBiases.append(row)
                self.previousLayerBiases.append(row)
                self.newLayerBiases.append(row)
                self.layerBiasMomentumDeltas.append(row)
            }
            
            // Errors cache
            self.layerErrors = []
            for layer in 0..<structure.numLayers {
                self.layerErrors.append([Float](repeatElement(0, count: structure.layerNodeCounts[layer] * structure.batchSize)))
            }
        }
        
    }
    
}
