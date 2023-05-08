using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using MathNet.Numerics.Distributions;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.Wpf;
using OxyPlot.Legends;

namespace NeuralNetworkVisualizer
{
    class NeuralNetwork
    {
        public event EventHandler PlotUpdated;
        public List<double> AccuracyHistory { get; private set; } = new List<double>();
        public List<double> LossHistory { get; private set; } = new List<double>();

        public bool StopTraining { get; set; } = false;
        private List<Layer> layers;
        private List<List<double>> data;
        private List<double> labels;

        public List<Layer> Layers { get { return layers; } set { layers = value;  } }
        public List<List<double>> Data { get { return data; } set { data = value; } }
        public List<double> Labels { get { return labels; } set { labels = value; } }

        public NeuralNetwork(int[] numNeurons, string activation)
        {
            this.layers = new List<Layer>();
            this.data = new List<List<double>>();
            this.labels = new List<double>();
            // Hidden layers
            for (int i = 0; i < numNeurons.Length - 1; i++)
            {
                this.layers.Add(new Layer(numNeurons[i], activation));
            }
            // Output layer
            this.layers.Add(new Layer(numNeurons[numNeurons.Length - 1], activation));

            this.InitializeWeightsInputs();
        }

        private void InitializeWeightsInputs()
        {
            // For each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                // For each neuron in the current layer
                foreach (Neuron currNeuron in this.layers[currLayer].Neurons)
                {
                    // Generate random weights for each neuron in the previous layer
                    foreach (Neuron _ in this.layers[currLayer - 1].Neurons)
                    {
                        currNeuron.Weights.Add(2 * new Random().NextDouble() - 1);
                    }
                    // Generate random bias
                    currNeuron.Bias = 2 * new Random().NextDouble() - 1;
                    // Set inputs to the outputs of the previous layer
                    currNeuron.Inputs = this.layers[currLayer - 1].Neurons.Select(neuron => neuron.Output).ToList();
                }
            }
        }

        public void Forward()
        {
            // send inputs into the first layer
            foreach (Neuron neuron in this.layers[1].Neurons)
            {
                neuron.Inputs = this.layers[0].Neurons.Select(neuron => neuron.Output).ToList();
            }

            // Propagate through each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                foreach (Neuron currNeuron in this.layers[currLayer].Neurons)
                {
                    // Take inputs as the outputs of the previous layer
                    currNeuron.Inputs = this.layers[currLayer - 1].Neurons.Select(neuron => neuron.Output).ToList();
                    // Calculated weighted sum (z)
                    currNeuron.CalculateWeightedSum();
                    // Apply activation function to z to find output
                    currNeuron.Output = currNeuron.ActivationFunction();
                }
            }
        }

        public void Backprop(double error)
        {
            // Iterate through each layer backwards excluding the input layer
            for (int currLayer = this.layers.Count - 1; currLayer > 0; currLayer--)
            {
                // Iterate through each neuron in the current layer
                for (int currNeuron = 0; currNeuron < this.layers[currLayer].Neurons.Count; currNeuron++)
                {
                    // error = (target - output)
                    double gradient = -error;
                    // Calculate bias derivative
                    for (int layer = this.layers.Count - 1; layer > currLayer; layer--)
                    {
                        // Multiply by weights and sigmoid derivatives of outputs along the way
                        // Always takes the path of the first neuron in the layer
                        gradient *= this.layers[layer].Neurons[0].ActivationDerivative();
                        // Ensure that the weight connecting currNeuron is multiplied on the final iteration, otherwise 
                        // multiply the weight connecting the first neuron in the layer
                        if (layer == currLayer + 1)
                        {
                            gradient *= this.layers[layer].Neurons[0].Weights[currNeuron];
                        }
                        else
                        {
                            gradient *= this.layers[layer].Neurons[0].Weights[0];
                        }
                    }
                    // Multiply by the sigmoid derivative of currNeuron
                    gradient *= this.layers[currLayer].Neurons[currNeuron].ActivationDerivative();
                    // set the bias derivative
                    this.layers[currLayer].Neurons[currNeuron].BiasDerivative = gradient;
                    // Calculate weight derivatives
                    // For each weight, multiply the gradient by the output of the neuron it connects to
                    for (int currWeight = 0; currWeight < this.layers[currLayer].Neurons[currNeuron].Weights.Count; currWeight++)
                    {
                        this.layers[currLayer].Neurons[currNeuron].WeightsDerivatives.Add(gradient * this.layers[currLayer].Neurons[currNeuron].Inputs[currWeight]);
                    }
                }
            }
        }

        public void UpdateWeights()
        {
            // Learning rate
            double alpha = 0.05;
            // Iterate through each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                // Iterate through each neuron in the current layer
                for (int currNeuron = 0; currNeuron < this.layers[currLayer].Neurons.Count; currNeuron++)
                {
                    // Update bias
                    this.layers[currLayer].Neurons[currNeuron].Bias -= alpha * this.layers[currLayer].Neurons[currNeuron].BiasDerivative;
                    // Update weights
                    for (int currWeight = 0; currWeight < this.layers[currLayer].Neurons[currNeuron].Weights.Count; currWeight++)
                    {
                        this.layers[currLayer].Neurons[currNeuron].Weights[currWeight] -= alpha * this.layers[currLayer].Neurons[currNeuron].WeightsDerivatives[currWeight];
                    }
                }
            }
            // Reset derivatives to prepare for next iteration
            foreach (Layer layer in this.layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.BiasDerivative = 0;
                    neuron.WeightsDerivatives = new List<double>();
                }
            }
        }
        
        public void Train()
        {
            int epochs = 0;
            // Train for a number of epochs
            while (!StopTraining)
            {
                double epochLoss = 0;
                // Iterate through the first 75% of the data for training
                for (int i = 0; i < (int)Math.Round(0.75 * this.data.Count); i++)
                {
                    // Send inputs into the input layer
                    // Allows for more than 2 input features
                    for (int inputNeuron = 0; inputNeuron < this.layers[0].Neurons.Count; inputNeuron++)
                    {
                        this.layers[0].Neurons[inputNeuron].Output = this.data[i][inputNeuron];
                    }
                    // Forward prop
                    this.Forward();
                    // Calculate error (target - output)
                    double error = this.labels[i] - this.layers[this.layers.Count - 1].Neurons[0].Output;
                    epochLoss += Math.Pow(error, 2);
                    // Backpropagation
                    this.Backprop(error);
                    // Update weights
                    this.UpdateWeights();
                }

                epochLoss /= (int)Math.Round(0.75 * this.data.Count);
                LossHistory.Add(epochLoss);

                double accuracy = this.Test();
                AccuracyHistory.Add(accuracy);


                //PlotUpdated?.Invoke(this, EventArgs.Empty);

                
                if (epochs % 5 == 0)
                {
                    PlotUpdated?.Invoke(this, EventArgs.Empty);
                }
                

                // Stop training if accuracy == 1
                //if (this.Test() > 0.99) { break; }

                epochs++;
            }
        }

        public double Test()
        {
            int correct = 0;
            // Iterate through the last 25% of the data for testing
            for (int i = (int)Math.Round(0.75 * this.data.Count); i < this.data.Count; i++)
            {
                // Send inputs into the input layer
                // Allows for more than 2 input features
                for (int inputNeuron = 0; inputNeuron < this.layers[0].Neurons.Count; inputNeuron++)
                {
                    this.layers[0].Neurons[inputNeuron].Output = this.data[i][inputNeuron];
                }
                // Forward prop
                this.Forward();
                // Calculate error (target - output)
                double error = this.labels[i] - this.layers[this.layers.Count - 1].Neurons[0].Output;
                // If the error is less than 0.5, the prediction is correct
                if (Math.Abs(error) < 0.5)
                {
                    correct++;
                }
            }
            // Print accuracy
            return (double)correct / (this.data.Count - (int)Math.Round(0.75 * this.data.Count));
        }

    }
}
