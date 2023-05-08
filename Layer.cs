using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer
{
    class Layer
    {
        private List<Neuron> neurons { get; set; }
        // Creates a list of the specified number of neurons with the 
        // specified activation function
        public Layer(int num_neurons, string activation) 
        { 
            this.neurons = new List<Neuron>();

            for (int i = 0; i < num_neurons; i++)
            {
                this.neurons.Add(new Neuron(activation));
            }
        }

        public List<Neuron> Neurons
        {
            get { return this.neurons; }
        }
    }
}
