using OxyPlot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNetworkVisualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        // nn: NeuralNetwork used to train/test/predict
        private NeuralNetwork nn;
        // nnPlot: NNPlot used to plot the loss/accuracy and the decision boundary
        private NNPlot nnPlot;
        // nnDataLoader: NNDataLoader used to load the data into nn
        private NNDataLoader nnDataLoader;
        // numNeurons: int[] used to store the number of neurons in each layer for initialization of nn
        private int[] numNeurons;
        public MainWindow()
        {
            InitializeComponent();
            // Default neural network with 2 hidden layers with 3 neurons/2 neurons & the sigmoid activation function
            numNeurons = new int[] { 2, 3, 2, 1 };
            nn = new NeuralNetwork(numNeurons, "sigmoid");
            // Display the configuration of the neural network
            CreateNeurons(numNeurons);
            CreateWeightLines();
        }

        /*
         * Creates the circles used to represent the neurons in the neural network
         * visualization.
         */
        private void CreateNeurons(int[] neuronCounts)
        {
            // Scales the size of the circles based on the number of neurons in the largest layer
            // this way regardless of the size of the network, the visualization will scale properly
            int largestLayer = neuronCounts.Max();
            double circleSize = Math.Min(Height, Width) / (largestLayer * 3);

            var layers = new List<object>();

            // Adds layers to be used as the ItemsSource for the CirclesContainer
            // to display each layer vertically column by column in the UI
            for (int neuronCount = 0; neuronCount < neuronCounts.Length; neuronCount++)
            {
                var layer = new List<object>();
                // Add labels X1 and X2 to the input layer
                if (neuronCount == 0)
                {
                    layer.Add(new { CircleSize = circleSize, Content = "X1" });
                    layer.Add(new { CircleSize = circleSize, Content = "X2" });
                }
                else if (neuronCount == neuronCounts.Length - 1)
                {
                    layer.Add(new { CircleSize = circleSize, Content = "O" });
                }
                else
                {
                    for (int i = 0; i < neuronCounts[neuronCount]; i++)
                    {
                        layer.Add(new { CircleSize = circleSize });
                    }
                }

                layers.Add(new { RowCount = neuronCounts[neuronCount], ColumnCount = 1, Neurons = layer });
            }

            CirclesContainer.ItemsSource = layers;
        }

        /*
         * Creates the lines used to represent the weights in the neural network visualization.
         */
        private void CreateWeightLines()
        {
            // Set color for the brush
            var converter = new BrushConverter();
            var brush = (Brush)converter.ConvertFromString("#66c7bc");
            // Clear the canvas and create the lines for the weights
            NetworkCanvas.Children.Clear();

            for (int layerIndex = 0; layerIndex < numNeurons.Length - 1; layerIndex++)
            {
                int currentLayerNeuronCount = numNeurons[layerIndex];
                int nextLayerNeuronCount = numNeurons[layerIndex + 1];

                for (int currentNeuronIndex = 0; currentNeuronIndex < currentLayerNeuronCount; currentNeuronIndex++)
                {
                    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; nextNeuronIndex++)
                    {
                        Line weightLine = new Line
                        {
                            Stroke = brush,
                            StrokeThickness = 1,
                            X1 = (double)85 + (layerIndex) * (CirclesContainer.ActualWidth - 110) / (numNeurons.Length - 1),
                            X2 = (double)85 + (layerIndex + 1) * (CirclesContainer.ActualWidth - 110) / (numNeurons.Length - 1),
                            Y1 = (double)(currentNeuronIndex + 1) * (CirclesContainer.ActualHeight + 50) / (currentLayerNeuronCount + 1),
                            Y2 = (double)(nextNeuronIndex + 1) * (CirclesContainer.ActualHeight + 50) / (nextLayerNeuronCount + 1)
                        };
                        NetworkCanvas.Children.Add(weightLine);
                    }
                }
            }
        }
        
        // Event handler to redraw weight lines when the window is resized
        private void MainWindow_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            CreateWeightLines();
        }

        // Event handler to draw weight lines when the window is loaded
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            CreateWeightLines();
        }

        // Updates accuracy/loss and decision boundary plots when the function is called
        private void Nn_PlotUpdated(object? sender, EventArgs e, NeuralNetwork nn)
        {
            // Update the plot view
            Dispatcher.Invoke(() =>
            {
                plotView.Model = nnPlot.PlotData();
                plotView.InvalidatePlot(true);

                accuracyPlotView.Model = nnPlot.LossAccuracyPlotData();
                accuracyPlotView.InvalidatePlot(true);
            });
        }

        // Trains the neural network asynchronously so it can be interrupted
        // and restarted when the play button is clicked
        private async Task TrainAsync()
        {
            await Task.Run(() => nn.Train());

            // Update the UI on the main thread after training has finished
            Dispatcher.Invoke(() =>
            {
                plotView.Model = nnPlot.PlotData();
            });
        }

        /*
         * Event handler for the play button. 
         * First initializes the neural network based on user input
         * Then randomly loads data into the network and begins training asynchronously
         */
        private async void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            // Stop the previous training, if any
            if (nn != null)
            {
                nn.StopTraining = true;
            }

            // Initialize the plots
            plotView.Model = new PlotModel();
            plotView.InvalidatePlot(true);

            accuracyPlotView.Model = new PlotModel();
            accuracyPlotView.InvalidatePlot(true);

            // Read hidden layers input into array
            string[] splitText = HiddenLayers_TextBox.Text.Split(new string[] {", ", "," }, StringSplitOptions.RemoveEmptyEntries);
            int numLayers = splitText.Length + 2;
            // Initialize numNeurons to # of hidden layers + 2 for input/output layers
            numNeurons = new int[numLayers];

            for (int i = 1; i < numLayers-1; i++)
            {
                try
                {
                    numNeurons[i] = int.Parse(splitText[i - 1]);
                }
                catch 
                {
                    MessageBox.Show("Hidden layers were formatted incorrectly.\nAn example of proper formatting: 5, 3, 2", "Hidden Layers Input Error");
                    return;
                }
            }

            // Set input/output layers
            numNeurons[0] = 2;
            numNeurons[numLayers-1] = 1;

            // Set activation function based on radio button selection
            string activation;
            if (SigmoidRadioButton.IsChecked == true)
            {
                activation = "sigmoid";
            }
            else
            {
                activation = "relu";
            }

            // Initialize new neural network, data loader, and plotter
            nn = new NeuralNetwork(numNeurons, activation);
            nnPlot = new NNPlot(nn);
            nnDataLoader = new NNDataLoader(nn);

            // Display neural network configuration in the UI
            CreateNeurons(numNeurons);
            CreateWeightLines();

            // Subscribe to plot update event
            nn.PlotUpdated += (sender, e) => Nn_PlotUpdated(sender, e, nn);

            // Randomly load data into the network
            Random random = new Random();
            int data = random.Next(0, 4);

            if (data == 0) 
            {
                nnDataLoader.LoadMoonData(noise: 0.4);
            }
            else if (data == 1)
            {
                nnDataLoader.LoadXORData();
            }
            else if (data == 2)
            {
                nnDataLoader.LoadCircleData();
            }
            else
            {
                nnDataLoader.LoadSpiralData();
            }
            
            // Allow the network to be trained
            nn.StopTraining = false;
            await TrainAsync();
        }
    }
}
