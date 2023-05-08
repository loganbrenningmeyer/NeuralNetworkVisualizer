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
        private NeuralNetwork nn;
        private int[] numNeurons;
        public MainWindow()
        {

            InitializeComponent();
            numNeurons = new int[] { 2, 3, 2, 1 };
            nn = new NeuralNetwork(numNeurons, "sigmoid");

            CreateNeurons(numNeurons);
            CreateWeightLines();
        }
        private void CreateNeurons(int[] neuronCounts)
        {
            int largestLayer = neuronCounts.Max();
            double circleSize = Math.Min(Height, Width) / (largestLayer * 3);

            var layers = new List<object>();

            bool first = true;

            foreach (var neuronCount in neuronCounts)
            {
                var layer = new List<object>();

                if (first == true)
                {
                    layer.Add(new { CircleSize = circleSize, Content = "X1" });
                    layer.Add(new { CircleSize = circleSize, Content = "X2" });
                    first = false;
                }
                else
                {
                    for (int i = 0; i < neuronCount; i++)
                    {
                        layer.Add(new { CircleSize = circleSize });
                    }
                }

                layers.Add(new { RowCount = neuronCount, ColumnCount = 1, Neurons = layer });
            }

            CirclesContainer.ItemsSource = layers;
        }

        private void CreateWeightLines()
        {
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

        private void MainWindow_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            CreateWeightLines();
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            CreateWeightLines();
        }

        private void Nn_PlotUpdated(object? sender, EventArgs e, NeuralNetwork nn)
        {
            // Update the plot view
            Dispatcher.Invoke(() =>
            {
                plotView.Model = nn.PlotData();
                plotView.InvalidatePlot(true);

                accuracyPlotView.Model = nn.LossAccuracyPlotData();
                accuracyPlotView.InvalidatePlot(true);
            });
        }

        private async Task TrainAsync()
        {
            await Task.Run(() => nn.Train());

            // Update the UI on the main thread after training has finished
            Dispatcher.Invoke(() =>
            {
                plotView.Model = nn.PlotData();
            });
        }


        private async void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            // Stop the previous training, if any
            if (nn != null)
            {
                nn.StopTraining = true;
            }

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

            string activation;

            if (SigmoidRadioButton.IsChecked == true)
            {
                activation = "sigmoid";
            }
            else
            {
                activation = "relu";
            }

            nn = new NeuralNetwork(numNeurons, activation);
            CreateNeurons(numNeurons);
            CreateWeightLines();
            // Subscribe to plot update event
            nn.PlotUpdated += (sender, e) => Nn_PlotUpdated(sender, e, nn);

            Random random = new Random();

            int data = random.Next(0, 4);

            if (data == 0) 
            {
                nn.LoadMoonData(noise: 0.4);
            }
            else if (data == 1)
            {
                nn.LoadXORData();
            }
            else if (data == 2)
            {
                nn.LoadCircleData();
            }
            else
            {
                nn.LoadSpiralData();
            }
            

            nn.StopTraining = false;

            await TrainAsync();
        }
    }
}
