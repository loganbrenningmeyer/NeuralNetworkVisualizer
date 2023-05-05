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

            foreach (var neuronCount in neuronCounts)
            {
                var layer = new List<object>();

                for (int i = 0; i < neuronCount; i++)
                {
                    layer.Add(new { CircleSize = circleSize });
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
                            X1 = (double)90 + (layerIndex) * (CirclesContainer.ActualWidth - 120) / (numNeurons.Length - 1),
                            X2 = (double)90 + (layerIndex + 1) * (CirclesContainer.ActualWidth - 120) / (numNeurons.Length - 1),
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

            Random random = new Random();

            int numLayers = random.Next(4, 8);

            numNeurons = new int[numLayers];
            numNeurons[0] = 2;

            for (int i = 1; i < numLayers-1; i++)
            {
                numNeurons[i] = random.Next(2, 7);
            }

            numNeurons[numLayers-1] = 1;

            string activation = "sigmoid";
            nn = new NeuralNetwork(numNeurons, activation);
            CreateNeurons(numNeurons);
            CreateWeightLines();
            // Subscribe to plot update event
            nn.PlotUpdated += (sender, e) => Nn_PlotUpdated(sender, e, nn);

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
