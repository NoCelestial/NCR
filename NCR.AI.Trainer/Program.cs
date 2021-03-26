using System;
using System.IO;
using Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using NCR.AI.Model;

namespace NCR.AI.Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine("News Classification Trainer Started");

            FindTheBestModel();


            Console.ResetColor();
        }

        private static void FindTheBestModel()
        {
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("\nFinding the Best Model Using AutoML");

            var mlContext = new MLContext(0);

            string trainDataPath = @"Data\uci-news-aggregator.csv";
            string trainCachePath = @"Cache\";

            var trainDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                trainDataPath,
                hasHeader:true,
                separatorChar:',',
                allowQuoting:true
                );
            var preProcessingPipeline = mlContext.Transforms
                .Conversion.MapValueToKey("Category","Category");
            var mappedInputData = preProcessingPipeline
                .Fit(trainDataView).Transform(trainDataView);
            var experimentSetting = new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 300,
                CacheBeforeTrainer = CacheBeforeTrainer.On,
                OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                CacheDirectory = new DirectoryInfo(trainCachePath)
            };

            var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSetting);
            var experimentResult = experiment.Execute(
                trainData:mappedInputData,
                labelColumnName:"Category",
                progressHandler:new MulticlassExperimentProgressHandler()
                );

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Metrics From Best Run ... ");

            var metrics = experimentResult.BestRun.ValidationMetrics;

            Console.ForegroundColor = ConsoleColor.DarkBlue;
            Console.WriteLine($"Metric Micro Accuracy : {metrics.MicroAccuracy:0.##}");
            Console.WriteLine($"Metric Micro Accuracy : {metrics.MicroAccuracy:0.##}");

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Success !");
        }
    }
}
