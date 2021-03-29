using System;
using System.IO;
using System.Linq;
using Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using NCR.AI.Model;
using SHA.BeautifulConsoleColor.Core.Class;
using SHA.BeautifulConsoleColor.Core.Extension;
using SHA.BeautifulConsoleColor.Core.Model;

namespace NCR.AI.Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            BCCConsole.Write(BCCConsoleColor.Blue,false, "News Classification Trainer Started");

            //FindTheBestModel();
            TrainModel();
        }

        private static void TrainModel()
        {
            BCCConsole.Write(BCCConsoleColor.Gray,false,"Trainer Base Is Started ...");
            var mlContext = new MLContext(0);
            string trainDataPath = "Data\\uci-news-aggregator.csv";
            string trainCachePath = @"Cache\";
            var trainDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                trainDataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true
            );
            var preProcessingPipeline = mlContext.Transforms
                .Conversion.MapValueToKey("Label","Category")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title",outputColumnName: "Features"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);
            var trainer = mlContext.MulticlassClassification.Trainers
                .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());
            var trainingPipeline = preProcessingPipeline
                .Append(trainer)
                .Append(
                    mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")
                    );
            BCCConsole.Write(BCCConsoleColor.Cyan,false,"Cross Validation Is Starting . . .");
            var cvResult = mlContext.MulticlassClassification
                .CrossValidate(trainDataView, trainingPipeline);
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false,"\n",
                "Cross Validation Result Metrics",
                "-----------------------------------");
            var micA = cvResult.Average(m => m.Metrics.MicroAccuracy).ToString("0.###").DarkGreen();
            var macA = cvResult.Average(m => m.Metrics.MacroAccuracy).ToString("0.###").DarkGreen();
            var logA = cvResult.Average(m => m.Metrics.LogLossReduction).ToString("0.###").DarkGreen();
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false, "-----------------------------------");
            var finalModel = trainingPipeline.Fit(trainDataView);
            var modelPath = "Model\\NewsClassificationModel.zip";
            if (!Directory.Exists("Model\\"))
            {
                Directory.CreateDirectory("Model\\");
            }
            BCCConsole.Write(BCCConsoleColor.Yellow,false,"Saving Model . . .");
            mlContext.Model.Save(finalModel,trainDataView.Schema,modelPath);
            BCCConsole.Write(BCCConsoleColor.Green,false,"Saved !");
        }

        private static void FindTheBestModel()
        {
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false, "\nFinding the Best Model Using AutoML");
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
            BCCConsole.Write(BCCConsoleColor.Yellow,false, "Metrics From Best Run ... ");
            var metrics = experimentResult.BestRun.ValidationMetrics;
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false, $"Metric Micro Accuracy : {metrics.MicroAccuracy:0.##}"); 
            BCCConsole.Write(BCCConsoleColor.Green,false,"Success !");
        }
    }
}
