using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace NCR.AI.Model
{
    public class ConsumeModel
    {
        private string _modelPath;

        public ConsumeModel(string modelPath)
        {
            _modelPath = modelPath;
        }
        //TODO
        public string Predict(string newsTitle)
        {
            DataViewSchema modelSchema;
            var context = new MLContext(0);
            var model = context.Model.Load(_modelPath,out modelSchema);
            var predict = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var predictResult = predict.Predict(new ModelInput() { Title = newsTitle, Category = "Medical" }).Category;
            switch (predictResult)
            {
                case "b":
                    return "Business";
                case "e":
                    return "Entertainment";
                case "m":
                    return "Medical";
                case "t":
                    return "Technology";
            }

            return "Amir";
        }
    } 
}
