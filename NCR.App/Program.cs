using System;
using System.Diagnostics;
using NCR.AI.Model;

namespace NCR.App
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine("Program Is Starting ...");

            Process.Start(Environment.CurrentDirectory + @"\Trainer\NCR.AI.Trainer.exe");

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("\nPlease Enter News Headers \nIf Input Empty Default Headers Is Run");

            string[] headers = new string[]
            {
                "Stocks move lower on discouraging news from space",
                "Respawn: Patch to increase game resolution likely",
                "'Some movie' sequel is in the works",
                "Measles Outbreak In Some County"
            };
            var cunsumeModel = new ConsumeModel(Environment.CurrentDirectory+ @"\Trainer\Model\NewsClassificationModel.zip");
            foreach (string header in headers)
            {
                var CalculatedResult = cunsumeModel.Predict(header);

                Console.ForegroundColor = ConsoleColor.DarkBlue;
                Console.WriteLine($"\n[{CalculatedResult}] : {header}");
                
            }

            Console.ResetColor();
        }
    }
}
