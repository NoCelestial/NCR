using System;
using System.Diagnostics;
using System.Threading.Tasks;
using NCR.AI.Model;

namespace NCR.App
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine("Program Is Starting ...");

            //await RunProcessAsync();

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("\nPlease Enter News Headers \nIf Input Empty Default Headers Is Run");

            string[] headers = new string[]
            {
                "Stocks move lower on discouraging news from space",
                "Respawn: Patch to increase game resolution likely",
                "'Some movie' sequel is in the works",
                "Measles Outbreak In Some County",
                " Cough In Throat .Net 5.02 Have"
            };
            var cunsumeModel = new ConsumeModel(@"F:\Project\Testing\NCR\NCR.App\bin\Debug\net5.0\Model\NewsClassificationModel.zip");
            foreach (string header in headers)
            {
                var CalculatedResult = cunsumeModel.Predict(header);

                Console.ForegroundColor = ConsoleColor.DarkBlue;
                Console.WriteLine($"\n[{CalculatedResult}] : {header}");
                
            }

            Console.ResetColor();
        }

        static Task<int> RunProcessAsync()
        {
            var tcs = new TaskCompletionSource<int>();

            var process = new Process
            {
                StartInfo = { FileName =Environment.CurrentDirectory+ @"\Trainer\NCR.AI.Trainer.exe" },
                EnableRaisingEvents = true
            };

            process.Exited += (sender, args) =>
            {
                tcs.SetResult(process.ExitCode);
                process.Dispose();
            };

            process.Start();

            return tcs.Task;
        }
    }
}
