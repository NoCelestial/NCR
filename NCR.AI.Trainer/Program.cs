using System;

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


        }
    }
}
