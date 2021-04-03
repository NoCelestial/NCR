using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace NCR.AI.Model
{
    public class ModelInput
    {
        [LoadColumn(1)]
        public string Title { get; set; }
    }
}
